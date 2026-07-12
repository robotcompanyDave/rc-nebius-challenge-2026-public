"""
Per-attempt dataset record + writer for the supervised grasp-success scorer.

One JSONL line per grasp attempt (append-safe / crash-resilient), PNGs for the
images, and an optional Parquet roll-up at job end for efficient training loads.
Pure-python (no Isaac/USD) so it imports anywhere.

Record layout (one grasp attempt):
    attempt_id, seed, batch, ts_step
    obs:    pre-grasp observation the scorer conditions on
            { topdown_rgb, topdown_depth, eih_rgb, eih_depth (relative PNG paths),
              parts: [ {kind, size, pose: [x,y,z,qw,qx,qy,qz], bbox: [dx,dy,dz]} ] }
    action: candidate grasp that was executed
            { grasp_pos: [x,y,z], grasp_yaw, width, approach_h, xy_offset, grasp_dz }
    outcome:{ success, lifted_mm, grasp_force_N, clamp_openness, slip_mm,
              sort_zone_correct, fail_reason }
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, asdict, field
from typing import Optional


def ensure_dir(d: str):
    """makedirs that tolerates an already-existing mount point. mountpoint-s3
    (an S3 bucket mounted into a Nebius Job at e.g. /data) raises FileExistsError
    on the mount root even with exist_ok=True, so swallow that one case."""
    try:
        os.makedirs(d, exist_ok=True)
    except FileExistsError:
        pass


def _s3_target(dest: str):
    """If dest is an s3://bucket[/prefix] URI return (bucket, prefix), else None."""
    if not dest.startswith("s3://"):
        return None
    bucket, _, prefix = dest[len("s3://"):].partition("/")
    return bucket, prefix.strip("/")


def _s3_client():
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("GS_S3_ENDPOINT", "https://storage.eu-north1.nebius.cloud"),
        region_name=os.environ.get("GS_S3_REGION", "eu-north1"),
    )


def emit_dir(local_dir: str, dest: str) -> str:
    """Publish every file under local_dir to `dest`, which is either a local path
    or an s3://bucket/prefix URI. We upload via boto3 rather than relying on a
    mounted bucket because Nebius mounts the --volume bucket effectively read-only
    (mountpoint-s3 → EACCES on write), whereas a static-key S3 client writes fine."""
    s3 = _s3_target(dest)
    if s3 is None:                                   # local sink
        ensure_dir(dest)
        for name in sorted(os.listdir(local_dir)):
            src = os.path.join(local_dir, name)
            if os.path.isdir(src):
                files = os.listdir(src)
                if files:
                    ensure_dir(os.path.join(dest, name))
                    for f in files:
                        shutil.copyfile(os.path.join(src, f), os.path.join(dest, name, f))
            else:
                shutil.copyfile(src, os.path.join(dest, name))
        return dest
    bucket, prefix = s3                              # S3 sink
    cli = _s3_client()
    for root, _dirs, files in os.walk(local_dir):
        for f in files:
            lp = os.path.join(root, f)
            rel = os.path.relpath(lp, local_dir).replace(os.sep, "/")
            key = f"{prefix}/{rel}" if prefix else rel
            cli.upload_file(lp, bucket, key)
    return dest


@dataclass
class AttemptRecord:
    attempt_id: str
    seed: int
    batch: int = 0
    ts_step: int = 0
    obs: dict = field(default_factory=dict)
    action: dict = field(default_factory=dict)
    outcome: dict = field(default_factory=dict)


class DatasetWriter:
    """Streams records to <stage>/records.jsonl + images to <stage>/img/, then
    copies the finished files to `final_dir` via flush_to_final().

    Staging is a LOCAL temp dir because the final_dir is typically an S3 bucket
    mounted with mountpoint-s3, which supports neither append-mode writes nor
    reading a file back before its write handle closes. We therefore do all the
    incremental/append + read-back work on local disk and emit each finished file
    to the mount with a single sequential copy (which mountpoint-s3 does support)."""

    def __init__(self, out_dir: str):
        self.final_dir = out_dir
        self.out_dir = tempfile.mkdtemp(prefix="gs_ds_")   # local staging
        self.img_dir = os.path.join(self.out_dir, "img")
        os.makedirs(self.img_dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.out_dir, "records.jsonl")
        self._fh = open(self.jsonl_path, "a", buffering=1)   # line-buffered
        self.n = 0

    def save_image(self, name: str, array) -> Optional[str]:
        """Write an HxWx{3,1} uint8/float array as a PNG; return its path relative
        to out_dir. None array → None (e.g. depth disabled)."""
        if array is None:
            return None
        import numpy as np
        import cv2
        a = np.asarray(array)
        if a.dtype != np.uint8:
            # depth or float rgb → normalise to 8-bit for a viewable PNG
            finite = np.isfinite(a)
            if finite.any():
                lo, hi = float(a[finite].min()), float(a[finite].max())
                a = (255.0 * (np.clip(a, lo, hi) - lo) / (hi - lo + 1e-9)).astype(np.uint8)
            else:
                a = np.zeros(a.shape, np.uint8)
        if a.ndim == 3 and a.shape[2] == 3:
            a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
        rel = os.path.join("img", name)
        cv2.imwrite(os.path.join(self.out_dir, rel), a)
        return rel

    def write(self, rec: AttemptRecord):
        self._fh.write(json.dumps(asdict(rec)) + "\n")
        self.n += 1

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass

    def flush_to_final(self) -> str:
        """Publish staged outputs (records.jsonl, records.parquet, img/*) to
        final_dir — a local path or an s3:// URI (see emit_dir). Safe to call
        repeatedly mid-run for crash-resilience; call after to_parquet()."""
        return emit_dir(self.out_dir, self.final_dir)

    def to_parquet(self) -> Optional[str]:
        """Roll the JSONL up into records.parquet (flattened outcome columns)."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            return None
        rows = []
        with open(self.jsonl_path) as f:
            for line in f:
                r = json.loads(line)
                flat = {"attempt_id": r["attempt_id"], "seed": r["seed"],
                        "batch": r.get("batch", 0)}
                for k, v in r.get("outcome", {}).items():
                    flat[f"outcome_{k}"] = v
                for k, v in r.get("action", {}).items():
                    if not isinstance(v, (list, dict)):
                        flat[f"action_{k}"] = v
                flat["obs"] = json.dumps(r.get("obs", {}))
                rows.append(flat)
        if not rows:
            return None
        cols = {k: [row.get(k) for row in rows] for k in rows[0]}
        out = os.path.join(self.out_dir, "records.parquet")
        pq.write_table(pa.table(cols), out)
        return out
