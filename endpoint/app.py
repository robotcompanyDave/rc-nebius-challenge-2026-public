"""
Soft-surface grasp micro-service — a CPU Nebius Serverless AI *Endpoint*.

Serves two pure-numpy deliverables from the rc-spike-soft-surface / push-grasp
spike, no GPU / no Isaac needed:

  • /score, /rank  — the trained grasp-success scorer (val AUC 0.823): given a
                     part + candidate grasp (+ scene clutter) → P(success).
  • /soft_surface  — the procedural compliant-surface sim (SoftPad): press the
                     work surface with fingertip "feet" → the dented height field
                     (the model behind the 13–15° washer tilt for the roll-up).

This is the qualifying "model-serving / inference endpoint" for the Nebius
Serverless AI Builders Challenge — the part of the robotics pipeline that runs
fine on CPU, so it dodges the GPU-pool capacity wall the Isaac Lab jobs hit.
"""
from __future__ import annotations
import os
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Any

from graspsort.scorer import GraspScorer
from graspsort.softsurface import SoftPad

MODEL_PATH = os.environ.get("MODEL_PATH", "model/model.npz")
SCORER = GraspScorer.load(MODEL_PATH)
META = SCORER.meta

app = FastAPI(title="soft-surface grasp micro-service", version="1.0")


# ---- schemas ---------------------------------------------------------------
class ScoreReq(BaseModel):
    part: dict[str, Any] = {"kind": "washer", "size": "m12", "pose": "flat"}
    action: dict[str, Any] = {"strategy": "tilt", "tilt_deg": 14, "xy_offset": [0.002, 0.0]}
    scene: dict[str, Any] | None = {"n_close": 1, "nearest_mm": 30}


class RankReq(BaseModel):
    part: dict[str, Any] = {"kind": "washer", "size": "m12", "pose": "flat"}
    scene: dict[str, Any] | None = None
    actions: list[dict[str, Any]]


class Foot(BaseModel):
    x: float; y: float; half_w: float = 0.004; pen: float = 0.0026


class SoftReq(BaseModel):
    size_m: float = 0.06          # square patch side
    n: int = 41                   # grid resolution per side
    rest_z: float = 0.0
    feet: list[Foot] = [Foot(x=0.006, y=0.0)]
    spread_base_m: float = 0.002
    spread_gain: float = 1.5
    max_indent_m: float = 0.010


# ---- routes ----------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_val_auc": META.get("val_auc"),
            "n_features": len(SCORER.params["feature_names"]), "hidden": META.get("hidden")}


@app.post("/score")
def score(r: ScoreReq):
    p = SCORER.score(r.part, r.action, r.scene)
    return {"p_success": round(float(p), 4), "part": r.part, "action": r.action}


@app.post("/rank")
def rank(r: RankReq):
    probs = SCORER.score_batch(r.part, r.actions, r.scene)
    ranked = sorted(
        ({"action": a, "p_success": round(float(p), 4)} for a, p in zip(r.actions, probs)),
        key=lambda d: d["p_success"], reverse=True)
    return {"part": r.part, "best": ranked[0] if ranked else None, "ranked": ranked}


@app.post("/soft_surface")
def soft_surface(r: SoftReq):
    pad = SoftPad(spread_base_m=r.spread_base_m, spread_gain=r.spread_gain,
                  max_indent_m=r.max_indent_m)
    ax = np.linspace(-r.size_m / 2, r.size_m / 2, r.n)
    X, Y = np.meshgrid(ax, ax)
    feet = [(f.x, f.y, f.half_w, f.pen) for f in r.feet]
    Z = pad.target_field(X, Y, r.rest_z, feet)
    dent = r.rest_z - Z
    return {
        "grid": {"n": r.n, "size_m": r.size_m},
        "max_dent_m": round(float(dent.max()), 6),
        "mean_dent_m": round(float(dent.mean()), 6),
        "dent_area_cells": int((dent > 1e-6).sum()),
        "spread_m": round(pad.spread_of(max(f.pen for f in r.feet)), 6),
        # coarse 11x11 view of the height field (mm), so a client can plot it
        "z_mm_coarse": np.round(Z[::max(1, r.n // 11), ::max(1, r.n // 11)] * 1000, 3).tolist(),
    }


@app.get("/", response_class=HTMLResponse)
def index():
    auc = META.get("val_auc")
    return f"""<html><body style="font:15px system-ui;max-width:760px;margin:40px auto;color:#222">
<h1>soft-surface grasp micro-service</h1>
<p>A CPU Nebius Serverless Endpoint serving two pure-numpy pieces of the
<b>rc-spike-soft-surface / push-grasp</b> robotics spike — no GPU, no Isaac.</p>
<ul>
<li><b>POST /score</b> — grasp-success probability from the trained scorer
    (val AUC <b>{auc:.3f}</b>, 26-feature numpy MLP). Body: <code>{{part, action, scene?}}</code></li>
<li><b>POST /rank</b> — rank many candidate grasps for one part. Body: <code>{{part, scene?, actions:[...]}}</code></li>
<li><b>POST /soft_surface</b> — procedural compliant-surface sim (SoftPad): press → dent height-field.
    Body: <code>{{size_m, n, feet:[{{x,y,half_w,pen}}]}}</code></li>
<li><b>GET /health</b> — model metadata</li>
</ul>
<pre>curl -sX POST $URL/score -H 'content-type: application/json' \\
  -d '{{"part":{{"kind":"washer","size":"m12","pose":"flat"}},
       "action":{{"strategy":"tilt","tilt_deg":14,"xy_offset":[0.002,0.0]}},
       "scene":{{"n_close":1,"nearest_mm":30}}}}'</pre>
</body></html>"""
