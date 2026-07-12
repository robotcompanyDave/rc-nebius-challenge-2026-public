"""
Behaviour tree for pick+sort — the decision layer over GraspSortController.

The controller stays what it is (a well-tuned per-pick sequencer driven through
`begin_pick_place` / `step`); this module adds the piece the platform's
autonomous sorter has and `run_sort_trial` lacks: **strategy escalation across
retries** (direct → tilt → lip, the SORT_TRAINING.md ladder), scorer-guided
candidate choice WITHIN the forced strategy, clutter-aware yaw for washers, and
a per-attempt trace so failures are diagnosable.

Non-blocking by design: `SortBT.tick()` is called once per control step and
never blocks, so the same tree drops into `parallel_env`'s
tick-all-cells-then-step-once loop (M5). Sequential callers use `run()`.

    bt = SortBT(env, ctrl, scorer=None)      # scorer: GraspScorer | None
    result = bt.run()                        # SortResult (controller vocab)
    bt.save_trace("data/<date>/<hhmm>-bt/trace.jsonl")

Env knobs: GS_BT_MAX_PICKS (12), GS_BT_ATTEMPTS (3 per part, one ladder rung
each), GS_BT_CANDS (24 scorer candidates per attempt), GS_BT_VERBOSE (1).
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional

import numpy as np

from .controller import SortResult

MAX_PICKS = int(os.environ.get("GS_BT_MAX_PICKS", "12"))
ATTEMPTS_PER_PART = int(os.environ.get("GS_BT_ATTEMPTS", "3"))
N_CANDS = int(os.environ.get("GS_BT_CANDS", "24"))
VERBOSE = os.environ.get("GS_BT_VERBOSE", "1") != "0"

# the platform sorter's escalation ladder (SORT_TRAINING.md): each retry of a
# part climbs one rung. lip degrades to tilt automatically without a soft rig.
# Washers get the press-lift pick (stage-1-proven lip physics, MILESTONES M3a)
# straight after the direct try — tilt/lip measured ~0% on flat washers.
STRATEGY_LADDER = ("direct", "tilt", "lip")
LADDER_BY_KIND = {"washer": ("direct", "press_lift", "press_lift")}


def ladder_for(kind: str):
    return LADDER_BY_KIND.get(kind, STRATEGY_LADDER)


class Status(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


@dataclass
class Blackboard:
    env: object
    ctrl: object
    scorer: object = None                    # GraspScorer or None → heuristic
    rng: random.Random = field(default_factory=lambda: random.Random(0))
    part: Optional[str] = None
    kind: str = "nut"
    attempt: int = 0                         # attempt # for the CURRENT part
    candidate: Optional[dict] = None
    picks: int = 0
    attempts_by_part: dict = field(default_factory=dict)
    skip: set = field(default_factory=set)
    trace: list = field(default_factory=list)

    def log(self, node: str, event: str, **detail):
        rec = {"t": round(time.monotonic(), 3), "node": node, "event": event,
               "part": self.part, "attempt": self.attempt, **detail}
        self.trace.append(rec)
        if VERBOSE and event in ("enter", "success", "failure"):
            extra = " ".join(f"{k}={v}" for k, v in detail.items())
            print(f"[bt] {node}:{event} part={_short(self.part)} "
                  f"attempt={self.attempt} {extra}", flush=True)


def _short(path):
    return path.rsplit("/", 1)[-1] if path else None


# ── nodes ────────────────────────────────────────────────────────────────────
class Node:
    name = "node"

    def tick(self, bb: Blackboard) -> Status:
        raise NotImplementedError

    def reset(self):
        pass


class Sequence(Node):
    """Tick children in order; FAILURE/RUNNING short-circuit. Remembers the
    running child so composites resume instead of restarting."""

    def __init__(self, name: str, children: List[Node]):
        self.name = name
        self.children = children
        self._idx = 0

    def tick(self, bb):
        while self._idx < len(self.children):
            s = self.children[self._idx].tick(bb)
            if s == Status.RUNNING:
                return s
            if s == Status.FAILURE:
                self.reset()
                return s
            self._idx += 1
        self.reset()
        return Status.SUCCESS

    def reset(self):
        self._idx = 0
        for c in self.children:
            c.reset()


class Action(Node):
    """Leaf around a fn(bb) -> Status."""

    def __init__(self, name: str, fn: Callable[[Blackboard], Status]):
        self.name = name
        self.fn = fn

    def tick(self, bb):
        return self.fn(bb)


# ── leaves ───────────────────────────────────────────────────────────────────
class SelectTarget(Node):
    """Next misplaced/unsorted part not yet given up on → bb.part. FAILURE when
    nothing is left to do (the sort is as done as it will get)."""
    name = "select_target"

    def tick(self, bb):
        ctrl = bb.ctrl
        cands = [p for p in ctrl._sort_candidates() if p not in bb.skip]
        while cands and bb.attempts_by_part.get(cands[0], 0) >= ATTEMPTS_PER_PART:
            bb.skip.add(cands[0])
            bb.log(self.name, "give_up", part_skipped=_short(cands[0]))
            cands.pop(0)
        if not cands or bb.picks >= MAX_PICKS:
            return Status.FAILURE
        bb.part = cands[0]
        bb.kind = bb.env.part_kinds.get(bb.part, "nut")
        bb.attempt = bb.attempts_by_part.get(bb.part, 0)
        bb.log(self.name, "enter", kind=bb.kind)
        return Status.SUCCESS


class ProposeGrasp(Node):
    """Choose the candidate for this attempt. The escalation ladder FORCES the
    strategy (attempt 0 direct, 1 tilt, 2 lip); the scorer (if any) picks the
    best xy/yaw/depth WITHIN that strategy from GS_BT_CANDS samples; washers
    get the clutter-aware yaw via the controller's nearest_free_dir."""
    name = "propose_grasp"

    def tick(self, bb):
        from . import randomize
        ctrl = bb.ctrl
        ladder = ladder_for(bb.kind)
        strategy = ladder[min(bb.attempt, len(ladder) - 1)]
        R = ctrl.grasp_R(bb.part, bb.kind)
        h_yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        base = {"xy_offset": (0.0, 0.0), "grasp_yaw": h_yaw, "grasp_dz": 0.0,
                "approach_dh": 0.12, "heuristic_yaw": h_yaw, "width": 1.0}
        if bb.scorer is None:
            cand = dict(base)
        else:
            spec = bb.env.part_specs.get(bb.part)
            part = {"kind": bb.kind,
                    "size": spec.size if spec else "m12",
                    "pose": spec.pose if spec else "flat"}
            cands = [dict(base)]
            for _ in range(N_CANDS):
                c = randomize.sample_candidate(bb.rng, bb.kind, heuristic_yaw=h_yaw)
                c["heuristic_yaw"] = h_yaw
                cands.append(c)
            for c in cands:
                c["strategy"] = strategy          # ladder overrides the sample
            probs = bb.scorer.score_batch(part, cands)
            cand = cands[int(probs.argmax())]
            bb.log(self.name, "scored", p_best=round(float(probs.max()), 3))
        cand["strategy"] = strategy
        if bb.kind == "washer":                   # clutter-aware yaw (platform)
            free = ctrl.nearest_free_dir(bb.part)
            if strategy == "press_lift" and bb.attempt >= 2:
                free = (-free[1], free[0])        # 2nd press try: rotate 90°
            cand["lead_dir"] = free
            cand["grasp_yaw"] = float(np.arctan2(free[1], free[0]))
        bb.candidate = cand
        bb.log(self.name, "enter", strategy=strategy)
        return Status.SUCCESS


class ExecutePickPlace(Node):
    """Drive the controller through one full pick→place. First tick arms the
    sequencer; subsequent ticks report RUNNING until it goes idle."""
    name = "pick_place"

    def __init__(self):
        self._armed = False

    def tick(self, bb):
        ctrl = bb.ctrl
        if not self._armed:
            bb.attempts_by_part[bb.part] = bb.attempts_by_part.get(bb.part, 0) + 1
            bb.picks += 1
            ctrl.begin_pick_place(bb.part, bb.candidate)
            self._armed = True
            return Status.RUNNING
        if ctrl.busy:
            return Status.RUNNING
        self._armed = False
        out = ctrl.outcome
        ok = bool(out and out.success)
        bb.log(self.name, "success" if ok else "failure",
               strategy=bb.candidate.get("strategy"),
               reason=None if ok else (out.fail_reason if out else "no_outcome"),
               force_N=round(out.grasp_force_N, 1) if out else None)
        return Status.SUCCESS if ok else Status.FAILURE

    def reset(self):
        self._armed = False


class VerifyZone(Node):
    """The pick reported success — confirm the part actually sits in its lane."""
    name = "verify_zone"

    def tick(self, bb):
        ctrl = bb.ctrl
        pose = ctrl.part_world_pose(bb.part)
        if pose is None:
            bb.log(self.name, "failure", reason="part_vanished")
            return Status.FAILURE
        want = ctrl.zone_for_kind(bb.kind)
        zone = ctrl.part_zone(float(pose[0, 3]), float(pose[1, 3]))
        ok = zone == want
        bb.log(self.name, "success" if ok else "failure", zone=zone, want=want)
        return Status.SUCCESS if ok else Status.FAILURE


class ResetHome(Node):
    """Non-blocking settle back to HOME between picks."""
    name = "reset_home"

    def __init__(self, settle_ticks: int = 20):
        self.settle_ticks = settle_ticks
        self._left = -1

    def tick(self, bb):
        if self._left < 0:
            bb.ctrl.reset_to_home()
            self._left = self.settle_ticks
        if self._left > 0:
            self._left -= 1
            return Status.RUNNING
        self._left = -1
        return Status.SUCCESS

    def reset(self):
        self._left = -1


# ── the tree ─────────────────────────────────────────────────────────────────
class SortBT:
    """Sort every part into its lane with strategy escalation.

    tick() → RUNNING while there is work, SUCCESS/FAILURE when done (SUCCESS =
    nothing left misplaced; FAILURE = gave up on ≥1 part or hit the pick cap —
    the SortResult tally is the real metric either way)."""

    def __init__(self, env, ctrl, scorer=None, seed: int = 0):
        self.bb = Blackboard(env=env, ctrl=ctrl, scorer=scorer,
                             rng=random.Random(seed))
        self._select = SelectTarget()
        self._pick = Sequence("pick_attempt", [
            ProposeGrasp(), ExecutePickPlace(), VerifyZone(),
        ])
        self._rest = ResetHome()                 # ALWAYS runs between attempts,
        self._resting = False                    # even after a failed pick
        self._done: Optional[Status] = None

    def tick(self) -> Status:
        if self._done is not None:
            return self._done
        bb = self.bb
        if self._resting:
            if self._rest.tick(bb) == Status.RUNNING:
                return Status.RUNNING
            self._resting = False
            bb.part = None                       # next tick re-selects — a failed
            return Status.RUNNING                # part re-enters one rung higher
        if bb.part is None:
            s = self._select.tick(bb)
            if s == Status.FAILURE:              # nothing left to try
                self._done = (Status.SUCCESS if not bb.ctrl._sort_candidates()
                              else Status.FAILURE)
                bb.log("sort_bt", "success" if self._done == Status.SUCCESS
                       else "failure", picks=bb.picks)
                return self._done
            # fall through: a part is selected, start its attempt this tick
        s = self._pick.tick(bb)
        if s == Status.RUNNING:
            return Status.RUNNING
        self._pick.reset()
        self._resting = True                     # settle home before re-selecting
        return Status.RUNNING

    # ── sequential convenience ───────────────────────────────────────────────
    def run(self, max_ticks: int = 40000) -> SortResult:
        env, ctrl = self.bb.env, self.bb.ctrl
        for _ in range(max_ticks):
            if self.tick() != Status.RUNNING:
                break
            ctrl.step()
            env.step(render=False)
        return self.result()

    def result(self) -> SortResult:
        env, ctrl = self.bb.env, self.bb.ctrl
        res = SortResult(n_parts=len(env.part_paths))
        res.attempts = self.bb.picks
        for p in env.part_paths:
            pose = ctrl.part_world_pose(p)
            if pose is None:
                continue
            kind = env.part_kinds.get(p, "nut")
            want = ctrl.zone_for_kind(kind)
            zone = ctrl.part_zone(float(pose[0, 3]), float(pose[1, 3]))
            ok = zone == want
            res.n_correct += int(ok)
            res.per_part.append({"part": p, "kind": kind, "zone": zone,
                                 "correct": ok})
        return res

    def save_trace(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for rec in self.bb.trace:
                f.write(json.dumps(rec) + "\n")
