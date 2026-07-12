#!/usr/bin/env python3
"""
arena_eval — evaluate or FINE-TUNE a pick_lab θ inside the DEPLOY env.

pick_lab trains in its own standalone World; θ that scores 6/6 there can still
fail in the full GraspSortEnv (the lab↔env transfer gap: arm articulation in
the scene, different solver load, different bed span). This tool closes the
loop by scoring/optimizing θ in the env itself — no rendering, so it's fast
and it runs headless on Nebius (gen_data already proved the env boots on H100).

Modes (GS_AE_MODE):
  eval  one fixed θ, RIGS×ROUNDS episodes over a deterministic jitter panel
        + per-rig roll spread. Reports success / snaps / worst-case.
  tune  CEM warm-started from GS_AE_THETA, random jitter per eval (same recipe
        as pick_lab SEARCH), per-round crash-safe MU/BEST θ prints, and a final
        deterministic eval round of MU.

Env:
  GS_AE_MODE (eval) GS_AE_RIGS (6) GS_AE_COLS (3) GS_AE_SPACING (0.15)
  GS_AE_ROUNDS (3 eval / 10 tune)  GS_AE_THETA  GS_AE_MAT
  GS_AE_JIT_XY (0.0015) GS_AE_JIT_DZ (0.0005) GS_AE_ROLL (20)
  GS_AE_SPAN (0.05)  GS_AE_SIG (0.08)  GS_AE_SEED (0)  GS_AE_OUT

Memory note: the rig harness leaks ~per rig-round; keep RIGS×(ROUNDS+1) ≲ 90
per process (chain warm-started runs for more depth).
"""
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

MODE = os.environ.get("GS_AE_MODE", "eval")
RIGS = int(os.environ.get("GS_AE_RIGS", "6"))
COLS = int(os.environ.get("GS_AE_COLS", "3"))
SPACING = float(os.environ.get("GS_AE_SPACING", "0.15"))
ROUNDS = int(os.environ.get("GS_AE_ROUNDS", "3" if MODE == "eval" else "10"))
JIT_XY = float(os.environ.get("GS_AE_JIT_XY", "0.0015"))
JIT_DZ = float(os.environ.get("GS_AE_JIT_DZ", "0.0005"))
ROLL = float(os.environ.get("GS_AE_ROLL", "20"))
SIG0 = float(os.environ.get("GS_AE_SIG", "0.08"))
SEED = int(os.environ.get("GS_AE_SEED", "0"))
SNAP_PEN = float(os.environ.get("GS_AE_SNAP_PEN", "0.0"))
# per-episode MATERIAL noise (sourcing tolerance), e.g. "0.85,1.18"; "1,1"=off
MATK = [float(v) for v in os.environ.get("GS_AE_MATK", "1,1").split(",")]
MATC = [float(v) for v in os.environ.get("GS_AE_MATC", "1,1").split(",")]
MATCP = [float(v) for v in os.environ.get("GS_AE_MATCP", "1,1").split(",")]


def _dseed(*parts):
    import zlib
    return zlib.crc32(repr(parts).encode()) & 0x7FFFFFFF
# pick_lab reads GS_PL_SPAN at import time — forward our span choice first.
os.environ["GS_PL_SPAN"] = os.environ.get("GS_AE_SPAN", "0.05")
_now = datetime.datetime.now()
OUT = os.environ.get("GS_AE_OUT", os.path.join(
    "data", _now.strftime("%Y-%m-%d"), _now.strftime("%H%M") + "-arena_eval"))

DEF_MAT = ('{"name":"PARKER_d24","stiffness":300,"damping":24,'
           '"couple":3000,"couple_damp":8,"falloff":0,"cutoff":0.1,'
           '"leveling":true,"btip":"square","mode":"parallel"}')


def _panel(rnd, r):
    """Deterministic eval jitter: rig r, round rnd -> (dx, dy, dz)."""
    entries = [(0.0, 0.0, 0.0),
               (+JIT_XY, 0.0, 0.0), (-JIT_XY, 0.0, 0.0),
               (0.0, +JIT_XY, 0.0), (0.0, -JIT_XY, 0.0),
               (0.0, 0.0, +JIT_DZ), (0.0, 0.0, -JIT_DZ),
               (0.7 * JIT_XY, 0.7 * JIT_XY, -JIT_DZ)]
    return entries[(r + rnd * 3) % len(entries)]


def main():
    os.environ.setdefault("GS_SOFT", "0")
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    from graspsort.sim_env import GraspSortEnv
    env = GraspSortEnv(headless=True, width=320, height=240)
    env.world.get_physics_context().set_physics_dt(1.0 / 240.0)
    env.reset_world()
    stage = env.stage

    import tools.pick_lab as PL
    from pxr import PhysxSchema
    wc = env._work_centre_from_stage()
    PL.SURF_Z = env.table_top_z + 0.030

    theta0 = np.array([float(v) for v in os.environ["GS_AE_THETA"].split(",")])
    if theta0.size < PL.THP_MU0.size:      # pad older θ (e.g. no yaw) with MU0
        theta0 = np.concatenate([theta0, PL.THP_MU0[theta0.size:]])
    mat = json.loads(os.environ.get("GS_AE_MAT", DEF_MAT))
    rows = (RIGS + COLS - 1) // COLS
    rolls = ([0.0] if RIGS == 1 else
             [round(float(v), 3) for v in np.linspace(-ROLL, ROLL, RIGS)])
    budget = RIGS * (ROUNDS + (2 if MODE == "tune" else 0))
    print(f"[ae] mode={MODE} rigs={RIGS} rounds={ROUNDS} span="
          f"{os.environ['GS_PL_SPAN']} jit_xy={JIT_XY} roll=+-{ROLL}", flush=True)
    print(f"[ae] rig rolls: {rolls}", flush=True)
    print(f"[ae] theta0: {[round(float(v), 4) for v in theta0]}", flush=True)
    if budget > 90:
        print(f"[ae] WARNING rig-rounds {budget} > 90 leak ceiling", flush=True)

    def rig_c(r):
        i, j = divmod(r, COLS)
        return (wc[0] + (j - (COLS - 1) / 2) * SPACING,
                wc[1] + (i - (rows - 1) / 2) * SPACING)

    rigs = []
    for r in range(RIGS):
        rig = PL.PickRig(stage, r, rig_c(r), mat, roll_deg=rolls[r])
        rig.base = f"/World/AERig{r}"
        rig.build()
        for pth in (rig.base + "/FingerB", rig.wpath):
            prb = PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(pth))
            prb.CreateSolverPositionIterationCountAttr().Set(32)
            prb.CreateSolverVelocityIterationCountAttr().Set(8)
        rigs.append(rig)
    env.world.reset()

    LO, HI = PL.THP_LO, PL.THP_HI
    mu = np.clip(theta0.copy(), LO, HI)
    sig = SIG0 * (HI - LO)
    best = dict(reward=-9.0)
    log = open(os.path.join(OUT, "evals.jsonl"), "w")
    t_total = PL.T_TOTAL_P + PL.T_CARRY + 0.2
    steps = int(t_total / PL.PHYS_DT)
    tally = []            # eval mode: (rig, roll, jitter, score)

    def run_round(rnd, ths, jits, tag):
        import time as _tm
        tw0 = _tm.time()
        for r, rig in enumerate(rigs):
            if MATK != [1, 1] or MATC != [1, 1] or MATCP != [1, 1]:
                mr = np.random.default_rng(_dseed("mat", SEED, rnd, r))
                rig.found.retune(mr.uniform(*MATK), mr.uniform(*MATC),
                                 mr.uniform(*MATCP))
            rig.reset(np.asarray(ths[r]), jitter=tuple(jits[r]))
        for s in range(steps):
            t = s * PL.PHYS_DT
            for rig in rigs:
                rig.drive(t)              # every physics step (env-validated)
            env.step(render=False)
            if t > PL.T_SET and s % 12 == 0:
                for rig in rigs:
                    rig.observe(t)
        scores = [rig.final_score() for rig in rigs]
        succ = sum(1 for x in scores if x["success"])
        snaps = sum(1 for x in scores if x.get("snap"))
        rews = [x["reward"] for x in scores]
        print(f"[ae] {tag} rnd{rnd}: succ={succ}/{RIGS} snaps={snaps} "
              f"mean={np.mean(rews):.2f} worst={np.min(rews):.2f} "
              f"({_tm.time()-tw0:.0f}s)", flush=True)
        for r, sc in enumerate(scores):
            log.write(json.dumps(dict(round=rnd, rig=r, roll=rolls[r],
                                      jitter=list(jits[r]), tag=tag,
                                      theta=[round(float(v), 5) for v in ths[r]],
                                      **{k: (bool(v) if isinstance(v, (bool, np.bool_))
                                             else round(float(v), 4))
                                         for k, v in sc.items()})) + "\n")
        log.flush()
        return scores, rews

    if MODE == "eval":
        for rnd in range(ROUNDS):
            ths = [theta0] * RIGS
            jits = [_panel(rnd, r) for r in range(RIGS)]
            scores, _ = run_round(rnd, ths, jits, "eval")
            for r, sc in enumerate(scores):
                tally.append((r, rolls[r], sc))
        n = len(tally)
        succ = sum(1 for _, _, sc in tally if sc["success"])
        snaps = sum(1 for _, _, sc in tally if sc.get("snap"))
        trig = sum(1 for _, _, sc in tally if sc.get("triggered"))
        print(f"[ae] ===== EVAL {succ}/{n} success, snaps={snaps}, "
              f"triggered={trig}/{n} =====", flush=True)
        by_roll = {}
        for r, roll, sc in tally:
            by_roll.setdefault(roll, []).append(sc["success"])
        for roll in sorted(by_roll):
            ss = by_roll[roll]
            print(f"[ae]   roll {roll:+7.2f}deg: {sum(ss)}/{len(ss)}", flush=True)
        json.dump(dict(theta=[float(v) for v in theta0], success=succ, n=n,
                       snaps=snaps, by_roll={str(k): f"{sum(v)}/{len(v)}"
                                             for k, v in by_roll.items()}),
                  open(os.path.join(OUT, "eval_summary.json"), "w"), indent=1)
    else:
        for rnd in range(ROUNDS):
            ths = [(mu if (rnd == 0 and r == 0) else
                    np.clip(rng.normal(mu, sig), LO, HI)) for r in range(RIGS)]
            jits = [(rng.uniform(-JIT_XY, JIT_XY), rng.uniform(-JIT_XY, JIT_XY),
                     rng.uniform(-JIT_DZ, JIT_DZ)) for _ in range(RIGS)]
            scores, rews = run_round(rnd, ths, jits, "tune")
            if SNAP_PEN:                  # gentle: flips score below failure
                rews = [r_ - SNAP_PEN * bool(s_.get("snap"))
                        for r_, s_ in zip(rews, scores)]
            for th, sc in zip(ths, scores):
                if sc["reward"] > best["reward"]:
                    best = dict(theta=[float(v) for v in th], **{
                        k: (bool(v) if isinstance(v, (bool, np.bool_))
                            else float(v)) if not isinstance(v, str) else v
                        for k, v in sc.items()})
            mu, sig = PL.cem_update(mu, sig, ths, rews, LO, HI)
            sig = np.maximum(sig, 0.015 * (HI - LO))   # keep exploring a little
            print(f"[ae] rnd{rnd} MUθ=[{','.join(f'{v:.5f}' for v in mu)}]",
                  flush=True)
            print(f"[ae] rnd{rnd} BESTθ r={best['reward']:.2f} "
                  f"[{','.join(f'{v:.5f}' for v in best['theta'])}]", flush=True)
        # deterministic panel check of the final MU (2 rounds)
        fin = []
        for k in range(2):
            ths = [mu] * RIGS
            jits = [_panel(k, r) for r in range(RIGS)]
            scores, _ = run_round(100 + k, ths, jits, "mu-eval")
            fin += scores
        succ = sum(1 for sc in fin if sc["success"])
        print(f"[ae] ===== TUNE done: MU panel {succ}/{len(fin)} =====", flush=True)
        json.dump(dict(mu=[float(v) for v in mu], best=best,
                       mu_panel=f"{succ}/{len(fin)}"),
                  open(os.path.join(OUT, "tune_summary.json"), "w"), indent=1)

    print(f"[ae] DONE -> {OUT}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
