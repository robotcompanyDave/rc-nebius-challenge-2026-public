#!/usr/bin/env python3
"""
M3d step 2 — validate the neoprene dent PROFILE against theory.

For each preset: build a foundation, press a kinematic BAR (line load, full
bed width) 3 mm at the centre, settle 3 s, subtract the gravity-sag baseline,
and fit the decay tail to w(x) = w0·exp(−x/ℓ). A line load makes the 1-D
Pasternak law ℓ = cell·√(k_link/k_cell) exact (a compact pad on a 2-D bed
decays faster — Bessel, not exp; first probe iteration chased that artifact).

  PASS  ℓ_fit within ±30% of analytic ℓ
        AND first-neighbour step ≤ 55% of (baseline-subtracted) peak

Physics-only standalone boot (numbers, no render). One boot, all presets.

    docker/run.sh tools/probe_profile.py
Env: GS_PP_MATS (json path; default = built-in N1/N2/N3 + M3 gel reference)
     GS_PP_OUT  (dated dir)
"""
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

_now = datetime.datetime.now()
OUT = os.environ.get("GS_PP_OUT", os.path.join(
    "data", _now.strftime("%Y-%m-%d"), _now.strftime("%H%M") + "-profile"))
SURF_Z = 0.50
CELL = 0.005
PRESS = 0.003          # 3 mm press
PAD_HW = 0.004         # bar half-width in x (8 mm wide); full width in y


def default_mats():
    # ratio sweep at fixed k_cell=250 (validated softness band). ratio 0.2 is
    # the gel reference; 2-D solve says a proper ramp needs ratio ~3-8.
    mats = []
    for name, ratio in (("R0.2_gel", 0.2), ("R3", 3.0), ("R6", 6.0), ("R10", 10.0)):
        k = 250.0
        mats.append(dict(name=name, stiffness=k, couple=k * ratio,
                         damping=8.0, couple_damp=8.0, falloff=0.0, cutoff=0.10,
                         ell_mm=CELL * 1000 * ratio ** 0.5))
    return mats


def fit_ell(prof_mm, cell_mm):
    """Baseline-subtract (gravity sag = median of the 2 outermost tiles per
    side), then log-linear fit of the decay tail outside the bar footprint."""
    prof = np.asarray(prof_mm, dtype=float)
    base = float(np.median(np.concatenate([prof[:2], prof[-2:]])))
    prof = prof - base
    pk = float(np.max(prof))
    n = len(prof)
    ci = int(np.argmax(prof))
    xs, ys = [], []
    for j in range(ci + 1, n):
        v = prof[j]
        if v < max(0.06 * pk, 0.03):
            break
        x = (j - ci) * cell_mm
        if x <= PAD_HW * 1000:
            continue
        xs.append(x)
        ys.append(np.log(v))
    if len(xs) < 3:
        return None, base
    A = np.vstack([xs, np.ones(len(xs))]).T
    slope, _ = np.linalg.lstsq(A, np.array(ys), rcond=None)[0]
    return (None if slope >= 0 else float(-1.0 / slope)), base


def main():
    os.makedirs(OUT, exist_ok=True)
    mats = default_mats()
    mp = os.environ.get("GS_PP_MATS", "")
    if mp and os.path.isfile(mp):
        mats = json.load(open(mp))

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    import omni.usd
    from isaacsim.core.api import World
    from pxr import UsdGeom, UsdPhysics, Gf, PhysxSchema

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    world = World(physics_dt=1.0 / 240.0, rendering_dt=1.0 / 60.0,
                  stage_units_in_meters=1.0)

    from graspsort.soft_foundation import SpringFoundation
    results = []
    for mat in mats:
        if stage.GetPrimAtPath("/World/F").IsValid():
            stage.RemovePrim("/World/F")
        if stage.GetPrimAtPath("/World/Pad").IsValid():
            stage.RemovePrim("/World/Pad")
        found = SpringFoundation(
            stage, (0.0, 0.0), SURF_Z, span=0.080, cell=CELL,
            stiffness=mat["stiffness"], damping=mat.get("damping", 8.0),
            couple=mat["couple"], couple_damp=mat.get("couple_damp", 8.0),
            couple_falloff=mat.get("falloff", 0.0),
            couple_cutoff=mat.get("cutoff", 0.10),
            visible=False, parent="/World/F").build()
        # kinematic press pad
        pad = UsdGeom.Xform.Define(stage, "/World/Pad")
        UsdGeom.XformCommonAPI(pad.GetPrim()).SetTranslate(
            Gf.Vec3d(0, 0, SURF_Z + 0.05))
        rb = UsdPhysics.RigidBodyAPI.Apply(pad.GetPrim())
        rb.CreateKinematicEnabledAttr(True)
        geo = UsdGeom.Cube.Define(stage, "/World/Pad/geo")
        geo.GetSizeAttr().Set(1.0)
        # BAR: narrow in x, spans the full bed in y -> clean 1-D line load
        UsdGeom.XformCommonAPI(geo.GetPrim()).SetScale(
            Gf.Vec3f(PAD_HW * 2, 0.10, 0.02))
        UsdPhysics.CollisionAPI.Apply(geo.GetPrim())
        pc = PhysxSchema.PhysxCollisionAPI.Apply(geo.GetPrim())
        pc.CreateContactOffsetAttr().Set(0.0015)
        pc.CreateRestOffsetAttr().Set(0.0)
        world.reset()
        op = UsdGeom.XformCommonAPI(pad.GetPrim())
        # descend over 0.8 s, hold 3.0 s (soft high-coupling beds settle slowly)
        for s in range(int(240 * 3.8)):
            t = s / 240.0
            z = SURF_Z + 0.05 - min(t / 0.8, 1.0) * (0.05 + PRESS)
            op.SetTranslate(Gf.Vec3d(0, 0, z + 0.01))
            world.step(render=False)
        prof = found.dish_profile_mm()
        ell_fit, base = fit_ell(prof, CELL * 1000)
        profb = np.asarray(prof) - base
        pk = float(np.max(profb))
        ci = int(np.argmax(profb))
        # first tile outside the bar footprint
        out1 = ci + 1 + int(round(PAD_HW / CELL))
        step1 = float(profb[out1 - 1] - profb[out1]) if out1 < len(profb) else pk
        max_step = step1
        ell_target = mat.get("ell_mm")
        ok_ell = (ell_fit is not None and ell_target
                  and abs(ell_fit - ell_target) / ell_target <= 0.30)
        ok_step = step1 <= 0.55 * pk
        r = dict(name=mat["name"], stiffness=mat["stiffness"],
                 couple=mat["couple"], ell_target_mm=ell_target,
                 ell_fit_mm=None if ell_fit is None else round(ell_fit, 2),
                 peak_mm=round(pk, 2), max_step_mm=round(max_step, 2),
                 max_step_frac=round(max_step / pk, 2) if pk else None,
                 profile_mm=[round(float(v), 2) for v in prof],
                 pass_ell=bool(ok_ell), pass_step=bool(ok_step),
                 PASS=bool(ok_ell and ok_step))
        results.append(r)
        print(f"[pp] {r['name']:12} k={r['stiffness']:.0f} link={r['couple']:.0f} "
              f"ell target={ell_target and round(ell_target,1)} fit={r['ell_fit_mm']} "
              f"peak={r['peak_mm']} step={r['max_step_mm']}mm "
              f"({r['max_step_frac']}) ell:{'OK' if ok_ell else 'FAIL'} "
              f"step:{'OK' if ok_step else 'FAIL'}", flush=True)
        print(f"[pp]   profile: {r['profile_mm']}", flush=True)

    with open(os.path.join(OUT, "profiles.json"), "w") as f:
        json.dump(results, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        for r in results:
            prof = r["profile_mm"]
            n = len(prof)
            xs = (np.arange(n) - n // 2) * CELL * 1000
            ax.plot(xs, prof, "-o", ms=3,
                    label=f"{r['name']} ℓfit={r['ell_fit_mm']}mm "
                          f"{'PASS' if r['PASS'] else 'FAIL'}")
        ax.set_xlabel("x from press centre (mm)")
        ax.set_ylabel("deflection (mm)")
        ax.set_title(f"Dent profiles, {PRESS*1000:.0f} mm press — gel vs neoprene presets")
        ax.grid(alpha=0.3)
        ax.invert_yaxis()
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "profiles.png"), dpi=130)
        print(f"[pp] wrote {OUT}/profiles.png", flush=True)
    except Exception as e:
        print(f"[pp] plot skipped: {e}", flush=True)

    print(f"[pp] DONE -> {OUT}", flush=True)
    app.close()


if __name__ == "__main__":
    main()
