#!/usr/bin/env python3
"""
The training arena, rendered: a grid of PickRigs running the SAME maneuver
in parallel inside the full env boot (the lab itself is a standalone scene,
which does not render dynamic prims). Each rig gets the grid skin and a
per-rig jitter from the gauntlet PANEL — this is what a pick_lab / gauntlet
round actually looks like.

    docker/run.sh tools/render_arena.py
Env:
  GS_RA_RIGS (16)  GS_RA_COLS (4)  GS_RA_SPACING (0.24)
  GS_RA_THETA / GS_RA_MAT  (default: Stage T2 cand2 on PARKER_d24)
  GS_RA_OUT (dated dir)
"""
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

RIGS = int(os.environ.get("GS_RA_RIGS", "16"))
COLS = int(os.environ.get("GS_RA_COLS", "4"))
SPACING = float(os.environ.get("GS_RA_SPACING", "0.15"))
ROLL = float(os.environ.get("GS_RA_ROLL", "0"))   # per-rig lattice-rotation spread (deg)
_now = datetime.datetime.now()
OUT = os.environ.get("GS_RA_OUT", os.path.join(
    "data", _now.strftime("%Y-%m-%d"), _now.strftime("%H%M") + "-arena"))

DEF_THETA = "0.5424,0.0286,0.0040,0.9616,0.0018,0.0083,0.0014,0.0025,17.59"
DEF_MAT = ('{"name":"PARKER_d24","stiffness":300,"damping":24,'
           '"couple":3000,"couple_damp":8,"falloff":0,"cutoff":0.1,'
           '"leveling":true,"btip":"square","mode":"parallel"}')


def main():
    os.environ.setdefault("GS_SOFT", "0")
    os.environ.setdefault("GS_PL_SPAN", "0.05")
    os.makedirs(OUT, exist_ok=True)

    from graspsort.sim_env import GraspSortEnv
    env = GraspSortEnv(headless=True, width=960, height=540)
    env.world.get_physics_context().set_physics_dt(1.0 / 240.0)
    env.reset_world()
    stage = env.stage

    from pxr import UsdGeom, UsdShade, Gf, Sdf

    import tools.pick_lab as PL
    from graspsort.soft_foundation import SurfaceSkin
    wc = env._work_centre_from_stage()
    SURF = env.table_top_z + 0.030
    # the arm's USD load activates dynamic rendering in this boot, but it
    # has no business in the arena shot — hide it (do NOT delete it)
    _arm = stage.GetPrimAtPath("/World/big_table/ur10e")
    if _arm.IsValid():
        UsdGeom.Imageable(_arm).MakeInvisible()
        print("[ra] hid arm", flush=True)
    PL.SURF_Z = SURF
    theta = np.array([float(v) for v in os.environ.get(
        "GS_RA_THETA", DEF_THETA).split(",")])
    mat = json.loads(os.environ.get("GS_RA_MAT", DEF_MAT))
    rows = (RIGS + COLS - 1) // COLS
    print(f"[ra] {RIGS} rigs ({rows}x{COLS}), theta={list(theta)}",
          flush=True)

    # grid centred on the work centre
    def rig_c(r):
        i, j = divmod(r, COLS)
        return (wc[0] + (j - (COLS - 1) / 2) * SPACING,
                wc[1] + (i - (rows - 1) / 2) * SPACING)

    rigs = []
    from pxr import PhysxSchema
    for r in range(RIGS):
        _roll = (0.0 if ROLL == 0 or RIGS == 1
                 else float(np.linspace(-ROLL, ROLL, RIGS)[r]))
        rig = PL.PickRig(stage, r, rig_c(r), mat, roll_deg=_roll)
        rig.base = f"/World/ArenaRig{r}"
        rig.build()
        # 16 rigs x ~280 joints strain the env scene's solver budget — the
        # B spring holds on descent but yields on rise. Give the sprung
        # finger (and washer) a bigger per-body iteration budget.
        for pth in (rig.base + "/FingerB", rig.wpath):
            prb = PhysxSchema.PhysxRigidBodyAPI.Apply(
                stage.GetPrimAtPath(pth))
            prb.CreateSolverPositionIterationCountAttr().Set(32)
            prb.CreateSolverVelocityIterationCountAttr().Set(8)
        rigs.append(rig)

    # looks: shared grid-texture skin material + finger/washer colors
    UsdGeom.Scope.Define(stage, "/World/RALooks")
    mp = "/World/RALooks/SkinTex"
    m = UsdShade.Material.Define(stage, mp)
    st = UsdShade.Shader.Define(stage, mp + "/stReader")
    st.CreateIdAttr("UsdPrimvarReader_float2")
    st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tx = UsdShade.Shader.Define(stage, mp + "/tex")
    tx.CreateIdAttr("UsdUVTexture")
    tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(os.path.abspath(
        os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "assets", "grid_10mm.png")))
    tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        st.ConnectableAPI(), "result")
    sh = UsdShade.Shader.Define(stage, mp + "/pbr")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f) \
        .ConnectToSource(tx.ConnectableAPI(), "rgb")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")

    def solid(name, rgb, rough=0.5):
        p = f"/World/RALooks/{name}"
        mm = UsdShade.Material.Define(stage, p)
        ss = UsdShade.Shader.Define(stage, p + "/Shader")
        ss.CreateIdAttr("UsdPreviewSurface")
        ss.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*rgb))
        ss.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
        mm.CreateSurfaceOutput().ConnectToSource(
            ss.ConnectableAPI(), "surface")
        return mm

    red = solid("WasherRed", (0.95, 0.10, 0.06))
    _rs = UsdShade.Shader(stage.GetPrimAtPath(
        "/World/RALooks/WasherRed/Shader"))
    _rs.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.35, 0.02, 0.01))
    fa_look = solid("FingA", (0.16, 0.18, 0.22), 0.4)
    fb_look = solid("FingB", (0.15, 0.35, 0.90), 0.4)

    skins = []
    for r, rig in enumerate(rigs):
        sk = SurfaceSkin(rig.found, path=f"/World/ArenaSkin{r}").build()
        UsdShade.MaterialBindingAPI.Apply(
            stage.GetPrimAtPath(sk.path)).Bind(m)
        skins.append(sk)
        UsdShade.MaterialBindingAPI.Apply(
            stage.GetPrimAtPath(rig.wpath)).Bind(
                red, UsdShade.Tokens.strongerThanDescendants)
        UsdShade.MaterialBindingAPI.Apply(
            stage.GetPrimAtPath(rig.base + "/FingerA/geo")).Bind(fa_look)
        UsdShade.MaterialBindingAPI.Apply(
            stage.GetPrimAtPath(rig.base + "/FingerB/geo")).Bind(fb_look)

    env.world.reset()
    for r, rig in enumerate(rigs):
        rig.reset(theta, jitter=PL.PANEL[r % len(PL.PANEL)])

    # wide 3/4 camera framing the whole grid
    from render_review import look_at_quat
    from isaacsim.sensors.camera import Camera
    import cv2
    ext = max(rows, COLS) * SPACING
    # low corner shot down the grid diagonal: the nearest rig reads in
    # detail (washer visible ~35deg off the grip axis, clear of the finger
    # pillars), the other 15 recede behind it
    # GS_RA_CAM=all (default): frame EVERY rig from an elevated 3/4 view
    # computed from the grid bounds; =corner keeps the old low diagonal shot
    # (nearest rig in detail, the rest receding behind it).
    import math
    _cam = os.environ.get("GS_RA_CAM", "all")
    if _cam == "closeup":
        # tight 3/4 hero shot of rig0: ~35deg off the grip axis, low, close
        c0 = rig_c(0)
        tgt = (c0[0], c0[1], SURF + 0.012)
        eye = (c0[0] - 0.20, c0[1] - 0.26, SURF + 0.17)
    elif _cam == "corner":
        near = rig_c(0)
        tgt = (wc[0] + 0.08, wc[1] + 0.08, SURF + 0.01)
        eye = (near[0] - 0.34, near[1] - 0.50, SURF + 0.34)
    else:
        # target the grid centre; back the eye off along a ~35deg-off-grip-axis
        # azimuth (washers stay readable past the finger pillars) far enough
        # that the whole rows x COLS grid fits with margin
        gw = (COLS - 1) * SPACING
        gh = (rows - 1) * SPACING
        diag = math.hypot(gw + 0.14, gh + 0.14)
        az = math.radians(float(os.environ.get("GS_RA_CAM_AZ", "215")))
        dist = float(os.environ.get("GS_RA_CAM_DIST",
                                     max(0.55, 1.35 * diag + 0.30)))
        _elev = float(os.environ.get("GS_RA_CAM_ELEV", "0.58"))
        tgt = (wc[0], wc[1], SURF + 0.02)
        eye = (wc[0] + dist * math.cos(az), wc[1] + dist * math.sin(az),
               SURF + _elev * dist)
    cam = Camera(prim_path="/World/RACam", position=np.array(eye),
                 frequency=30, resolution=(1920, 1080),
                 orientation=look_at_quat(eye, tgt))
    cam.initialize()
    try:      # default near-clip (~1 m) erases close-up shots
        cam.set_clipping_range(0.03, 40.0)
    except Exception as e:
        print(f"[ra] clip range: {e}", flush=True)
    for _ in range(20):
        env.step(render=True)

    frames = []
    stills = {0.45: "arena_rest.png", 2.2: "arena_press.png",
              3.2: "arena_rollup.png", 3.6: "arena_t36.png",
              4.0: "arena_t40.png", 4.4: "arena_t44.png",
              4.8: "arena_t48.png", 5.2: "arena_t52.png",
              6.3: "arena_carry.png"}
    done = set()
    t_total = PL.T_TOTAL_P + PL.T_CARRY + 0.2
    steps = int(t_total / PL.PHYS_DT)
    for s in range(steps):
        t = s * PL.PHYS_DT
        for rig in rigs:
            rig.drive(t)          # every step — matches the validated env
                                  # replays (dec-4 loses the catch in env)
        env.step(render=(s % 8 == 0))
        if s % 8 == 0:
            for sk in skins:
                sk.update()
            rgba = cam.get_rgba()
            if rgba is not None and getattr(rgba, "size", 0) > 0:
                fr = np.asarray(rgba)[:, :, :3].astype(np.uint8).copy()
                _lbl = os.environ.get(
                    "GS_RA_LABEL", f"pick_lab arena  {RIGS} rigs")
                cv2.putText(fr, f"{_lbl}  t={t:4.1f}s",
                            (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (245, 245, 245), 2, cv2.LINE_AA)
                frames.append(fr)
                for st_t, name in stills.items():
                    if name not in done and t >= st_t:
                        done.add(name)
                        cv2.imwrite(os.path.join(OUT, name),
                                    cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        if s % 480 == 0:
            print(f"[ra] t={t:4.1f}s", flush=True)
            from pxr import UsdGeom as _UG
            xc = _UG.XformCache()
            for r in [i for i in (0, 5) if i < RIGS]:
                cz = xc.GetLocalToWorldTransform(
                    rigs[r].primC).ExtractTranslation()[2]
                bz = xc.GetLocalToWorldTransform(
                    stage.GetPrimAtPath(rigs[r].base + "/FingerB"))                     .ExtractTranslation()[2]
                print(f"[ra]   rig{r} carrierC_z={1000*(cz-SURF):+7.1f} "
                      f"fingerB_z={1000*(bz-SURF):+7.1f} ph={rigs[r]._ph} "
                      f"trig={rigs[r]._triggered}", flush=True)
    scores = [rig.final_score() for rig in rigs]
    for r, sc in enumerate(scores):
        print(f"[ra]   rig{r:02d} trig={int(sc['triggered'])} "
              f"cap={int(sc['captured'])} flew={int(sc['flew'])} "
              f"tilt={sc['tilt']:5.1f} lift={sc['lifted_mm']:+6.1f} "
              f"carry={sc.get('carry_mm', -99):+6.1f}", flush=True)
    print(f"[ra] succ={sum(1 for x in scores if x['success'])}/{RIGS} "
          f"carry_ok={sum(1 for x in scores if x.get('carry_ok'))}/{RIGS}",
          flush=True)
    if frames:
        h, w = frames[0].shape[:2]
        vw = cv2.VideoWriter(os.path.join(OUT, "arena.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
        for fr in frames:
            vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        vw.release()
        try:
            from graspsort.videoio import to_h264
            to_h264(os.path.join(OUT, "arena.mp4"))
        except Exception as e:
            print(f"[ra] h264: {e}", flush=True)
        print(f"[ra] arena.mp4 ({len(frames)} frames)", flush=True)
    print(f"[ra] DONE -> {OUT}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
