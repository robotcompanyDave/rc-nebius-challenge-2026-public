#!/usr/bin/env python3
"""
Soft-surface TILT probe (design frame 2) — can a compliant elastic surface tilt
a flat washer 10–20° when one side is pressed?

Standalone minimal scene (no arm): a SpringFoundation (Winkler bed of sprung
tiles) + one flat m12 washer + a rigid low-friction "finger" box that descends
at an OFFSET point and presses. We measure the washer's tilt (and lateral slip,
near/far edge height, surface deflection) as the press deepens — the goal is a
SMOOTH tilt ramp that dwells in 10–20°, not a snap.

    ~/TOOLS/isaac-sim/python.sh tools/probe_soft_tilt.py
Env:
  GS_SF_STIFF (300)  tile spring stiffness N/m       GS_SF_DAMP (6)
  GS_SF_CELL (0.004) tile size m                     GS_SF_FRIC (0.9) tile µ
  GS_SF_MASS (0.006) tile mass kg                    GS_SF_TRAVEL (0.014)
  GS_ST_OFFSET (0.006) press offset from centre m    GS_ST_DEPTH (0.008) max press m
  GS_ST_RATE (0.006) press speed m/s                 GS_ST_FING_FRIC (0.1)
  GS_ST_VIDEO (0)    1 → record an MP4               GS_ST_OUT (dated dir)
"""
import datetime
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

STIFF = float(os.environ.get("GS_SF_STIFF", "300"))
DAMP = float(os.environ.get("GS_SF_DAMP", "6"))
CELL = float(os.environ.get("GS_SF_CELL", "0.004"))
FRIC = float(os.environ.get("GS_SF_FRIC", "0.9"))
TMASS = float(os.environ.get("GS_SF_MASS", "0.006"))
TRAVEL = float(os.environ.get("GS_SF_TRAVEL", "0.014"))
OFFSET = float(os.environ.get("GS_ST_OFFSET", "0.006"))
MAXDEPTH = float(os.environ.get("GS_ST_DEPTH", "0.008"))
RATE = float(os.environ.get("GS_ST_RATE", "0.006"))
FING_FRIC = float(os.environ.get("GS_ST_FING_FRIC", "0.1"))
VIDEO = os.environ.get("GS_ST_VIDEO", "0") != "0"
_now = datetime.datetime.now()
OUT = os.environ.get("GS_ST_OUT", os.path.join(
    "data", _now.strftime("%Y-%m-%d"), _now.strftime("%H%M") + "-soft_tilt"))

SURF_Z = 0.50
CENTRE = (0.0, 0.0)


def main():
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "width": 960, "height": 540})

    import omni.usd
    import carb
    from isaacsim.core.api import World
    from pxr import UsdGeom, UsdPhysics, UsdShade, UsdLux, Gf, Sdf, PhysxSchema

    try:
        s = carb.settings.get_settings()
        s.set("/rtx/materialDb/syncLoads", True)
        s.set("/rtx/hydra/materialSyncLoads", True)
    except Exception:
        pass

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")

    # NO authored PhysicsScene — let isaacsim World() create its default scene.
    # sim_env.py (load-bearing boot note) shows the full env authors none and its
    # PHYSICS prims render fine; a hand-authored scene here is the leading suspect
    # for why dynamic prims stay invisible while static geometry renders.

    # safety ground well below the surface
    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.GetSizeAttr().Set(1.0)
    UsdGeom.XformCommonAPI(ground.GetPrim()).SetTranslate(Gf.Vec3d(0, 0, SURF_Z - 0.20))
    UsdGeom.XformCommonAPI(ground.GetPrim()).SetScale(Gf.Vec3f(2.0, 2.0, 0.02))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    # visible dark backdrop slab just under the tiles (reference + contrast)
    base = UsdGeom.Cube.Define(stage, "/World/Base")
    base.GetSizeAttr().Set(1.0)
    UsdGeom.XformCommonAPI(base.GetPrim()).SetTranslate(
        Gf.Vec3d(CENTRE[0], CENTRE[1], SURF_Z - 0.013))
    UsdGeom.XformCommonAPI(base.GetPrim()).SetScale(Gf.Vec3f(0.11, 0.11, 0.006))
    base.CreateDisplayColorAttr([Gf.Vec3f(0.10, 0.11, 0.13)])
    dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
    dome.CreateIntensityAttr(700.0)
    dist = UsdLux.DistantLight.Define(stage, "/World/Sun")
    dist.CreateIntensityAttr(2500.0)
    UsdGeom.XformCommonAPI(dist.GetPrim()).SetRotate(Gf.Vec3f(-42.0, 18.0, 0.0))

    # physics: small substep for stiff springs
    world = World(physics_dt=1.0 / 240.0, rendering_dt=1.0 / 60.0,
                  stage_units_in_meters=1.0)

    from graspsort.soft_foundation import SpringFoundation
    found = SpringFoundation(stage, CENTRE, SURF_Z, span=0.060, cell=CELL,
                             stiffness=STIFF, damping=DAMP, tile_mass=TMASS,
                             travel=TRAVEL, friction=FRIC, visible=True,
                             tile_color=(0.82, 0.68, 0.46)).build()

    # ── VISUAL materials (OmniPBR / MDL) ─────────────────────────────────────
    # displayColor alone does NOT render in this minimal stage once a PHYSICS
    # material is also bound (only material-free prims show displayColor — the
    # backdrop). Bind real OmniPBR visual materials (allPurpose) so RTX colours
    # the tiles + washer + finger; the physics-friction materials stay separate.
    UsdGeom.Scope.Define(stage, "/World/Looks")

    def omnipbr(name, rgb, rough=0.6, metallic=0.0):
        mp = f"/World/Looks/{name}"
        mat = UsdShade.Material.Define(stage, mp)
        sh = UsdShade.Shader.Define(stage, mp + "/Shader")
        sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
        sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(rough)
        sh.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(metallic)
        mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
        return mat

    tileA = omnipbr("TileA", (0.82, 0.68, 0.46))
    tileB = omnipbr("TileB", (0.52, 0.40, 0.24))
    n_tiles = max(2, int(round(0.060 / CELL)))
    for idx, tp in enumerate(found.tile_paths):
        i, j = divmod(idx, n_tiles)
        geo = stage.GetPrimAtPath(tp + "/geo")
        if geo.IsValid():
            UsdShade.MaterialBindingAPI.Apply(geo).Bind(
                tileA if (i + j) % 2 == 0 else tileB)

    # flat m12 washer at centre
    from graspsort import parts
    spec = parts.PartSpec(kind="washer", size="m12", pose="flat",
                          xy=CENTRE, rotz_deg=0.0)
    wpath = parts.spawn_part(stage, "/World/Washer", spec, SURF_Z)
    # bright red OmniPBR so the washer reads clearly (overrides the steel look)
    redmat = omnipbr("WasherRed", (0.86, 0.14, 0.10), rough=0.5)
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(wpath)).Bind(
        redmat, UsdShade.Tokens.strongerThanDescendants)

    # low-friction rigid KINEMATIC finger (a small box) over the press point
    fx, fy = CENTRE[0] - OFFSET, CENTRE[1]
    fw, fh = 0.009, 0.030
    fmat_p = "/World/FingerMat"
    UsdShade.Material.Define(stage, fmat_p)
    fm = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(fmat_p))
    fm.CreateStaticFrictionAttr(FING_FRIC)
    fm.CreateDynamicFrictionAttr(FING_FRIC)
    fmat = UsdShade.Material(stage.GetPrimAtPath(fmat_p))
    finger = UsdGeom.Xform.Define(stage, "/World/Finger")
    f_start_bottom = SURF_Z + 0.020
    f_cz = f_start_bottom + fh / 2.0
    UsdGeom.XformCommonAPI(finger.GetPrim()).SetTranslate(Gf.Vec3d(fx, fy, f_cz))
    rb = UsdPhysics.RigidBodyAPI.Apply(finger.GetPrim())
    rb.CreateKinematicEnabledAttr(True)
    fgeo = UsdGeom.Cube.Define(stage, "/World/Finger/geo")
    fgeo.GetSizeAttr().Set(1.0)
    UsdGeom.XformCommonAPI(fgeo.GetPrim()).SetScale(Gf.Vec3f(fw, fw, fh))
    fgeo.CreateDisplayColorAttr([Gf.Vec3f(0.20, 0.22, 0.26)])
    UsdPhysics.CollisionAPI.Apply(fgeo.GetPrim())
    pc = PhysxSchema.PhysxCollisionAPI.Apply(fgeo.GetPrim())
    pc.CreateContactOffsetAttr().Set(0.0015)
    pc.CreateRestOffsetAttr().Set(0.0)
    UsdShade.MaterialBindingAPI.Apply(fgeo.GetPrim()).Bind(
        fmat, UsdShade.Tokens.weakerThanDescendants, "physics")
    UsdShade.MaterialBindingAPI.Apply(fgeo.GetPrim()).Bind(
        omnipbr("Finger", (0.16, 0.18, 0.22), rough=0.4))   # visual
    finger_op = UsdGeom.XformCommonAPI(finger.GetPrim())

    # DIAGNOSTIC (GS_ST_TESTCUBE=1): a canonical isaacsim-registered dynamic body.
    # If THIS renders top-down but the hand-authored tiles/washer don't, the fix is
    # to create parts as world.scene-registered core objects, not raw USD prims.
    if os.environ.get("GS_ST_TESTCUBE", "0") != "0":
        from isaacsim.core.api.objects import DynamicCuboid
        world.scene.add(DynamicCuboid(
            prim_path="/World/TestCube", name="testcube",
            position=np.array([CENTRE[0] + 0.020, CENTRE[1] + 0.020, SURF_Z + 0.010]),
            size=0.012, color=np.array([0.05, 0.9, 0.15])))
        print("[diag] added registered DynamicCuboid /World/TestCube", flush=True)

    world.reset()

    # washer geometry (for rim-height tilt)
    OD = parts.size_dims("m12")["w_od"]
    R = OD / 2.0
    xc_cls = UsdGeom.XformCache

    def washer_pose():
        M = UsdGeom.XformCache().GetLocalToWorldTransform(
            stage.GetPrimAtPath(wpath))
        return M

    def rim_world(M, sx):
        # local rim point (sx*R, 0, 0) → world
        p = M.Transform(Gf.Vec3d(sx * R, 0.0, 0.0))
        return np.array([p[0], p[1], p[2]])

    def measure():
        M = washer_pose()
        t = M.ExtractTranslation()
        cxy = np.array([t[0], t[1]])
        near = rim_world(M, -1.0)   # press side (−x)
        far = rim_world(M, +1.0)    # far side (+x)
        # tilt along press axis from rim heights
        dz = far[2] - near[2]
        dx = math.hypot(far[0] - near[0], far[1] - near[1])
        tilt = math.degrees(math.atan2(dz, dx)) if dx > 1e-6 else 0.0
        lateral = float(np.hypot(cxy[0] - CENTRE[0], cxy[1] - CENTRE[1]))
        return {"tilt": tilt, "near_z": (near[2] - SURF_Z) * 1000.0,
                "far_z": (far[2] - SURF_Z) * 1000.0,
                "lateral_mm": lateral * 1000.0,
                "defl_mm": found.max_deflection_mm(),
                "cz": (t[2] - SURF_Z) * 1000.0}

    # optional camera (side-on 3/4)
    cam = cv2 = None
    frames = []
    if VIDEO:
        from render_review import look_at_quat
        from isaacsim.sensors.camera import Camera
        import cv2 as _cv2
        cv2 = _cv2
        # far 3/4 view (the one that rendered the backdrop band) — discriminating
        eye = (CENTRE[0] + 0.30, CENTRE[1] - 0.35, SURF_Z + 0.26)
        cam = Camera(prim_path="/World/Cam", position=np.array(eye), frequency=30,
                     resolution=(960, 540),
                     orientation=look_at_quat(eye, (CENTRE[0], CENTRE[1], SURF_Z + 0.004)))
        cam.initialize()
        # warm up the render product. Docker RTX needs several REAL render ticks
        # (app.update) before the rgb annotator returns data — world.step at
        # physics_dt 1/240 renders too rarely, so the annotator stays None.
        for _ in range(8):
            world.step(render=False)
        for _ in range(80):
            app.update()
            r = cam.get_rgba()
            if r is not None and getattr(r, "size", 0) > 0:
                break

    def _shot(label):
        rgba = None
        for _ in range(12):
            app.update()
            rgba = cam.get_rgba()
            if rgba is not None and getattr(rgba, "size", 0) > 0:
                break
        if rgba is None or getattr(rgba, "size", 0) == 0:
            return None
        fr = np.asarray(rgba)[:, :, :3].astype(np.uint8).copy()
        print(f"[cam] frame mean={fr.mean():.0f} std={fr.std():.1f}", flush=True)
        cv2.putText(fr, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (25, 25, 25), 2, cv2.LINE_AA)
        return fr

    def grab(label):
        if cam is None:
            return
        fr = _shot(label)
        if fr is not None:
            frames.append(fr)

    def still(name, label):
        if cam is None:
            return
        fr = _shot(label)
        if fr is not None:
            cv2.imwrite(os.path.join(OUT, name), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))

    # settle the washer on the bed
    for _ in range(120):
        world.step(render=False)
    rest = measure()
    print(f"[tilt] REST tilt={rest['tilt']:.1f}° sink(cz)={rest['cz']:.2f}mm "
          f"defl={rest['defl_mm']:.2f}mm lateral={rest['lateral_mm']:.2f}mm",
          flush=True)
    still("rest.png", "REST  flat washer on the sprung bed")

    os.makedirs(OUT, exist_ok=True)
    rec = open(os.path.join(OUT, "descent.jsonl"), "w")
    rec.write(json.dumps({"phase": "rest", **rest}) + "\n")

    # descend the finger to MAXDEPTH below the surface at RATE, holding steady
    dt = 1.0 / 60.0
    depth = 0.0
    curve = []
    step = 0
    hold_after = 90
    reached = False
    held = 0
    while True:
        if not reached:
            depth = min(MAXDEPTH, depth + RATE * dt)
            if depth >= MAXDEPTH - 1e-6:
                reached = True
        else:
            held += 1
        f_bottom = SURF_Z - depth
        finger_op.SetTranslate(Gf.Vec3d(fx, fy, f_bottom + fh / 2.0))
        world.step(render=False)
        step += 1
        if step % 6 == 0:
            m = measure()
            m["press_mm"] = depth * 1000.0
            curve.append(m)
            rec.write(json.dumps({"phase": "press", **m}) + "\n")
            if step % 24 == 0:
                print(f"[tilt] press={depth*1000:5.1f}mm tilt={m['tilt']:5.1f}° "
                      f"far={m['far_z']:5.1f} near={m['near_z']:5.1f} "
                      f"defl={m['defl_mm']:4.1f} lat={m['lateral_mm']:4.1f}mm",
                      flush=True)
            if VIDEO and step % 4 == 0:
                grab(f"press {depth*1000:.1f}mm  tilt {m['tilt']:.0f}deg")
        if reached and held >= hold_after:
            break

    final = measure()
    still("tilted.png", f"HELD  tilt {final['tilt']:.0f}deg  @ press {MAXDEPTH*1000:.1f}mm")
    rec.write(json.dumps({"phase": "final", **final}) + "\n")
    rec.close()

    # camera framing sweep: render the held pose from several viewpoints, save
    # each still + its frame std, so we can pick a viewpoint that actually
    # frames the (tiny) parts. GS_ST_CAMSWEEP=1.
    if os.environ.get("GS_ST_CAMSWEEP", "0") != "0":
        from render_review import look_at_quat
        from isaacsim.sensors.camera import Camera
        import cv2 as _cv2
        finger_op.SetTranslate(Gf.Vec3d(fx, fy, (SURF_Z - MAXDEPTH) + fh / 2.0))
        tgt = (CENTRE[0], CENTRE[1], SURF_Z + 0.003)
        cands = {
            "top":  (0.002, 0.002, 0.16),
            "e70":  (0.04, -0.02, 0.13),
            "e50":  (0.07, -0.05, 0.10),
            "e35":  (0.09, -0.08, 0.07),
            "e20":  (0.12, -0.11, 0.045),
            "e10":  (0.14, -0.13, 0.028),
        }
        for i, (nm, off) in enumerate(cands.items()):
            eye = (CENTRE[0] + off[0], CENTRE[1] + off[1], SURF_Z + off[2])
            c = Camera(prim_path=f"/World/Csweep{i}", position=np.array(eye),
                       frequency=30, resolution=(960, 540),
                       orientation=look_at_quat(eye, tgt))
            c.initialize()
            rgba = None
            for _ in range(80):
                # step WITH rendering on (render_review's proven approach) so the
                # physics→Hydra transform sync fires for DYNAMIC prims — app.update()
                # alone renders the scene graph but may not pull fresh physics poses.
                world.step(render=True)
                rgba = c.get_rgba()
                if rgba is not None and getattr(rgba, "size", 0) > 0:
                    break
            if rgba is None or getattr(rgba, "size", 0) == 0:
                print(f"[camsweep] {nm} EMPTY", flush=True)
                continue
            fr = np.asarray(rgba)[:, :, :3].astype(np.uint8).copy()
            print(f"[camsweep] {nm} mean={fr.mean():.0f} std={fr.std():.1f} "
                  f"eye={tuple(round(e,3) for e in eye)}", flush=True)
            _cv2.putText(fr, nm, (12, 30), _cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                         (20, 20, 20), 2, _cv2.LINE_AA)
            _cv2.imwrite(os.path.join(OUT, f"cam_{nm}.png"),
                         _cv2.cvtColor(fr, _cv2.COLOR_RGB2BGR))
    print(f"[tilt] FINAL press={MAXDEPTH*1000:.1f}mm tilt={final['tilt']:.1f}° "
          f"far={final['far_z']:.1f}mm lateral={final['lateral_mm']:.1f}mm", flush=True)

    peak = max((c["tilt"] for c in curve), default=0.0)
    in_band = [c["press_mm"] for c in curve if 10.0 <= c["tilt"] <= 20.0]
    summary = {"stiffness": STIFF, "damping": DAMP, "cell": CELL, "friction": FRIC,
               "offset_mm": OFFSET * 1000, "max_depth_mm": MAXDEPTH * 1000,
               "rest": rest, "final": final, "peak_tilt": round(peak, 1),
               "press_mm_for_10_20deg": [round(x, 1) for x in in_band],
               "washer_od_mm": OD * 1000}
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if curve:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            xs = [c["press_mm"] for c in curve]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.axhspan(10, 20, color="#ffd23f", alpha=0.35, label="target 10–20°")
            ax.plot(xs, [c["tilt"] for c in curve], "-o", color="#157a4f",
                    ms=3, label="washer tilt")
            ax.plot(xs, [c["far_z"] for c in curve], color="#2f6fe0",
                    label="far edge z (mm)")
            ax.plot(xs, [c["near_z"] for c in curve], color="#e0483c",
                    label="near edge z (mm)")
            ax.plot(xs, [c["lateral_mm"] for c in curve], "--", color="#8a6d1f",
                    label="lateral slip (mm)")
            ax.set_xlabel("finger press depth below surface (mm)")
            ax.set_ylabel("tilt (°)  /  height, slip (mm)")
            ax.set_title(f"Soft-surface tilt — k={STIFF} µ={FRIC} offset={OFFSET*1000:.0f}mm")
            ax.legend(fontsize=8); ax.grid(alpha=.3)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, "tilt_curve.png"), dpi=130)
            print(f"[tilt] wrote {OUT}/tilt_curve.png", flush=True)
        except Exception as e:
            print(f"[tilt] plot skipped: {e}", flush=True)

    if VIDEO and frames:
        vp = os.path.join(OUT, "press.mp4")
        vw = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"mp4v"), 15,
                             (frames[0].shape[1], frames[0].shape[0]))
        for fr in frames:
            vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        vw.release()
        print(f"[tilt] wrote {vp} ({len(frames)} frames)", flush=True)

    print(f"[tilt] DONE peak_tilt={peak:.1f}° -> {OUT}", flush=True)
    app.close()


if __name__ == "__main__":
    main()
