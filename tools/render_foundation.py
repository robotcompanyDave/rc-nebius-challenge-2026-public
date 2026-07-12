#!/usr/bin/env python3
"""
Render the compliant SpringFoundation washer-tilt INSIDE the full GraspSortEnv.

Why not the standalone probe? The minimal hand-built stage (tools/probe_soft_tilt.py)
renders STATIC prims but NOT physics-dynamic bodies under the headless RTX boot
(proven exhaustively: displayColor/PreviewSurface/OmniPBR, every camera angle, with
and without an authored PhysicsScene, and a canonical world.scene DynamicCuboid — all
invisible). The FULL env boot (arm USD load + kit render extensions) does NOT have
this limitation — render_review captures dynamic parts fine. So we reuse that boot:
GraspSortEnv(GS_SOFT=0) for a render-capable stage, then drop the SpringFoundation +
one flat washer + a kinematic pressing finger onto env.stage and capture the tilt.

    GS_SOFT=0 docker/run.sh tools/render_foundation.py
Env:
  GS_SF_STIFF (300) GS_SF_DAMP (6) GS_SF_CELL (0.004) GS_SF_FRIC (0.9)
  GS_RF_OFFSET (0.006)  press offset from centre m
  GS_RF_DEPTH  (0.0025) max press m (2.5mm ~ the 10-15 deg tilt band)
  GS_RF_RATE   (0.006)  press speed m/s
  GS_RF_OUT    (dated dir)
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
COUPLE = float(os.environ.get("GS_SF_COUPLE", "150"))       # shear coupling (0=Winkler)
COUPLE_DAMP = float(os.environ.get("GS_SF_COUPLE_DAMP", "0"))
FALLOFF = float(os.environ.get("GS_SF_FALLOFF", "0"))        # neighbor-of-neighbor decay
CUTOFF = float(os.environ.get("GS_SF_CUTOFF", "0.10"))       # drop links weaker than this
SKIN = os.environ.get("GS_RF_SKIN", "1") not in ("0", "", "false")  # smooth foam skin
OFFSET = float(os.environ.get("GS_RF_OFFSET", "0.006"))
MAXDEPTH = float(os.environ.get("GS_RF_DEPTH", "0.0025"))
RATE = float(os.environ.get("GS_RF_RATE", "0.006"))
FING_FRIC = float(os.environ.get("GS_RF_FING_FRIC", "0.1"))
_now = datetime.datetime.now()
OUT = os.environ.get("GS_RF_OUT", os.path.join(
    "data", _now.strftime("%Y-%m-%d"), _now.strftime("%H%M") + "-render_foundation"))


def main():
    os.environ.setdefault("GS_SOFT", "0")   # drop the old kinematic soft_rig
    os.makedirs(OUT, exist_ok=True)

    from graspsort.sim_env import GraspSortEnv
    env = GraspSortEnv(headless=True, width=960, height=540)
    # the stiff (k=300) spring tiles need a fine physics substep or they overshoot
    # and catapult the washer to vertical; the standalone probe used 1/240. The full
    # env's World() defaults to 1/60 — too coarse. Refine it (rendering_dt stays 1/60).
    try:
        pcx = env.world.get_physics_context()
        pcx.set_physics_dt(1.0 / 240.0)
    except Exception as e:
        print(f"[rf] could not set physics_dt: {e}", flush=True)
    env.reset_world()
    try:
        print(f"[rf] physics_dt now = {env.world.get_physics_context().get_physics_dt():.5f}",
              flush=True)
    except Exception:
        pass
    stage = env.stage

    from pxr import UsdGeom, UsdPhysics, UsdShade, UsdLux, Gf, Sdf, PhysxSchema

    # place the bed on the platform, at the arm's work centre, lifted clear of the
    # slab so the sprung tiles hang from their world joints without slab contact
    wc = env._work_centre_from_stage()
    # lift the bed clear of the platform slab so the sprung tiles get their full
    # downward travel. At only +12mm the tile bottoms hit the slab after ~2mm and
    # act as a hard fulcrum that catapults the washer vertical; +30mm gives room for
    # the smooth 7mm sink that produces the gentle 10-15 deg tilt (standalone regime).
    SURF_Z = env.table_top_z + 0.030
    CENTRE = (wc[0], wc[1])
    print(f"[rf] work centre={CENTRE} surf_z={SURF_Z:.3f} table={env.table_top_z:.3f}",
          flush=True)

    # ── visual materials (OmniPBR / MDL) — these DO render in the full env ────
    UsdGeom.Scope.Define(stage, "/World/RFLooks")

    def omnipbr(name, rgb, rough=0.6, metallic=0.0):
        mp = f"/World/RFLooks/{name}"
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

    from graspsort.soft_foundation import SpringFoundation, SurfaceSkin
    found = SpringFoundation(stage, CENTRE, SURF_Z, span=0.060, cell=CELL,
                             stiffness=STIFF, damping=DAMP, tile_mass=TMASS,
                             travel=TRAVEL, friction=FRIC,
                             couple=COUPLE, couple_damp=COUPLE_DAMP,
                             couple_falloff=FALLOFF, couple_cutoff=CUTOFF,
                             visible=not SKIN,        # hide tiles when the skin is on
                             tile_color=(0.82, 0.68, 0.46)).build()

    if not SKIN:
        tileA = omnipbr("TileA", (0.82, 0.68, 0.46))
        tileB = omnipbr("TileB", (0.52, 0.40, 0.24))
        n_tiles = max(2, int(round(0.060 / CELL)))
        for idx, tp in enumerate(found.tile_paths):
            i, j = divmod(idx, n_tiles)
            geo = stage.GetPrimAtPath(tp + "/geo")
            if geo.IsValid():
                UsdShade.MaterialBindingAPI.Apply(geo).Bind(
                    tileA if (i + j) % 2 == 0 else tileB)

    # smooth non-colliding foam skin draped over the tile tops (follows the dish)
    skin = None
    if SKIN:
        skin = SurfaceSkin(found, color=(0.86, 0.64, 0.46)).build()
        UsdShade.MaterialBindingAPI.Apply(
            stage.GetPrimAtPath(skin.path)).Bind(
                omnipbr("Skin", (0.86, 0.64, 0.46), rough=0.75))

    # flat m12 washer at centre
    from graspsort import parts
    spec = parts.PartSpec(kind="washer", size="m12", pose="flat",
                          xy=CENTRE, rotz_deg=0.0)
    wpath = parts.spawn_part(stage, "/World/RFWasher", spec, SURF_Z)
    redmat = omnipbr("WasherRed", (0.86, 0.14, 0.10), rough=0.5)
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(wpath)).Bind(
        redmat, UsdShade.Tokens.strongerThanDescendants)
    OD = parts.size_dims("m12")["w_od"]

    # low-friction KINEMATIC pressing finger over the offset point
    fx, fy = CENTRE[0] - OFFSET, CENTRE[1]
    fw, fh = 0.009, 0.030
    fmat_p = "/World/RFFingerMat"
    UsdShade.Material.Define(stage, fmat_p)
    fm = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(fmat_p))
    fm.CreateStaticFrictionAttr(FING_FRIC)
    fm.CreateDynamicFrictionAttr(FING_FRIC)
    fmat = UsdShade.Material(stage.GetPrimAtPath(fmat_p))
    finger = UsdGeom.Xform.Define(stage, "/World/RFFinger")
    f_start_bottom = SURF_Z + 0.020
    UsdGeom.XformCommonAPI(finger.GetPrim()).SetTranslate(
        Gf.Vec3d(fx, fy, f_start_bottom + fh / 2.0))
    rb = UsdPhysics.RigidBodyAPI.Apply(finger.GetPrim())
    rb.CreateKinematicEnabledAttr(True)
    fgeo = UsdGeom.Cube.Define(stage, "/World/RFFinger/geo")
    fgeo.GetSizeAttr().Set(1.0)
    UsdGeom.XformCommonAPI(fgeo.GetPrim()).SetScale(Gf.Vec3f(fw, fw, fh))
    UsdPhysics.CollisionAPI.Apply(fgeo.GetPrim())
    pc = PhysxSchema.PhysxCollisionAPI.Apply(fgeo.GetPrim())
    pc.CreateContactOffsetAttr().Set(0.0015)
    pc.CreateRestOffsetAttr().Set(0.0)
    UsdShade.MaterialBindingAPI.Apply(fgeo.GetPrim()).Bind(
        fmat, UsdShade.Tokens.weakerThanDescendants, "physics")
    UsdShade.MaterialBindingAPI.Apply(fgeo.GetPrim()).Bind(
        omnipbr("Finger", (0.16, 0.18, 0.22), rough=0.4))
    finger_op = UsdGeom.XformCommonAPI(finger.GetPrim())

    # the foundation + finger were added AFTER world.reset(); re-init physics views
    env.world.reset()

    # DEBUG: where did things actually land? (world translations)
    def _wpos(p):
        m = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(p))
        t = m.ExtractTranslation()
        return (round(t[0], 3), round(t[1], 3), round(t[2], 3))
    print(f"[dbg] washer @ {_wpos(wpath)}  tile0 @ {_wpos(found.tile_paths[0])}  "
          f"tileN @ {_wpos(found.tile_paths[len(found.tile_paths)//2])}  "
          f"finger @ {_wpos('/World/RFFinger')}", flush=True)

    # ── washer tilt measurement (rim heights along the press axis) ────────────
    def washer_pose():
        M = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(wpath))
        return M

    def measure():
        M = washer_pose()
        t = M.ExtractTranslation()
        r = OD / 2.0
        def rim(dx, dy):
            p = Gf.Vec3d(dx, dy, 0.0)
            wp = M.Transform(p)
            return wp
        near = rim(-r, 0.0)
        far = rim(r, 0.0)
        dz = far[2] - near[2]
        dx = far[0] - near[0]
        tilt = math.degrees(math.atan2(dz, dx)) if abs(dx) > 1e-6 else 0.0
        return {"tilt": tilt, "near_z": (near[2] - SURF_Z) * 1000.0,
                "far_z": (far[2] - SURF_Z) * 1000.0,
                "defl_mm": found.max_deflection_mm(),
                "cz": (t[2] - SURF_Z) * 1000.0}

    # ── camera (3/4 close on the bed) + capture, render_review recipe ─────────
    from render_review import look_at_quat
    from isaacsim.sensors.camera import Camera
    import cv2
    tgt = (CENTRE[0], CENTRE[1], SURF_Z + 0.004)
    # PROVEN far 3/4 view (frames the bed cleanly + balanced exposure). Moving closer
    # blows out the auto-exposure on the big white platform, so we ZOOM IN POST by
    # cropping the frame centre instead of moving the camera.
    eye = (CENTRE[0] + 0.62, CENTRE[1] - 0.72, SURF_Z + 0.58)
    cam = Camera(prim_path="/World/RFCam", position=np.array(eye), frequency=30,
                 resolution=(1280, 720), orientation=look_at_quat(eye, tgt))
    cam.initialize()
    for _ in range(20):
        env.step(render=True)

    # the bed sits around frame centre at this distance; this box crops a tight zoom
    def _crop(fr):
        h, w = fr.shape[:2]
        cx, cy = int(w * 0.50), int(h * 0.52)
        hw, hh = int(w * 0.22), int(h * 0.22)
        sub = fr[max(0, cy - hh):cy + hh, max(0, cx - hw):cx + hw]
        return sub

    def shot(label):
        rgba = None
        for _ in range(12):
            env.step(render=True)
            if skin is not None:
                skin.update()           # drape the foam skin over the current dish
            rgba = cam.get_rgba()
            if rgba is not None and getattr(rgba, "size", 0) > 0:
                break
        if rgba is None or getattr(rgba, "size", 0) == 0:
            return None
        fr = np.asarray(rgba)[:, :, :3].astype(np.uint8).copy()
        print(f"[cam] {label}: mean={fr.mean():.0f} std={fr.std():.1f}", flush=True)
        cv2.putText(fr, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (245, 245, 245), 2, cv2.LINE_AA)
        return fr

    frames = []

    def write(name, fr):
        cv2.imwrite(os.path.join(OUT, name), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        sub = _crop(fr)
        if sub.size:
            cv2.imwrite(os.path.join(OUT, name.replace(".png", "_zoom.png")),
                        cv2.cvtColor(sub, cv2.COLOR_RGB2BGR))

    def save(name, label):
        fr = shot(label)
        if fr is not None:
            write(name, fr)
        return fr

    # settle the washer, capture rest
    for _ in range(120):
        env.step(render=False)
    rest = measure()
    print(f"[rf] REST tilt={rest['tilt']:.1f} cz={rest['cz']:.2f}mm "
          f"defl={rest['defl_mm']:.2f}mm", flush=True)
    save("rest.png", "REST  flat washer on the sprung bed")

    rec = open(os.path.join(OUT, "descent.jsonl"), "w")
    rec.write(json.dumps({"phase": "rest", **rest}) + "\n")

    # descend the finger. The held pose is not always stable in this env (the washer
    # can roll past the tip-over cliff), so we grab the MONEY SHOT the moment the tilt
    # first lands in the target band (10-16 deg) during the descent — an honest image
    # of the compliant tilt regardless of what the washer does afterwards.
    dt = 1.0 / 60.0
    depth = 0.0
    curve = []
    step = 0
    reached = False
    held = 0
    money = False
    best = None            # (abs(tilt-13), frame, tilt) fallback if band never hit
    while True:
        if not reached:
            depth = min(MAXDEPTH, depth + RATE * dt)
            if depth >= MAXDEPTH - 1e-6:
                reached = True
        else:
            held += 1
        finger_op.SetTranslate(Gf.Vec3d(fx, fy, (SURF_Z - depth) + fh / 2.0))
        env.step(render=(step % 4 == 0))
        step += 1
        if step % 3 == 0:
            m = measure()
            m["press_mm"] = depth * 1000.0
            curve.append(m)
            rec.write(json.dumps({"phase": "press", **m}) + "\n")
            fr = shot(f"press {depth*1000:.1f}mm  tilt {m['tilt']:.0f}deg")
            if fr is not None:
                frames.append(fr)
                # only consider frames PAST the initial-contact transient (press>0.6mm)
                # so we don't grab a noisy near-flat spike; target a clean ~14 deg cock
                TARGET = float(os.environ.get("GS_RF_TILT", "22"))
                if m["press_mm"] > 0.3:
                    d = abs(m["tilt"] - TARGET)
                    if best is None or d < best[0]:
                        best = (d, fr, m["tilt"])
                    if (not money) and (TARGET - 6.0) <= m["tilt"] <= (TARGET + 6.0):
                        money = True
                        write("tilt_moneyshot.png", fr)
                        print(f"[rf] MONEY SHOT @ tilt={m['tilt']:.1f} "
                              f"press={m['press_mm']:.1f}mm", flush=True)
        if reached and held >= 60:
            break

    # if the band was never sampled, fall back to the closest-to-13deg frame we saw
    if (not money) and best is not None:
        write("tilt_moneyshot.png", best[1])
        print(f"[rf] money-shot fallback: closest tilt={best[2]:.1f}", flush=True)

    final = measure()
    save("tilted.png", f"HELD  tilt {final['tilt']:.0f}deg  @ press {MAXDEPTH*1000:.1f}mm")
    rec.write(json.dumps({"phase": "final", **final}) + "\n")
    rec.close()
    print(f"[rf] FINAL tilt={final['tilt']:.1f} near={final['near_z']:.1f} "
          f"far={final['far_z']:.1f} defl={final['defl_mm']:.1f}mm", flush=True)
    # surface-shape readout: how the dent spreads (foam = wide dish, Winkler = spike)
    prof = found.dish_profile_mm()
    print(f"[rf] couple={COUPLE}  spread_ratio={found.spread_ratio():.3f}  "
          f"dish_profile_mm=[{' '.join(f'{v:.1f}' for v in prof)}]", flush=True)

    if frames:
        h, w = frames[0].shape[:2]
        mp4 = os.path.join(OUT, "tilt.mp4")
        vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
        for fr in frames:
            vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        vw.release()
        print(f"[rf] wrote {mp4} ({len(frames)} frames)", flush=True)

    peak = max((c["tilt"] for c in curve), default=0.0)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump({"stiffness": STIFF, "cell": CELL, "friction": FRIC,
                   "offset_mm": OFFSET * 1000, "max_depth_mm": MAXDEPTH * 1000,
                   "rest": rest, "final": final, "peak_tilt": round(peak, 1)},
                  f, indent=2)
    print(f"[rf] DONE peak_tilt={peak:.1f} -> {OUT}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
