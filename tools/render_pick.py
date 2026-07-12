#!/usr/bin/env python3
"""
Replay a pick_lab maneuver θ in the FULL GraspSortEnv (the render-capable boot)
and capture phase stills + a small MP4 of the two-finger press→drag→roll-up.

Uses pick_lab's PickRig/drive verbatim (imported, surface height rebound), so
what renders is exactly what trained. Tiles are shown (no skin) by default —
set GS_RP_SKIN=1 for the smooth foam-skin look on a highlight run.

    GS_SOFT=0 docker/run.sh tools/render_pick.py
Env:
  GS_RP_RESULTS  path to a pick_lab results.json  (+ GS_RP_MATNAME to pick one)
  GS_RP_THETA    "a_off,press,b_out,b_tip,b_drag,gap,lift" — overrides RESULTS
  GS_RP_MAT      inline JSON material dict — overrides the RESULTS material
  GS_RP_SKIN (0) 1 → hide tiles, drape the SurfaceSkin instead
  GS_RP_OUT      output dir (default dated)
"""
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

_now = datetime.datetime.now()
OUT = os.environ.get("GS_RP_OUT", os.path.join(
    "data", _now.strftime("%Y-%m-%d"), _now.strftime("%H%M") + "-render_pick"))
SKIN = os.environ.get("GS_RP_SKIN", "0") not in ("0", "", "false")


def _load_theta_mat():
    import tools.pick_lab as PL
    theta, mat = None, None
    rp = os.environ.get("GS_RP_RESULTS", "")
    if rp and os.path.isfile(rp):
        res = json.load(open(rp))
        name = os.environ.get("GS_RP_MATNAME", next(iter(res)))
        theta = res[name]["best"].get("theta")
        mat = res[name]["material"]
    ts = os.environ.get("GS_RP_THETA", "")
    if ts:
        theta = [float(v) for v in ts.split(",")]
    ms = os.environ.get("GS_RP_MAT", "")
    if ms:
        mat = json.loads(ms)
    if theta is None:
        theta = list(PL.TH_MU0)
    if mat is None:
        mat = dict(PL.DEFAULT_MATS[0])
    return np.asarray(theta, dtype=np.float64), mat


def main():
    os.environ.setdefault("GS_SOFT", "0")
    os.makedirs(OUT, exist_ok=True)

    from graspsort.sim_env import GraspSortEnv
    env = GraspSortEnv(headless=True, width=960, height=540)
    try:
        env.world.get_physics_context().set_physics_dt(1.0 / 240.0)
    except Exception as e:
        print(f"[rp] physics_dt: {e}", flush=True)
    env.reset_world()
    stage = env.stage

    from pxr import UsdGeom, UsdShade, Gf, Sdf

    import tools.pick_lab as PL
    wc = env._work_centre_from_stage()
    SURF = env.table_top_z + 0.030          # clear of the slab (render_foundation)
    PL.SURF_Z = SURF                        # rebind the lab's surface height
    CENTRE = (wc[0], wc[1])
    theta, mat = _load_theta_mat()
    print(f"[rp] mat={mat}", flush=True)
    print(f"[rp] theta={[round(float(v), 4) for v in theta]}", flush=True)

    rig = PL.PickRig(stage, 0, CENTRE, mat)
    rig.base = "/World/PickRig"
    rig.build()

    # ── looks (OmniPBR renders in the full env) ──────────────────────────────
    UsdGeom.Scope.Define(stage, "/World/RPLooks")

    def omnipbr(name, rgb, rough=0.6, metallic=0.0):
        mp = f"/World/RPLooks/{name}"
        m = UsdShade.Material.Define(stage, mp)
        sh = UsdShade.Shader.Define(stage, mp + "/Shader")
        sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
        sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        sh.CreateInput("diffuse_color_constant",
                       Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput("reflection_roughness_constant",
                       Sdf.ValueTypeNames.Float).Set(rough)
        sh.CreateInput("metallic_constant",
                       Sdf.ValueTypeNames.Float).Set(metallic)
        m.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
        return m

    tileA = omnipbr("TileA", (0.82, 0.68, 0.46))
    tileB = omnipbr("TileB", (0.52, 0.40, 0.24))
    skin = None
    if SKIN:
        from graspsort.soft_foundation import SurfaceSkin
        skin = SurfaceSkin(rig.found, path="/World/PickSkin",
                           color=(0.86, 0.64, 0.46)).build()
        tex = os.environ.get(
            "GS_RP_SKIN_TEX",
            os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "assets", "grid_10mm.png"))
        if tex in ("0", "none"):
            tex = ""
        if tex:
            mp = "/World/PickLooks/SkinTex"
            m = UsdShade.Material.Define(stage, mp)
            st = UsdShade.Shader.Define(stage, mp + "/stReader")
            st.CreateIdAttr("UsdPrimvarReader_float2")
            st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            tx = UsdShade.Shader.Define(stage, mp + "/tex")
            tx.CreateIdAttr("UsdUVTexture")
            tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
                os.path.abspath(tex))
            tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                st.ConnectableAPI(), "result")
            sh = UsdShade.Shader.Define(stage, mp + "/pbr")
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("diffuseColor",
                           Sdf.ValueTypeNames.Color3f).ConnectToSource(
                tx.ConnectableAPI(), "rgb")
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
            m.CreateSurfaceOutput().ConnectToSource(
                sh.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(
                stage.GetPrimAtPath(skin.path)).Bind(m)
        else:
            UsdShade.MaterialBindingAPI.Apply(
                stage.GetPrimAtPath(skin.path)).Bind(
                    omnipbr("Skin", (0.86, 0.64, 0.46), rough=0.75))
    else:
        n = rig.found._grid_n
        for idx, tp in enumerate(rig.found.tile_paths):
            i, j = divmod(idx, n)
            geo = stage.GetPrimAtPath(tp + "/geo")
            if geo.IsValid():
                UsdGeom.Imageable(geo).MakeVisible()
                UsdShade.MaterialBindingAPI.Apply(geo.GetPrim()).Bind(
                    tileA if (i + j) % 2 == 0 else tileB)

    UsdShade.MaterialBindingAPI.Apply(
        stage.GetPrimAtPath(rig.wpath)).Bind(
            omnipbr("WasherRed", (0.86, 0.14, 0.10), rough=0.5),
            UsdShade.Tokens.strongerThanDescendants)
    UsdShade.MaterialBindingAPI.Apply(
        stage.GetPrimAtPath(rig.base + "/FingerA/geo")).Bind(
            omnipbr("FingA", (0.16, 0.18, 0.22), rough=0.4))
    UsdShade.MaterialBindingAPI.Apply(
        stage.GetPrimAtPath(rig.base + "/FingerB/geo")).Bind(
            omnipbr("FingB", (0.12, 0.30, 0.55), rough=0.4))

    env.world.reset()
    rig.reset(theta)

    # ── camera: proven far 3/4 view + centre crop (render_foundation recipe) ──
    from render_review import look_at_quat
    from isaacsim.sensors.camera import Camera
    import cv2
    tgt = (CENTRE[0], CENTRE[1], SURF + 0.004)
    eye = (CENTRE[0] + 0.62, CENTRE[1] - 0.72, SURF + 0.58)
    cam = Camera(prim_path="/World/RPCam", position=np.array(eye), frequency=30,
                 resolution=(1280, 720), orientation=look_at_quat(eye, tgt))
    cam.initialize()
    for _ in range(20):
        env.step(render=True)

    def _crop(fr):
        h, w = fr.shape[:2]
        cx_, cy_ = int(w * 0.50), int(h * 0.52)
        hw, hh = int(w * 0.24), int(h * 0.24)
        return fr[max(0, cy_ - hh):cy_ + hh, max(0, cx_ - hw):cx_ + hw]

    def write(name, fr):
        cv2.imwrite(os.path.join(OUT, name), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        sub = _crop(fr)
        if sub.size:
            cv2.imwrite(os.path.join(OUT, name.replace(".png", "_zoom.png")),
                        cv2.cvtColor(sub, cv2.COLOR_RGB2BGR))

    frames = []
    stills = {  # sim-time → still name (phase boundaries)
        PL.T_SET - 0.02: "p0_rest.png",
        PL.T_SET + PL.T_PRS - 0.02: "p1_pressed.png",
        PL.T_SET + PL.T_PRS + PL.T_BDN + PL.T_DRG - 0.02: "p2_engaged.png",
        PL.T_SET + PL.T_PRS + PL.T_BDN + PL.T_DRG + PL.T_LFT - 0.02: "p3_lifted.png",
        PL.T_TOTAL - 0.02: "p4_final.png",
    }
    done_stills = set()
    rp_tracing = os.environ.get("GS_RP_TRACE", "0") != "0"
    rp_trace = []
    t_total = (PL.T_TOTAL_P + PL.T_CARRY + 0.2
               if mat.get("mode") == "parallel" else PL.T_TOTAL)
    steps_total = int(t_total / PL.PHYS_DT)
    for s in range(steps_total):
        t = s * PL.PHYS_DT
        rig.drive(t)
        if rp_tracing and s % 12 == 0:
            (_wx, _wy, _wz), _tl = rig.washer_state()
            rp_trace.append((round(t, 3), round(rig._vc, 4),
                             round(float(_tl), 2),
                             round(1000 * (_wz - SURF), 2)))
        render = (s % 6 == 0)
        env.step(render=render)
        if render:
            if skin is not None:
                skin.update()
            rgba = cam.get_rgba()
            if rgba is not None and getattr(rgba, "size", 0) > 0:
                fr = np.asarray(rgba)[:, :, :3].astype(np.uint8).copy()
                (_, _, wz), tilt = rig.washer_state()
                cv2.putText(fr, f"t={t:4.1f}s tilt={tilt:3.0f} "
                            f"z={1000 * (wz - SURF):+.0f}mm",
                            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (245, 245, 245), 2, cv2.LINE_AA)
                frames.append(_crop(fr))
                for st_t, name in stills.items():
                    if name not in done_stills and t >= st_t:
                        done_stills.add(name)
                        write(name, fr)
        if s % 240 == 0:
            (_, _, wz), tilt = rig.washer_state()
            print(f"[rp] t={t:4.1f} tilt={tilt:5.1f} wz={1000 * (wz - SURF):+.1f}mm",
                  flush=True)

    sc = rig.final_score()
    print(f"[rp] RESULT {json.dumps(sc)}", flush=True)
    with open(os.path.join(OUT, "replay.json"), "w") as f:
        json.dump(dict(theta=list(map(float, theta)), material=mat, score=sc),
                  f, indent=2)
    if rp_tracing:
        json.dump(rp_trace, open(os.path.join(OUT, "trace.json"), "w"))
        print(f"[rp] trace.json ({len(rp_trace)} samples)", flush=True)

    if frames:
        h, w = frames[0].shape[:2]
        mp4 = os.path.join(OUT, "pick.mp4")
        vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), 24, (w, h))
        for fr in frames:
            vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        vw.release()
        try:
            from graspsort.videoio import to_h264
            to_h264(mp4)   # browsers won't play raw mp4v — report videos must be H.264
        except Exception as e:
            print(f"[rp] h264 re-encode skipped: {e}", flush=True)
        print(f"[rp] wrote {mp4} ({len(frames)} frames)", flush=True)
    print(f"[rp] DONE -> {OUT}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
