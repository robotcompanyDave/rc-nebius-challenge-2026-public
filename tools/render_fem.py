#!/usr/bin/env python3
"""
Render the FEM deformable pad (POC-2 v4 PASS recipe) INSIDE the full
GraspSortEnv boot — the canonical press + a deep center press, stills + video.
(Standalone scenes don't render dynamic prims; the full env does. Deformables
need the GPU pipeline, enabled here before reset.)

    docker/run.sh tools/render_fem.py
Env: GS_RF_OUT (dated dir), GS_FEM_E (0.2e6)
"""
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

PAD = (0.080, 0.080, 0.010)
W_R, W_T = 0.012, 0.0025
FW, FH = 0.008, 0.030
A_OVER, G0, PRESS = 0.70, 0.028, 0.003
T_SET, T_DSC, T_HOLD = 0.5, 1.2, 1.0
E_PAD = float(os.environ.get("GS_FEM_E", "0.2e6"))
_now = datetime.datetime.now()
OUT = os.environ.get("GS_RF_OUT", os.path.join(
    "data", _now.strftime("%Y-%m-%d"), _now.strftime("%H%M") + "-render_fem"))


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def make_box_mesh(stage, path, size, center, div=(12, 12, 2)):
    """Watertight tessellated box — src tessellation IS the collision
    resolution knob for the cooked deformable (POC-2 v4)."""
    from pxr import UsdGeom, Gf
    nx, ny, nz = div
    cx, cy, cz = center
    vid, pts = {}, []

    def vert(i, j, k):
        key = (i, j, k)
        if key not in vid:
            vid[key] = len(pts)
            pts.append(Gf.Vec3f(cx + (i / nx - 0.5) * size[0],
                                cy + (j / ny - 0.5) * size[1],
                                cz + (k / nz - 0.5) * size[2]))
        return vid[key]

    idx, cnt = [], []

    def quad(a, b, c, d):
        idx.extend([a, b, c, a, c, d])
        cnt.extend([3, 3])

    for i in range(nx):
        for j in range(ny):
            quad(vert(i, j, nz), vert(i + 1, j, nz),
                 vert(i + 1, j + 1, nz), vert(i, j + 1, nz))
            quad(vert(i, j, 0), vert(i, j + 1, 0),
                 vert(i + 1, j + 1, 0), vert(i + 1, j, 0))
    for i in range(nx):
        for k in range(nz):
            quad(vert(i, 0, k), vert(i + 1, 0, k),
                 vert(i + 1, 0, k + 1), vert(i, 0, k + 1))
            quad(vert(i, ny, k), vert(i, ny, k + 1),
                 vert(i + 1, ny, k + 1), vert(i + 1, ny, k))
    for j in range(ny):
        for k in range(nz):
            quad(vert(0, j, k), vert(0, j, k + 1),
                 vert(0, j + 1, k + 1), vert(0, j + 1, k))
            quad(vert(nx, j, k), vert(nx, j + 1, k),
                 vert(nx, j + 1, k + 1), vert(nx, j, k + 1))
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(pts)
    m.CreateFaceVertexIndicesAttr(idx)
    m.CreateFaceVertexCountsAttr(cnt)
    return m


def main():
    os.environ.setdefault("GS_SOFT", "0")
    os.makedirs(OUT, exist_ok=True)

    from graspsort.sim_env import GraspSortEnv
    env = GraspSortEnv(headless=True, width=960, height=540)
    pcx = env.world.get_physics_context()
    pcx.set_physics_dt(1.0 / 240.0)
    # deformables REQUIRE the GPU pipeline
    pcx.enable_gpu_dynamics(True)
    pcx.set_broadphase_type("GPU")
    env.reset_world()
    stage = env.stage

    from pxr import UsdGeom, UsdPhysics, UsdShade, Gf, PhysxSchema
    from omni.physx.scripts import deformableUtils, physicsUtils
    from graspsort import parts

    wc = env._work_centre_from_stage()
    CX, CY = wc[0], wc[1]
    PAD_BOT = env.table_top_z + 0.001
    SURF_Z = PAD_BOT + PAD[2]
    print(f"[fem] centre=({CX:.3f},{CY:.3f}) surf_z={SURF_Z:.3f}", flush=True)

    UsdGeom.Scope.Define(stage, "/World/FemLooks")

    def omnipbr(name, rgb, rough=0.6):
        mp = f"/World/FemLooks/{name}"
        m = UsdShade.Material.Define(stage, mp)
        sh = UsdShade.Shader.Define(stage, mp + "/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*rgb))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
        m.CreateSurfaceOutput().ConnectToSource(
            sh.ConnectableAPI(), "surface")
        return m

    from pxr import Sdf

    # ── FEM pad (POC-2 v4 recipe) ────────────────────────────────────────────
    UsdGeom.Xform.Define(stage, "/World/FemPad")
    make_box_mesh(stage, "/World/FemPad/src", PAD,
                  (CX, CY, PAD_BOT + PAD[2] / 2), div=(16, 16, 3))
    ok = deformableUtils.create_auto_volume_deformable_hierarchy(
        stage, "/World/FemPad", "/World/FemPad/sim", "/World/FemPad/col",
        "/World/FemPad/src",
        simulation_hex_mesh_enabled=True,
        cooking_src_simplification_enabled=False)
    print(f"[fem] hierarchy ok={ok}", flush=True)
    rootp = stage.GetPrimAtPath("/World/FemPad")
    for a in rootp.GetAttributes():
        if "esolution" in a.GetName():
            a.Set(16)
    matp = "/World/FemPadMat"
    deformableUtils.add_deformable_material(
        stage, matp, density=300.0,
        static_friction=0.9, dynamic_friction=0.9,
        youngs_modulus=E_PAD, poissons_ratio=0.3)
    physicsUtils.add_physics_material_to_prim(
        stage, stage.GetPrimAtPath("/World/FemPad/sim"), matp)
    physicsUtils.add_physics_material_to_prim(
        stage, stage.GetPrimAtPath("/World/FemPad/col"), matp)
    pca = PhysxSchema.PhysxCollisionAPI.Apply(
        stage.GetPrimAtPath("/World/FemPad/col"))
    pca.CreateContactOffsetAttr().Set(0.002)
    pca.CreateRestOffsetAttr().Set(0.0)
    # neither cooked mesh renders deformed: src stays flat in USD, and the
    # sim prim is a TetMesh (not drawn by Hydra). But the sim points DO
    # update in USD (POC-1 tracked them) — so build our own graphics skin
    # from the sim tetmesh surface triangles and re-point it every frame.
    UsdGeom.Imageable(stage.GetPrimAtPath("/World/FemPad/src")) \
        .MakeInvisible()
    simp = stage.GetPrimAtPath("/World/FemPad/sim")
    skin_mesh = UsdGeom.Mesh.Define(stage, "/World/FemSkin")
    UsdShade.MaterialBindingAPI.Apply(skin_mesh.GetPrim()).Bind(
        omnipbr("Foam", (0.86, 0.64, 0.46), rough=0.8))

    def skin_init():
        pts = simp.GetAttribute("points").Get()
        sfi = simp.GetAttribute("surfaceFaceVertexIndices").Get()
        if not sfi:
            # cooked surface list not populated — derive the boundary from
            # the tets: faces that appear exactly once are the surface
            tets = simp.GetAttribute("tetVertexIndices").Get()
            from collections import Counter
            faces = []
            for t in tets:                       # Vec4i per tet
                a, b, c, d = int(t[0]), int(t[1]), int(t[2]), int(t[3])
                faces += [(a, b, c), (a, c, d), (a, d, b), (b, d, c)]
            cnt = Counter(tuple(sorted(f)) for f in faces)
            sfi = [v for f in faces if cnt[tuple(sorted(f))] == 1
                   for v in f]
        skin_mesh.CreatePointsAttr(pts)
        skin_mesh.CreateFaceVertexIndicesAttr(list(sfi))
        skin_mesh.CreateFaceVertexCountsAttr([3] * (len(sfi) // 3))
        skin_mesh.CreateDoubleSidedAttr(True)
        print(f"[fem] skin: {len(pts)} pts, {len(sfi)//3} tris", flush=True)

    def skin_update():
        skin_mesh.GetPointsAttr().Set(simp.GetAttribute("points").Get())

    # ── washer ───────────────────────────────────────────────────────────────
    spec = parts.PartSpec(kind="washer", size="m12", pose="flat",
                          xy=(CX, CY), rotz_deg=0.0)
    wpath = parts.spawn_part(stage, "/World/FemWasher", spec, SURF_Z + 0.002)
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(wpath)).Bind(
        omnipbr("WasherRed", (0.86, 0.14, 0.10), rough=0.5),
        UsdShade.Tokens.strongerThanDescendants)

    # ── fingers: hull meshes + friction (A slippery presser, B grippy blue) ──
    def rigid_mat(path, fric):
        m = UsdShade.Material.Define(stage, path)
        api = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
        api.CreateStaticFrictionAttr().Set(fric)
        api.CreateDynamicFrictionAttr().Set(fric)
        api.CreateRestitutionAttr().Set(0.0)
        return m

    def finger(name, x, y, fric, rgb):
        p = f"/World/{name}"
        xf = UsdGeom.Xform.Define(stage, p)
        op = UsdGeom.XformCommonAPI(xf.GetPrim())
        op.SetTranslate(Gf.Vec3d(x, y, SURF_Z + 0.06 + FH / 2))
        UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim()) \
            .CreateKinematicEnabledAttr(True)
        mesh = make_box_mesh(stage, p + "/geo", (FW, FW, FH), (0, 0, 0),
                             div=(1, 1, 1))
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()) \
            .CreateApproximationAttr().Set("convexHull")
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        pc = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
        pc.CreateContactOffsetAttr().Set(0.0015)
        pc.CreateRestOffsetAttr().Set(0.0)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
            rigid_mat(p + "/Mat", fric),
            UsdShade.Tokens.weakerThanDescendants, "physics")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
            omnipbr(name + "Look", rgb, rough=0.4))
        return op

    a_c = CX - W_R - (A_OVER - 0.5) * FW
    fa = finger("FemFingerA", a_c, CY, 0.10, (0.16, 0.18, 0.22))
    fb = finger("FemFingerB", a_c + G0, CY, 1.20, (0.15, 0.35, 0.90))
    hi_z = SURF_Z + 0.06 + FH / 2
    press_z = SURF_Z - PRESS + FH / 2

    env.world.reset()
    skin_init()

    # ── camera + capture (render_review recipe) ─────────────────────────────
    from render_review import look_at_quat
    from isaacsim.sensors.camera import Camera
    import cv2
    tgt = (CX, CY, SURF_Z + 0.002)
    eye = (CX + 0.62, CY - 0.72, SURF_Z + 0.58)
    cam = Camera(prim_path="/World/FemCam", position=np.array(eye),
                 frequency=30, resolution=(1280, 720),
                 orientation=look_at_quat(eye, tgt))
    cam.initialize()
    for _ in range(20):
        env.step(render=True)

    def _crop(fr):
        h, w = fr.shape[:2]
        cx, cy = int(w * 0.50), int(h * 0.52)
        hw, hh = int(w * 0.22), int(h * 0.22)
        return fr[max(0, cy - hh):cy + hh, max(0, cx - hw):cx + hw]

    def shot(label):
        rgba = None
        for _ in range(12):
            skin_update()
            env.step(render=True)
            rgba = cam.get_rgba()
            if rgba is not None and getattr(rgba, "size", 0) > 0:
                break
        if rgba is None or getattr(rgba, "size", 0) == 0:
            return None
        fr = np.asarray(rgba)[:, :, :3].astype(np.uint8).copy()
        cv2.putText(fr, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (245, 245, 245), 2, cv2.LINE_AA)
        return fr

    def write(name, fr):
        cv2.imwrite(os.path.join(OUT, name),
                    cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        sub = _crop(fr)
        if sub.size:
            cv2.imwrite(os.path.join(OUT, name.replace(".png", "_zoom.png")),
                        cv2.cvtColor(sub, cv2.COLOR_RGB2BGR))

    vid = []

    def grab(label):
        fr = shot(label)
        if fr is not None:
            vid.append(_crop(fr).copy())
        return fr

    # settle + rest still
    for _ in range(240):
        env.step(render=False)
    fr = shot("FEM pad  rest — washer sits, no tunnel (June: fell through)")
    if fr is not None:
        write("fem_rest.png", fr)

    # ── act 1: canonical press (the POC-2 deterministic press) ─────────────
    dt = 1.0 / 240.0
    steps = int((T_SET + T_DSC + T_HOLD) / dt)
    wprim = stage.GetPrimAtPath(wpath)
    for s in range(steps):
        t = s * dt
        if s % 4 == 0:
            if t < T_SET:
                z = hi_z
            elif t < T_SET + T_DSC:
                z = hi_z + (press_z - hi_z) * ease((t - T_SET) / T_DSC)
            else:
                z = press_z
            fa.SetTranslate(Gf.Vec3d(a_c, CY, z))
            fb.SetTranslate(Gf.Vec3d(a_c + G0, CY, z))
        env.step(render=False)
        if s % 16 == 0:
            grab(f"canonical press  t={t:.2f}s")
        if s == int((T_SET + T_DSC) / dt) + 60:
            fr = shot("FEM press hold — deterministic dish (spread 0.07mm)")
            if fr is not None:
                write("fem_press_hold.png", fr)

    # ── act 2: slow deep centre press to SHOW the smooth continuum dish ─────
    for s in range(int(0.8 / dt)):     # lift straight up first
        u = ease(s * dt / 0.8)
        z = press_z + (hi_z - press_z) * u
        if s % 4 == 0:
            fa.SetTranslate(Gf.Vec3d(a_c, CY, z))
            fb.SetTranslate(Gf.Vec3d(a_c + G0, CY, z))
        env.step(render=False)
        if s % 24 == 0:
            grab("lift")
    for s in range(int(0.7 / dt)):     # translate at height
        u = ease(s * dt / 0.7)
        x = a_c + ((CX - 0.010) - a_c) * u
        if s % 4 == 0:
            fa.SetTranslate(Gf.Vec3d(x, CY, hi_z))
            fb.SetTranslate(Gf.Vec3d(x + G0, CY, hi_z))
        env.step(render=False)
        if s % 24 == 0:
            grab("reposition")
    deep_z = SURF_Z - 0.006 + FH / 2
    for s in range(int(2.0 / dt)):
        u = ease(s * dt / 2.0)
        z = hi_z + (deep_z - hi_z) * u
        if s % 4 == 0:
            fa.SetTranslate(Gf.Vec3d(CX - 0.010, CY, z))
            fb.SetTranslate(Gf.Vec3d(CX - 0.010 + G0, CY, z))
        env.step(render=False)
        if s % 16 == 0:
            grab(f"deep 6mm dish  t={s*dt:.2f}s")
    fr = shot("FEM 6mm dish — one continuum, no tiles, no leveling hack")
    if fr is not None:
        write("fem_deep_dish.png", fr)
    for s in range(int(1.0 / dt)):     # release
        u = ease(s * dt / 1.0)
        z = deep_z + (hi_z - deep_z) * u
        if s % 4 == 0:
            fa.SetTranslate(Gf.Vec3d(CX - 0.010, CY, z))
            fb.SetTranslate(Gf.Vec3d(CX - 0.010 + G0, CY, z))
        env.step(render=False)
        if s % 16 == 0:
            grab("release — foam springs back")

    # video
    if vid:
        h, w = vid[0].shape[:2]
        w -= w % 2
        h -= h % 2
        vw = cv2.VideoWriter(os.path.join(OUT, "fem_poc.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
        for f in vid:
            vw.write(cv2.cvtColor(f[:h, :w], cv2.COLOR_RGB2BGR))
        vw.release()
        try:
            from graspsort.videoio import to_h264
            to_h264(os.path.join(OUT, "fem_poc.mp4"))
        except Exception as e:
            print(f"[fem] h264 re-encode failed: {e}", flush=True)
        print(f"[fem] wrote fem_poc.mp4 ({len(vid)} frames)", flush=True)

    print(f"[fem] DONE -> {OUT}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
