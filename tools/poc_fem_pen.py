#!/usr/bin/env python3
"""
FEM POC-2b (David, 2026-07-05): during the press the washer visually sinks
INTO the pad. Quantify true interpenetration = washer probe points vs the
LOCAL deformed surface height (interpolated from sim-mesh top vertices), and
find the mesh fineness that keeps it acceptable.

Matrix: src tessellation {16, 24, 32} divisions (= 5.0 / 3.3 / 2.5 mm cells
on the 80 mm pad, hex sim res matched). Two acts, mirroring render_fem:
  act 1 canonical press (A+B to -3 mm, ao 0.70)
  act 2 deep 6 mm press right beside the washer (the tilt David watched)
Metrics per act: max penetration (mm), plus wall-time per 1000 steps.

    docker/run.sh tools/poc_fem_pen.py
Env: GS_POC_OUT
"""
import datetime
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

_now = datetime.datetime.now()
OUT = os.environ.get("GS_POC_OUT", os.path.join(
    "data", _now.strftime("%Y-%m-%d"), "poc_fem_pen"))

SURF_Z = 0.50
PAD = (0.080, 0.080, 0.010)
W_R, W_T = 0.012, 0.0025
FW, FH = 0.008, 0.030
A_OVER, G0, PRESS = 0.70, 0.028, 0.003
T_SET, T_DSC, T_HOLD = 0.5, 1.2, 0.6
PHYS_DT = 1.0 / 240.0
E_PAD = 0.2e6


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def make_box_mesh(stage, path, size, center, div=(12, 12, 2)):
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
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})

    import omni.usd
    from isaacsim.core.api import World
    from pxr import UsdGeom, UsdPhysics, UsdShade, Gf, PhysxSchema
    from omni.physx.scripts import deformableUtils, physicsUtils
    from graspsort import parts

    os.makedirs(OUT, exist_ok=True)
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")

    g = UsdGeom.Cube.Define(stage, "/World/Ground")
    g.GetSizeAttr().Set(1.0)
    UsdGeom.XformCommonAPI(g.GetPrim()).SetTranslate(
        Gf.Vec3d(0, 0, SURF_Z - PAD[2] - 0.05))
    UsdGeom.XformCommonAPI(g.GetPrim()).SetScale(Gf.Vec3f(0.5, 0.5, 0.1))
    UsdPhysics.CollisionAPI.Apply(g.GetPrim())

    world = World(physics_dt=PHYS_DT, rendering_dt=1.0 / 60.0,
                  stage_units_in_meters=1.0)
    pc = world.get_physics_context()
    pc.enable_gpu_dynamics(True)
    pc.set_broadphase_type("GPU")

    def rigid_mat(path, fric):
        m = UsdShade.Material.Define(stage, path)
        api = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
        api.CreateStaticFrictionAttr().Set(fric)
        api.CreateDynamicFrictionAttr().Set(fric)
        api.CreateRestitutionAttr().Set(0.0)
        return m

    def finger(name, x, y, fric):
        p = f"/World/{name}"
        if stage.GetPrimAtPath(p).IsValid():
            stage.RemovePrim(p)
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
        pca = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
        pca.CreateContactOffsetAttr().Set(0.0015)
        pca.CreateRestOffsetAttr().Set(0.0)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
            rigid_mat(p + "/Mat", fric),
            UsdShade.Tokens.weakerThanDescendants, "physics")
        return op

    # thin sheet on rigid base: press bottoms out under the near rim ->
    # fulcrum -> far rim must rise (the real bench IS thin neoprene on a
    # table; the 10mm pad let the washer sink bodily with no see-saw)
    CONFIGS = [
        dict(tag="iters64", div=32, th=0.010, nu=0.3, E=0.2e6, wmass=None,
             slow=1.0, iters=64, dens=300),
        dict(tag="dens1200", div=32, th=0.010, nu=0.3, E=0.2e6, wmass=None,
             slow=1.0, iters=None, dens=1200),
        dict(tag="m20_it32", div=32, th=0.010, nu=0.3, E=0.2e6, wmass=0.020,
             slow=1.0, iters=32, dens=300),
        dict(tag="hz480it32", div=32, th=0.010, nu=0.3, E=0.2e6, wmass=None,
             slow=1.0, iters=32, dens=300, dt480=True),
    ]
    for c in CONFIGS:
        c["dt"] = 1.0 / 480.0 if c.get("dt480") else 1.0 / 240.0
    results = []
    for cfg in CONFIGS:
        div = cfg["div"]
        for pth in ("/World/Pad", "/World/Washer", "/World/PadMat"):
            if stage.GetPrimAtPath(pth).IsValid():
                stage.RemovePrim(pth)
        th = cfg["th"]
        pad_sz = (PAD[0], PAD[1], th)
        UsdGeom.XformCommonAPI(g.GetPrim()).SetTranslate(
            Gf.Vec3d(0, 0, SURF_Z - th - 0.05))
        UsdGeom.Xform.Define(stage, "/World/Pad")
        make_box_mesh(stage, "/World/Pad/src", pad_sz,
                      (0, 0, SURF_Z - th / 2),
                      div=(div, div, max(2, int(th / 0.0017))))
        ok = deformableUtils.create_auto_volume_deformable_hierarchy(
            stage, "/World/Pad", "/World/Pad/sim", "/World/Pad/col",
            "/World/Pad/src",
            simulation_hex_mesh_enabled=True,
            cooking_src_simplification_enabled=False)
        if not ok:
            print(f"[pen] hierarchy FAILED div={div}", flush=True)
            continue
        rootp = stage.GetPrimAtPath("/World/Pad")
        for a in rootp.GetAttributes():
            if "esolution" in a.GetName():
                a.Set(div)
        matp = "/World/PadMat"
        deformableUtils.add_deformable_material(
            stage, matp, density=float(cfg["dens"]),
            static_friction=0.9, dynamic_friction=0.9,
            youngs_modulus=cfg["E"], poissons_ratio=cfg["nu"])
        world.get_physics_context().set_physics_dt(cfg["dt"])
        if cfg is CONFIGS[0]:
            oa = [a.GetName() for a in rootp.GetAttributes()]
            print(f"[pen] root deformable attrs: {oa}", flush=True)
        physicsUtils.add_physics_material_to_prim(
            stage, stage.GetPrimAtPath("/World/Pad/sim"), matp)
        physicsUtils.add_physics_material_to_prim(
            stage, stage.GetPrimAtPath("/World/Pad/col"), matp)
        pca = PhysxSchema.PhysxCollisionAPI.Apply(
            stage.GetPrimAtPath("/World/Pad/col"))
        pca.CreateContactOffsetAttr().Set(0.002)
        pca.CreateRestOffsetAttr().Set(0.0)

        spec = parts.PartSpec(kind="washer", size="m12", pose="flat",
                              xy=(0.0, 0.0), rotz_deg=0.0)
        wpath = parts.spawn_part(stage, "/World/Washer", spec,
                                 SURF_Z + 0.002)
        if cfg.get("wmass"):
            UsdPhysics.MassAPI.Apply(
                stage.GetPrimAtPath(wpath)).CreateMassAttr().Set(
                    cfg["wmass"])
        if cfg.get("iters"):
            prb = PhysxSchema.PhysxRigidBodyAPI.Apply(
                stage.GetPrimAtPath(wpath))
            prb.CreateSolverPositionIterationCountAttr().Set(cfg["iters"])
            prb.CreateSolverVelocityIterationCountAttr().Set(8)
            prb.CreateMaxDepenetrationVelocityAttr().Set(10.0)

        a_c = -W_R - (A_OVER - 0.5) * FW
        fa = finger("FingerA", a_c, 0.0, 0.10)
        fb = finger("FingerB", a_c + G0, 0.0, 1.20)
        hi_z = SURF_Z + 0.06 + FH / 2
        press_z = SURF_Z - PRESS + FH / 2
        deep_z = SURF_Z - 0.006 + FH / 2

        world.reset()
        wprim = stage.GetPrimAtPath(wpath)
        simp = stage.GetPrimAtPath("/World/Pad/sim")

        # top-surface vertex set (rest z within 1mm of the pad top)
        pts0 = np.array(simp.GetAttribute("points").Get())
        top_idx = np.where(pts0[:, 2] > SURF_Z - 0.001)[0]
        print(f"[pen] div={div}: {len(pts0)} sim pts, "
              f"{len(top_idx)} top verts", flush=True)

        def local_surface_z(x, y, pts):
            """IDW over the 4 nearest top vertices in XY."""
            top = pts[top_idx]
            d2 = (top[:, 0] - x) ** 2 + (top[:, 1] - y) ** 2
            k = np.argsort(d2)[:4]
            w = 1.0 / np.maximum(d2[k], 1e-10)
            return float(np.sum(top[k, 2] * w) / np.sum(w))

        probes_l = [(-W_R + 0.001, 0.0, -W_T / 2),
                    (0.0, 0.0, -W_T / 2),
                    (W_R - 0.001, 0.0, -W_T / 2)]

        def penetration():
            xc = UsdGeom.XformCache()
            M = xc.GetLocalToWorldTransform(wprim)
            pts = np.array(simp.GetAttribute("points").Get())
            worst = -99.0
            for lp in probes_l:
                wp = M.Transform(Gf.Vec3d(*lp))
                sz = local_surface_z(wp[0], wp[1], pts)
                worst = max(worst, (sz - wp[2]) * 1000.0)
            return worst

        def dish():
            pts = np.array(simp.GetAttribute("points").Get())
            return (SURF_Z - pts[top_idx][:, 2].min()) * 1000.0

        profiles = []

        def snap_profile(label):
            pts = np.array(simp.GetAttribute("points").Get())
            top = pts[top_idx]
            row = top[np.abs(top[:, 1]) < 0.0015]
            row = row[np.argsort(row[:, 0])]
            xc = UsdGeom.XformCache()
            M = xc.GetLocalToWorldTransform(wprim)
            wr = [list(M.Transform(Gf.Vec3d(lx, 0, lz)))
                  for lx, lz in ((-W_R, -W_T / 2), (W_R, -W_T / 2),
                                 (W_R, W_T / 2), (-W_R, W_T / 2))]
            profiles.append(dict(label=label,
                                 x=row[:, 0].tolist(),
                                 z=[(v - SURF_Z) * 1000 for v in
                                    row[:, 2].tolist()],
                                 washer=[[p[0], (p[2] - SURF_Z) * 1000]
                                         for p in wr]))

        def far_rim_z():
            xc = UsdGeom.XformCache()
            M = xc.GetLocalToWorldTransform(wprim)
            wp = M.Transform(Gf.Vec3d(W_R - 0.001, 0.0, W_T / 2))
            return (wp[2] - SURF_Z) * 1000.0

        def run_phase(duration, drive, label):
            steps = int(duration / cfg['dt'])
            mx, dsh, rise = -99.0, 0.0, -99.0
            t0 = time.time()
            for s in range(steps):
                if s % 4 == 0:
                    drive(s * cfg['dt'] / duration)
                world.step(render=False)
                if s % 24 == 0:
                    mx = max(mx, penetration())
                    dsh = max(dsh, dish())
                    rise = max(rise, far_rim_z())
            dt_wall = (time.time() - t0) / steps * 1000
            print(f"[pen]   {label}: pen={mx:5.2f}mm dish={dsh:5.2f}mm "
                  f"far_rim_max={rise:+6.2f}mm", flush=True)
            return mx, dt_wall

        for _ in range(int(0.5 / cfg['dt'])):
            world.step(render=False)
        pen_rest = penetration()

        # act 0: press the BARE pad (x=+25mm, clear of the washer) —
        # isolates rigid<->deformable contact from the washer transmission
        def drv_bare(u):
            z = hi_z + (press_z - hi_z) * ease(u)
            fa.SetTranslate(Gf.Vec3d(0.025, 0.0, z))
            fb.SetTranslate(Gf.Vec3d(0.025 + G0, 0.0, z))

        run_phase(1.2, drv_bare, "bare-pad press 3mm")

        def drv_bareup(u):
            z = press_z + (hi_z - press_z) * ease(u)
            fa.SetTranslate(Gf.Vec3d(0.025, 0.0, z))
            fb.SetTranslate(Gf.Vec3d(0.025 + G0, 0.0, z))

        run_phase(0.5, drv_bareup, "bare lift")
        fa.SetTranslate(Gf.Vec3d(a_c, 0.0, hi_z))
        fb.SetTranslate(Gf.Vec3d(a_c + G0, 0.0, hi_z))
        for _ in range(int(0.3 / cfg['dt'])):
            world.step(render=False)

        def drv_press(u):
            z = hi_z + (press_z - hi_z) * ease(u)
            fa.SetTranslate(Gf.Vec3d(a_c, 0.0, z))
            fb.SetTranslate(Gf.Vec3d(a_c + G0, 0.0, z))

        snap_profile("rest")
        dur_p = (T_DSC + T_HOLD) * cfg["slow"]

        def drv_press_s(u):
            drv_press(min(1.0, u * dur_p / (T_DSC * cfg["slow"])))
            t_now = u * dur_p
            for frac, lbl in ((0.33, "press 1mm"), (0.66, "press 2mm"),
                              (1.0, "press 3mm")):
                tgt = T_DSC * cfg["slow"] * frac
                if abs(t_now - tgt) < cfg["dt"] * 2.5 and \
                        not any(pr["label"] == lbl for pr in profiles):
                    snap_profile(lbl)

        pen_p, ms1 = run_phase(dur_p, drv_press_s, "press")
        snap_profile("hold end")

        def drv_lift(u):
            z = press_z + (hi_z - press_z) * ease(u)
            fa.SetTranslate(Gf.Vec3d(a_c, 0.0, z))
            fb.SetTranslate(Gf.Vec3d(a_c + G0, 0.0, z))

        run_phase(0.6, drv_lift, "lift")

        def drv_move(u):
            x = a_c + ((-0.010) - a_c) * ease(u)
            fa.SetTranslate(Gf.Vec3d(x, 0.0, hi_z))
            fb.SetTranslate(Gf.Vec3d(x + G0, 0.0, hi_z))

        run_phase(0.5, drv_move, "move")

        def drv_deep(u):
            z = hi_z + (deep_z - hi_z) * ease(u)
            fa.SetTranslate(Gf.Vec3d(-0.010, 0.0, z))
            fb.SetTranslate(Gf.Vec3d(-0.010 + G0, 0.0, z))

        pen_d, ms2 = run_phase(2.0, drv_deep, "deep")

        json.dump(profiles, open(os.path.join(
            OUT, f"profiles_{cfg['tag']}.json"), "w"))
        r = dict(tag=cfg["tag"], div=div,
                 cell_mm=round(PAD[0] / div * 1000, 1),
                 pen_rest=round(pen_rest, 2),
                 pen_press=round(pen_p, 2), pen_deep=round(pen_d, 2),
                 ms_per_step=round((ms1 + ms2) / 2, 2))
        results.append(r)
        print(f"[pen] {cfg['tag']:8s} cell={r['cell_mm']}mm "
              f"pen rest={pen_rest:5.2f} press={pen_p:5.2f} "
              f"deep={pen_d:5.2f}mm  {r['ms_per_step']:.2f}ms/step",
              flush=True)
        world.stop()

    json.dump(results, open(os.path.join(OUT, "pen.json"), "w"), indent=1)
    print(f"[pen] DONE -> {OUT}", flush=True)
    app.close()


if __name__ == "__main__":
    main()
