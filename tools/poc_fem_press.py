#!/usr/bin/env python3
"""
FEM POC-2 (fem-proposal report): canonical A-press at 70% overhang on the
deformable pad — is the far-rim response MONOTONE and POSITION-INVARIANT
(the tile model's pop/swallow bistability is the disease we're testing for)?

Matrix: E {0.2, 0.3 MPa} x 4 washer positions (sub-hex-cell phase shifts).
Pad recipe = POC-1 PASS cell: res 8, material on sim mesh, explicit offsets.
Fingers are CONVEX-HULL MESH boxes — POC-1 run 5 showed primitive colliders
tunnel through deformables even with explicit offsets.

    docker/run.sh tools/poc_fem_press.py
Env: GS_POC_OUT (dated dir)
"""
import datetime
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

_now = datetime.datetime.now()
OUT = os.environ.get("GS_POC_OUT", os.path.join(
    "data", _now.strftime("%Y-%m-%d"), "poc_fem_press"))

SURF_Z = 0.50
PAD = (0.080, 0.080, 0.010)
W_R, W_T = 0.012, 0.0025
FW, FH = 0.008, 0.030
A_OVER, G0, PRESS = 0.70, 0.028, 0.003      # dwell-winner press params
T_SET, T_DSC, T_HOLD = 0.5, 1.2, 1.0
PHYS_DT = 1.0 / 240.0

# sub-cell phase shifts (mm) — res 8 on an 80mm pad = 10mm hexes, so these
# sample distinct positions inside one hex cell (the FEM analogue of the
# tile model's scene-slot phase)
SHIFTS = [(0.0, 0.0), (0.0021, 0.0013), (0.0044, 0.0032), (0.0067, 0.0055)]


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def make_cube_mesh(stage, path, size, center, div=(12, 12, 2)):
    """Watertight TESSELLATED box (surface grid). An 8-vertex box source
    cooks into giant collision tets no matter the hex sim resolution — the
    src tessellation is the real collision-resolution knob (POC-2 v4)."""
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
                 vert(i + 1, j + 1, nz), vert(i, j + 1, nz))      # top +z
            quad(vert(i, j, 0), vert(i, j + 1, 0),
                 vert(i + 1, j + 1, 0), vert(i + 1, j, 0))        # bottom -z
    for i in range(nx):
        for k in range(nz):
            quad(vert(i, 0, k), vert(i + 1, 0, k),
                 vert(i + 1, 0, k + 1), vert(i, 0, k + 1))        # -y
            quad(vert(i, ny, k), vert(i, ny, k + 1),
                 vert(i + 1, ny, k + 1), vert(i + 1, ny, k))      # +y
    for j in range(ny):
        for k in range(nz):
            quad(vert(0, j, k), vert(0, j, k + 1),
                 vert(0, j + 1, k + 1), vert(0, j + 1, k))        # -x
            quad(vert(nx, j, k), vert(nx, j + 1, k),
                 vert(nx, j + 1, k + 1), vert(nx, j, k + 1))      # +x
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
    from pxr import UsdGeom, UsdPhysics, Gf, PhysxSchema
    from omni.physx.scripts import deformableUtils, physicsUtils

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

    from graspsort import parts

    def rigid_mat(path, fric):
        from pxr import UsdShade
        m = UsdShade.Material.Define(stage, path)
        mp = m.GetPrim()
        UsdPhysics.MaterialAPI.Apply(mp)
        api = UsdPhysics.MaterialAPI(mp)
        api.CreateStaticFrictionAttr().Set(fric)
        api.CreateDynamicFrictionAttr().Set(fric)
        api.CreateRestitutionAttr().Set(0.0)
        return m

    def finger(name, x, y, fric=0.5):
        """Kinematic finger with a CONVEX-HULL mesh collider (primitives
        don't contact deformables — POC-1 run 5)."""
        p = f"/World/{name}"
        if stage.GetPrimAtPath(p).IsValid():
            stage.RemovePrim(p)
        xf = UsdGeom.Xform.Define(stage, p)
        op = UsdGeom.XformCommonAPI(xf.GetPrim())
        op.SetTranslate(Gf.Vec3d(x, y, SURF_Z + 0.06 + FH / 2))
        UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim()) \
            .CreateKinematicEnabledAttr(True)
        mesh = make_cube_mesh(stage, p + "/geo", (FW, FW, FH), (0, 0, 0), div=(1, 1, 1))
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()) \
            .CreateApproximationAttr().Set("convexHull")
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        pca = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
        pca.CreateContactOffsetAttr().Set(0.0015)
        pca.CreateRestOffsetAttr().Set(0.0)
        from pxr import UsdShade
        mat = rigid_mat(p + "/Mat", fric)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, "physics")
        return op

    results = []
    for young in (0.2e6, 0.3e6):
        for si, (dx, dy) in enumerate(SHIFTS):
            for pth in ("/World/Pad", "/World/Washer", "/World/PadMat"):
                if stage.GetPrimAtPath(pth).IsValid():
                    stage.RemovePrim(pth)
            UsdGeom.Xform.Define(stage, "/World/Pad")
            make_cube_mesh(stage, "/World/Pad/src", PAD,
                           (0, 0, SURF_Z - PAD[2] / 2), div=(16, 16, 3))
            ok = deformableUtils.create_auto_volume_deformable_hierarchy(
                stage, "/World/Pad", "/World/Pad/sim", "/World/Pad/col",
                "/World/Pad/src",
                simulation_hex_mesh_enabled=True,
                cooking_src_simplification_enabled=False)
            if not ok:
                print(f"[poc2] hierarchy FAILED E={young}", flush=True)
                continue
            rootp = stage.GetPrimAtPath("/World/Pad")
            for a in rootp.GetAttributes():
                if "esolution" in a.GetName():
                    a.Set(16)
            matp = "/World/PadMat"
            deformableUtils.add_deformable_material(
                stage, matp, density=300.0,
                static_friction=0.9, dynamic_friction=0.9,
                youngs_modulus=young, poissons_ratio=0.3)
            physicsUtils.add_physics_material_to_prim(
                stage, stage.GetPrimAtPath("/World/Pad/sim"), matp)
            physicsUtils.add_physics_material_to_prim(
                stage, stage.GetPrimAtPath("/World/Pad/col"), matp)
            colp = stage.GetPrimAtPath("/World/Pad/col")
            pca = PhysxSchema.PhysxCollisionAPI.Apply(colp)
            pca.CreateContactOffsetAttr().Set(0.002)
            pca.CreateRestOffsetAttr().Set(0.0)

            spec = parts.PartSpec(kind="washer", size="m12", pose="flat",
                                  xy=(dx, dy), rotz_deg=0.0)
            wpath = parts.spawn_part(stage, "/World/Washer", spec,
                                     SURF_Z + 0.003)
            wprim0 = stage.GetPrimAtPath(wpath)
            prb = PhysxSchema.PhysxRigidBodyAPI.Apply(wprim0)
            prb.CreateSolverPositionIterationCountAttr().Set(32)
            prb.CreateSolverVelocityIterationCountAttr().Set(4)

            a_c = dx - W_R - (A_OVER - 0.5) * FW
            fa = finger("FingerA", a_c, dy, fric=0.10)
            fb = finger("FingerB", a_c + G0, dy, fric=1.20)
            hi_z = SURF_Z + 0.06 + FH / 2
            press_z = SURF_Z - PRESS + FH / 2

            world.reset()
            wprim = stage.GetPrimAtPath(wpath)
            traj = []          # (t, far_rim_mm) at 20 Hz
            steps = int((T_SET + T_DSC + T_HOLD) / PHYS_DT)
            for s in range(steps):
                t = s * PHYS_DT
                if s % 4 == 0:
                    if t < T_SET:
                        z = hi_z
                    elif t < T_SET + T_DSC:
                        z = hi_z + (press_z - hi_z) * ease((t - T_SET) / T_DSC)
                    else:
                        z = press_z
                    fa.SetTranslate(Gf.Vec3d(a_c, dy, z))
                    fb.SetTranslate(Gf.Vec3d(a_c + G0, dy, z))
                world.step(render=False)
                if t > T_SET and s % 12 == 0:
                    xc = UsdGeom.XformCache()
                    m = xc.GetLocalToWorldTransform(wprim)
                    rim = m.Transform(Gf.Vec3d(W_R - 0.001, 0.0, W_T / 2))
                    traj.append(round((rim[2] - SURF_Z) * 1000.0, 3))
            arr = np.array(traj)
            hold = arr[-18:]                     # last ~0.9 s of the hold
            dwell_ms = int(((arr >= 4.0) & (arr <= 6.0)).sum() * 50)
            r = dict(young_mpa=young / 1e6, shift_mm=[dx * 1000, dy * 1000],
                     hold_mean=round(float(hold.mean()), 2),
                     hold_std=round(float(hold.std()), 2),
                     max_rim=round(float(arr.max()), 2),
                     dwell_ms=dwell_ms, traj=traj)
            results.append(r)
            print(f"[poc2] E={young/1e6:.1f}MPa shift=({dx*1000:.1f},"
                  f"{dy*1000:.1f})mm hold={r['hold_mean']:6.2f}"
                  f"+-{r['hold_std']:.2f}mm max={r['max_rim']:6.2f} "
                  f"dwell={dwell_ms}ms", flush=True)
            world.stop()

    json.dump(results, open(os.path.join(OUT, "poc2.json"), "w"))
    for young in (0.2, 0.3):
        holds = [r["hold_mean"] for r in results if r["young_mpa"] == young]
        if holds:
            spread = max(holds) - min(holds)
            print(f"[poc2] E={young}MPa hold across positions: "
                  f"{holds} spread={spread:.2f}mm "
                  f"{'PASS' if spread < 1.0 else 'FAIL'} (<1mm gate)",
                  flush=True)
    print(f"[poc2] DONE -> {OUT}", flush=True)
    app.close()


if __name__ == "__main__":
    main()
