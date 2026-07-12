#!/usr/bin/env python3
"""
FEM POC-1 (fem-proposal report): does an m12 washer REST on a PhysX deformable
pad in the current stack, or does it tunnel through like it did in June?

Matrix: simulation resolution {8,12,16} x Young's modulus {0.1,0.3,1.0 MPa}.
Each cell: fresh pad + washer, settle 3 s, measure penetration; fall-through
if the washer sinks below half the pad. GPU pipeline (deformables are
GPU-only).

    docker/run.sh tools/poc_fem_rest.py       # push-grasp:dev image
Env: GS_POC_OUT (dated dir)
"""
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

_now = datetime.datetime.now()
OUT = os.environ.get("GS_POC_OUT", os.path.join(
    "data", _now.strftime("%Y-%m-%d"), "poc_fem_rest"))

SURF_Z = 0.50          # pad top
PAD = (0.060, 0.060, 0.010)
W_T = 0.0025


def make_cube_mesh(stage, path, size, center):
    """Closed triangle-mesh box (deformable source mesh)."""
    from pxr import UsdGeom, Gf
    hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2
    cx, cy, cz = center
    pts = [Gf.Vec3f(cx + sx * hx, cy + sy * hy, cz + sz * hz)
           for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    # cube faces as 12 triangles (indices into the 8 corners above)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    idx, cnt = [], []
    for a, b, c, d in quads:
        idx += [a, b, c, a, c, d]
        cnt += [3, 3]
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
    from pxr import UsdGeom, UsdPhysics, Gf
    from omni.physx.scripts import deformableUtils, physicsUtils

    os.makedirs(OUT, exist_ok=True)
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")

    # ground under the pad
    g = UsdGeom.Cube.Define(stage, "/World/Ground")
    g.GetSizeAttr().Set(1.0)
    UsdGeom.XformCommonAPI(g.GetPrim()).SetTranslate(
        Gf.Vec3d(0, 0, SURF_Z - PAD[2] - 0.05))
    UsdGeom.XformCommonAPI(g.GetPrim()).SetScale(Gf.Vec3f(0.5, 0.5, 0.1))
    UsdPhysics.CollisionAPI.Apply(g.GetPrim())

    world = World(physics_dt=1.0 / 240.0, rendering_dt=1.0 / 60.0,
                  stage_units_in_meters=1.0)
    # deformables REQUIRE the GPU pipeline
    pc = world.get_physics_context()
    pc.enable_gpu_dynamics(True)
    pc.set_broadphase_type("GPU")

    from graspsort import parts

    results = []
    for res in (8,):
        for young in (0.2e6, 0.3e6, 0.5e6):
            for pth in ("/World/Pad", "/World/Washer", "/World/PadMat"):
                if stage.GetPrimAtPath(pth).IsValid():
                    stage.RemovePrim(pth)
            # Sim 6.0 deformable API: root Xform + cooking-source mesh child;
            # sim/collision tetmeshes are auto-generated (hex sim mesh).
            UsdGeom.Xform.Define(stage, "/World/Pad")
            make_cube_mesh(stage, "/World/Pad/src", PAD,
                           (0, 0, SURF_Z - PAD[2] / 2))
            ok = deformableUtils.create_auto_volume_deformable_hierarchy(
                stage, "/World/Pad", "/World/Pad/sim", "/World/Pad/col",
                "/World/Pad/src",
                simulation_hex_mesh_enabled=True,
                cooking_src_simplification_enabled=False)
            if not ok:
                print(f"[poc1] hierarchy creation FAILED res={res}", flush=True)
                continue
            # resolution knob: PhysxAutoDeformableHexahedralMeshAPI sits on
            # the ROOT prim (deformableUtils applies it there, not on /sim)
            rootp = stage.GetPrimAtPath("/World/Pad")
            res_attrs = [a.GetName() for a in rootp.GetAttributes()
                         if "esolution" in a.GetName()]
            if res_attrs:
                rootp.GetAttribute(res_attrs[0]).Set(res)
            elif res == 8 and young == 0.1e6:
                print(f"[poc1] NOTE no resolution attr on root; attrs="
                      f"{[a.GetName() for a in rootp.GetAttributes()][:24]}",
                      flush=True)
            matp = "/World/PadMat"
            deformableUtils.add_deformable_material(
                stage, matp, density=300.0,
                static_friction=0.9, dynamic_friction=0.9,
                youngs_modulus=young, poissons_ratio=0.3)
            # material must bind to the SIMULATION mesh — PhysX ignores
            # deformable materials on collision/graphics meshes (June bug #1:
            # we bound /col and ran on the DEFAULT material with no contact)
            physicsUtils.add_physics_material_to_prim(
                stage, stage.GetPrimAtPath("/World/Pad/sim"), matp)
            physicsUtils.add_physics_material_to_prim(
                stage, stage.GetPrimAtPath("/World/Pad/col"), matp)
            # explicit offsets on the collision tetmesh: defaults floated the
            # pad ~20mm above the ground (run 3) — an offset cushion thicker
            # than the washer itself, which then slipped inside it
            from pxr import PhysxSchema
            colp = stage.GetPrimAtPath("/World/Pad/col")
            pca = PhysxSchema.PhysxCollisionAPI.Apply(colp)
            pca.CreateContactOffsetAttr().Set(0.002)
            pca.CreateRestOffsetAttr().Set(0.0)

            spec = parts.PartSpec(kind="washer", size="m12", pose="flat",
                                  xy=(0.0, 0.0), rotz_deg=0.0)
            # drop from 5mm up: spawning flush with the pad top ejected the
            # washer (+123mm) in the first run — contact-offset overlap
            wpath = parts.spawn_part(stage, "/World/Washer", spec,
                                     SURF_Z + 0.005)
            # control: 10mm rigid cube beside the washer — if the cube rests
            # but the washer tunnels, the failure is thin-part-specific
            if stage.GetPrimAtPath("/World/Cube").IsValid():
                stage.RemovePrim("/World/Cube")
            cb = UsdGeom.Cube.Define(stage, "/World/Cube")
            cb.GetSizeAttr().Set(0.010)
            UsdGeom.XformCommonAPI(cb.GetPrim()).SetTranslate(
                Gf.Vec3d(0.020, 0.0, SURF_Z + 0.010))
            UsdPhysics.CollisionAPI.Apply(cb.GetPrim())
            cca = PhysxSchema.PhysxCollisionAPI.Apply(cb.GetPrim())
            cca.CreateContactOffsetAttr().Set(0.004)
            cca.CreateRestOffsetAttr().Set(0.0)
            UsdPhysics.RigidBodyAPI.Apply(cb.GetPrim())
            UsdPhysics.MassAPI.Apply(cb.GetPrim()).CreateMassAttr().Set(0.02)

            world.reset()
            zs, pad_tops, cz = [], [], []
            simmesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Pad/sim"))
            for step in range(int(240 * 10.0)):
                world.step(render=False)
                if step % 480 == 0:
                    xc = UsdGeom.XformCache()
                    tr = xc.GetLocalToWorldTransform(
                        stage.GetPrimAtPath(wpath)).ExtractTranslation()
                    zs.append(float(tr[2]))
                    pts = simmesh.GetPointsAttr().Get()
                    pad_tops.append(max(p[2] for p in pts) if pts else -1.0)
                    ct = xc.GetLocalToWorldTransform(
                        cb.GetPrim()).ExtractTranslation()
                    cz.append(float(ct[2]))
            wz = zs[-1]
            pen = (SURF_Z - (wz - W_T / 2)) * 1000.0
            fell = wz < SURF_Z - PAD[2] / 2
            r = dict(resolution=res, young_mpa=young / 1e6,
                     wz_mm=round((wz - SURF_Z) * 1000, 2),
                     penetration_mm=round(pen, 2),
                     fell_through=bool(fell),
                     trace_mm=[round((z - SURF_Z) * 1000, 2) for z in zs],
                     pad_top_mm=[round((z - SURF_Z) * 1000, 2)
                                 for z in pad_tops],
                     cube_mm=[round((z - SURF_Z) * 1000, 2) for z in cz])
            results.append(r)
            print(f"[poc1] res={res:2d} E={young/1e6:.1f}MPa "
                  f"pen={r['penetration_mm']:6.2f}mm fell={int(fell)} "
                  f"trace={r['trace_mm']} padtop={r['pad_top_mm']} "
                  f"cube={r['cube_mm']}", flush=True)
            world.stop()

    json.dump(results, open(os.path.join(OUT, "poc1.json"), "w"), indent=1)
    ok = [r for r in results if not r["fell_through"]
          and r["penetration_mm"] <= 0.5]
    print(f"[poc1] PASS cells (pen<=0.5mm, no fall-through): "
          f"{len(ok)}/{len(results)}", flush=True)
    print(f"[poc1] DONE -> {OUT}", flush=True)
    app.close()


if __name__ == "__main__":
    main()
