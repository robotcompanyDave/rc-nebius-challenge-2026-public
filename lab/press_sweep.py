#!/usr/bin/env python3
"""
Gate M (Isaac Lab): material sweep for the 5 mm MEET condition.

Hundreds of envs, each a full rig (anchored SpringFoundation + m12 washer +
bound two-finger gripper), each env with DIFFERENT material parameters.
The canonical press runs open-loop in all envs at once; we measure the
far-rim TOP height at finger B's station during the hold and score
  err = |rim_height − 5 mm|,  stability = std(rim_height) over the hold.

Modes:
  --mode parity  N identical SOFT220 envs; compare washer sink/tilt at end of
                 press vs the pick_lab CPU numbers (wz ≈ −3.3 mm, tilt ≈ 10.5°)
                 → quantifies the Lab-GPU vs lab-CPU physics delta.
  --mode sweep   grid over (k_cell × ratio × damping) → results.json,
                 Gate-M finalists = err ≤ 1 mm and stable.

    docker/run_lab.sh "cd /workspace/isaaclab && ./isaaclab.sh -p \\
        /workspace/push-grasp/lab/press_sweep.py --mode sweep --num_envs 256 \\
        --out /workspace/push-grasp/data/<date>/lab_sweep"
"""
import argparse
import json
import math
import os
import sys

# Isaac Lab 3.0 pattern: import torch/numpy BEFORE the app boots — the good
# torch gets cached in sys.modules; post-boot imports can hit a broken
# prebundled torch shadowed in by a deprecated extension (nccl symbol error).
import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="parity", choices=["parity", "sweep", "gate", "gate2", "dwell", "beside"])
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--out", default="/workspace/push-grasp/data/lab_out")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# 3.0: headless is the default (the old --headless flag is deprecated; GUI
# only with --viz). Just launch.
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── post-app imports ─────────────────────────────────────────────────────────
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, PhysxSchema
import omni.usd

sys.path.insert(0, "/workspace/push-grasp")
from graspsort.soft_foundation import SpringFoundation  # noqa: E402
from graspsort import parts  # noqa: E402

# ── canonical press (matches grip_smoke3 P1: the bound-gripper descend) ──────
SURF_Z = 0.50
SPAN, CELL = 0.050, 0.005
W_R, W_T = 0.012, 0.0025
FW, FH = 0.008, 0.030
A_OVER, G0, PRESS = 0.50, 0.028, 0.0028
T_SET, T_DSC, T_HOLD = 0.5, 1.2, 1.0
PHYS_DT = 1.0 / 240.0
CTRL_DEC = 4
TARGET_MM = 5.0


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def build_env(stage, e, origin, mat):
    """Author one full rig at /World/envs/env_{e}, positions offset by origin.
    (No cloning: Sdf.CopySpec keeps ABSOLUTE joint-rel targets, so every
    clone's springs anchored to env_0 — building each env directly is simple,
    bulletproof, and lets per-env materials bake in at author time.)"""
    gx, gy = origin
    base = f"/World/envs/env_{e:03d}"
    UsdGeom.Xform.Define(stage, base)
    anchor = base + "/Anchor"
    ax = UsdGeom.Xform.Define(stage, anchor)
    UsdGeom.XformCommonAPI(ax.GetPrim()).SetTranslate(Gf.Vec3d(gx, gy, 0))
    rb = UsdPhysics.RigidBodyAPI.Apply(ax.GetPrim())
    rb.CreateKinematicEnabledAttr(True)
    # PhysX can drop a SHAPELESS kinematic body (and a dropped body0 degrades
    # every joint that references it) — give the anchor mass and a small
    # collider buried 50mm below the tiles where nothing ever touches it.
    UsdPhysics.MassAPI.Apply(ax.GetPrim()).CreateMassAttr(1.0)
    ageo = UsdGeom.Cube.Define(stage, anchor + "/geo")
    ageo.GetSizeAttr().Set(1.0)
    UsdGeom.XformCommonAPI(ageo.GetPrim()).SetTranslate(
        Gf.Vec3d(0, 0, SURF_Z - 0.08))
    UsdGeom.XformCommonAPI(ageo.GetPrim()).SetScale(
        Gf.Vec3f(0.01, 0.01, 0.01))
    UsdPhysics.CollisionAPI.Apply(ageo.GetPrim())

    SpringFoundation(
        stage, (gx, gy), SURF_Z, span=SPAN, cell=CELL,
        stiffness=mat["stiffness"], damping=mat["damping"],
        couple=mat["couple"], couple_damp=mat["couple_damp"],
        anchor=anchor, visible=False, parent=base + "/Foundation").build()

    spec = parts.PartSpec(kind="washer", size="m12", pose="flat",
                          xy=(gx, gy), rotz_deg=0.0)
    parts.spawn_part(stage, base + "/Washer", spec, SURF_Z)

    def finger(name, x, fric):
        p = f"{base}/{name}"
        xf = UsdGeom.Xform.Define(stage, p)
        UsdGeom.XformCommonAPI(xf.GetPrim()).SetTranslate(
            Gf.Vec3d(x, gy, SURF_Z + 0.06 + FH / 2))
        r = UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
        r.CreateKinematicEnabledAttr(True)
        geo = UsdGeom.Cube.Define(stage, p + "/geo")
        geo.GetSizeAttr().Set(1.0)
        UsdGeom.XformCommonAPI(geo.GetPrim()).SetScale(Gf.Vec3f(FW, FW, FH))
        UsdPhysics.CollisionAPI.Apply(geo.GetPrim())
        pc = PhysxSchema.PhysxCollisionAPI.Apply(geo.GetPrim())
        pc.CreateContactOffsetAttr().Set(0.0015)
        pc.CreateRestOffsetAttr().Set(0.0)
        mp = p + "/Mat"
        UsdShadeMat(stage, mp, fric)
        from pxr import UsdShade
        UsdShade.MaterialBindingAPI.Apply(geo.GetPrim()).Bind(
            UsdShade.Material(stage.GetPrimAtPath(mp)),
            UsdShade.Tokens.weakerThanDescendants, "physics")

    a_over = mat.get("a_over", A_OVER)
    a_c = gx - W_R - (a_over - 0.5) * FW
    finger("FingerA", a_c, 0.10)
    finger("FingerB", a_c + G0, 1.20)
    return a_c - gx


def UsdShadeMat(stage, path, fric):
    from pxr import UsdShade
    UsdShade.Material.Define(stage, path)
    m = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(path))
    m.CreateStaticFrictionAttr(fric)
    m.CreateDynamicFrictionAttr(fric)
    m.CreateRestitutionAttr(0.0)


def system_grid():
    """Material x PRESS grid — the 5mm meet is a SYSTEM property (sweep 1:
    zero candidates with the fixed 2.8mm/0.5-overhang press; 5mm steady rim
    needs ~19 deg of held tilt, i.e. real press leverage)."""
    ks = [160, 220, 300, 420]
    ratios = [3, 6, 10]
    presses = [0.003, 0.004, 0.005]
    overs = [0.50, 0.60, 0.70]
    damps = [8.0, 12.0]
    grid = []
    for k in ks:
        for r in ratios:
            for pd in presses:
                for ao in overs:
                    for d in damps:
                        grid.append(dict(
                            stiffness=float(k), couple=float(k * r),
                            damping=d, couple_damp=8.0, ratio=float(r),
                            press=pd, a_over=ao))
    return grid   # 4*3*3*3*2 = 216


def main():
    os.makedirs(args.out, exist_ok=True)
    N = args.num_envs
    base_mat = dict(stiffness=220.0, damping=7.0, couple=1760.0,
                    couple_damp=8.0, ratio=8.0)

    sim = SimulationContext(SimulationCfg(dt=PHYS_DT, device=args.device))
    stage = omni.usd.get_context().get_stage()

    # per-env materials, baked at build time; envs authored directly (see
    # build_env docstring for why not cloning)
    mats = [dict(base_mat, press=PRESS, a_over=A_OVER) for _ in range(N)]
    if args.mode in ("sweep", "dwell"):
        grid = system_grid()
        mats = [grid[i % len(grid)] for i in range(N)]
    elif args.mode == "gate2":
        # slot-averaged truth: top configs x 8 grid slots each (no jitter —
        # the SLOT is the perturbation; gate showed +-10mm slot sensitivity)
        cands = [
            dict(stiffness=300.0, couple=3000.0, damping=12.0, press=0.003, a_over=0.70),
            dict(stiffness=300.0, couple=3000.0, damping=24.0, press=0.003, a_over=0.70),
            dict(stiffness=420.0, couple=2520.0, damping=8.0,  press=0.004, a_over=0.50),
            dict(stiffness=420.0, couple=2520.0, damping=16.0, press=0.004, a_over=0.50),
            dict(stiffness=300.0, couple=3000.0, damping=20.0, press=0.004, a_over=0.60),
            # dwell-sweep runners-up (100ms transits in their sweep slot)
            dict(stiffness=220.0, couple=2200.0, damping=12.0, press=0.004, a_over=0.70),
            dict(stiffness=300.0, couple=900.0,  damping=8.0,  press=0.003, a_over=0.70),
        ]
        for c in cands:
            c.setdefault("couple_damp", 8.0)
            c.setdefault("ratio", c["couple"] / c["stiffness"])
        mats = [dict(cands[e // 8]) for e in range(N)]
    elif args.mode == "gate":
        # Gate-M winner under the 8-jitter panel (washer +-1mm, gripper +-0.5mm)
        win = dict(stiffness=300.0, couple=3000.0, damping=12.0,
                   couple_damp=8.0, ratio=10.0, press=0.003, a_over=0.70)
        mats = [dict(win) for _ in range(N)]
    elif args.mode == "beside":
        # press-beside-part primitive (from the FEM POC lever observation:
        # deep press NEXT to the washer raised the far rim +8mm). Material
        # fixed = d24 parker; grid = press depth x A stand-off. a_over>0.5
        # moves A outward: 1.0 = A inner edge kisses the rim (all on pad),
        # 1.25 = 2mm gap. B does NOT descend — it hovers at catch height
        # beside the far rim (set_fingers beside branch).
        parker = dict(stiffness=300.0, couple=3000.0, damping=24.0,
                      couple_damp=8.0, ratio=10.0)
        bgrid = [dict(parker, press=pd, a_over=ao)
                 for pd in (0.003, 0.005, 0.007)
                 for ao in (0.70, 1.00, 1.25)]
        mats = [dict(bgrid[e // 8]) for e in range(N)]
    PANEL = [(0.0, 0.0, 0.0),
             (0.001, 0.0, 0.0), (-0.001, 0.0, 0.0),
             (0.0, 0.001, 0.0), (0.0, -0.001, 0.0),
             (0.0, 0.0, 0.0005), (0.0, 0.0, -0.0005),
             (0.0007, 0.0007, -0.0005)]
    jit = [PANEL[e % len(PANEL)] if args.mode == "gate" else (0.0, 0.0, 0.0)
           for e in range(N)]
    side = int(math.ceil(math.sqrt(N)))
    UsdGeom.Xform.Define(stage, "/World/envs")
    a_c = 0.0
    for e in range(N):
        gx, gy = (e % side) * 0.30, (e // side) * 0.30
        a_c = build_env(stage, e, (gx, gy), mats[e])
    print(f"[build] {N} envs authored", flush=True)

    # Lab view for the WASHER only (batched pose reads + reset teleport).
    # Fingers are driven through USD like pick_lab: Lab-view teleports on
    # kinematic bodies produced NO contact (A sat 5mm inside the washer with
    # zero response); USD-driven kinematics are the proven pattern.
    washer = RigidObject(RigidObjectCfg(
        prim_path="/World/envs/env_.*/Washer", spawn=None))
    fa_ops, fb_ops = [], []
    from pxr import UsdGeom as _UG
    for e in range(N):
        fa_ops.append(_UG.XformCommonAPI(
            stage.GetPrimAtPath(f"/World/envs/env_{e:03d}/FingerA")))
        fb_ops.append(_UG.XformCommonAPI(
            stage.GetPrimAtPath(f"/World/envs/env_{e:03d}/FingerB")))

    sim.reset()
    washer.update(PHYS_DT)

    dev = torch.device(str(washer.data.root_pos_w.device))
    # Lab's RigidObject RESET writes cfg.init_state (default 0,0,0) into the
    # sim — wrapping existing prims teleports everything to the world origin!
    # Rebuild poses analytically from the grid and write them back.
    origins = torch.zeros((N, 3), device=dev)
    for e in range(N):
        origins[e, 0] = (e % side) * 0.30
        origins[e, 1] = (e // side) * 0.30
    identq = torch.zeros((N, 4), device=dev)
    identq[:, 0] = 1.0
    wpose = torch.zeros((N, 7), device=dev)
    wpose[:, 0:3] = origins
    for e in range(N):
        wpose[e, 0] += jit[e][0]
        wpose[e, 1] += jit[e][1]
    wpose[:, 2] = SURF_Z + 0.0024          # seat height; settles during T_SET
    wpose[:, 3:7] = identq
    washer.write_root_pose_to_sim(wpose)
    washer.write_root_velocity_to_sim(torch.zeros((N, 6), device=dev))

    hi_z = SURF_Z + 0.06 + FH / 2
    a_cs = [-W_R - (m.get("a_over", A_OVER) - 0.5) * FW for m in mats]
    press_zs = [SURF_Z - m.get("press", PRESS) + FH / 2 + jit[e][2]
                for e, m in enumerate(mats)]
    ident_q = identq

    def set_fingers(u_or_z, phase):
        """phase: 'hi' hold high; 'desc' u in [0,1] toward per-env press_z;
        'press' hold per-env press depth."""
        for e in range(N):
            gx = (e % side) * 0.30
            gy = (e // side) * 0.30
            if phase == "hi":
                z = hi_z
            elif phase == "desc":
                z = hi_z + (press_zs[e] - hi_z) * u_or_z
            else:
                z = press_zs[e]
            fa_ops[e].SetTranslate(Gf.Vec3d(gx + a_cs[e], gy, z))
            if args.mode == "beside":
                # B = catcher: hover 0.5mm above the surface with its inner
                # face 1mm beyond the far rim; only A presses
                fb_ops[e].SetTranslate(Gf.Vec3d(
                    gx + W_R + 0.001 + FW / 2, gy,
                    SURF_Z + 0.0005 + FH / 2))
            else:
                fb_ops[e].SetTranslate(Gf.Vec3d(gx + a_cs[e] + G0, gy, z))

    total = T_SET + T_DSC + T_HOLD
    steps = int(total / PHYS_DT)
    rim_samples = []
    dwell_traj = []
    rloc = torch.tensor([W_R - 0.001, 0.0, W_T / 2], device=dev)
    rloc_n = torch.tensor([-(W_R - 0.001), 0.0, W_T / 2], device=dev)

    for s in range(steps):
        t = s * PHYS_DT
        if s % CTRL_DEC == 0:
            if t < T_SET:
                set_fingers(0.0, "hi")
            elif t < T_SET + T_DSC:
                set_fingers(ease((t - T_SET) / T_DSC), "desc")
            else:
                set_fingers(0.0, "press")
        sim.step(render=False)
        if s == int((T_SET + T_DSC * 0.99) / PHYS_DT):   # end-of-press probe
            washer.update(PHYS_DT)
            print(f"[probe] t={t:.2f} "
                  f"washer_wz={1000*(float(washer.data.root_pos_w[0,2])-SURF_Z):+.2f}mm",
                  flush=True)
            if args.mode == "gate2":
                # cross-section dump: center-row tile tops + washer pose +
                # finger geometry per env (for the annotated section figures)
                n_t = int(round(SPAN / CELL))
                mid = n_t // 2
                xs_state = []
                for e in range(N):
                    gx = (e % side) * 0.30
                    gy = (e // side) * 0.30
                    row = []
                    xc = UsdGeom.XformCache()
                    for i in range(n_t):
                        tp = stage.GetPrimAtPath(
                            f"/World/envs/env_{e:03d}/Foundation/t_{i}_{mid}")
                        M = xc.GetLocalToWorldTransform(tp)
                        tr = M.ExtractTranslation()
                        row.append([1000 * (float(tr[0]) - gx),
                                    1000 * (float(tr[2]) + 0.005 - SURF_Z)])
                    wp = washer.data.root_pos_w[e]
                    wq = washer.data.root_quat_w[e]
                    xs_state.append(dict(
                        env=e, mat={k: float(v) for k, v in mats[e].items()},
                        tiles=row,
                        washer=[1000 * (float(wp[0]) - gx),
                                1000 * (float(wp[2]) - SURF_Z)],
                        quat=[float(v) for v in wq],
                        a_c_mm=1000 * a_cs[e],
                        press_z_mm=1000 * (press_zs[e] - FH / 2 - SURF_Z),
                        g0_mm=1000 * G0))
                json.dump(xs_state, open(os.path.join(
                    args.out, "sections.json"), "w"))
                print(f"[dump] sections.json ({N} envs)", flush=True)
        if args.mode in ("dwell", "gate2", "beside") and t > T_SET and s % 12 == 0:
            washer.update(PHYS_DT)
            pos = washer.data.root_pos_w
            q = washer.data.root_quat_w
            w_, x_, y_, z_ = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            def rotd(v):
                vx, vy, vz = v[0], v[1], v[2]
                cx = y_ * vz - z_ * vy
                cy = z_ * vx - x_ * vz
                cz = x_ * vy - y_ * vx
                rx = vx + 2.0 * (w_ * cx + (y_ * cz - z_ * cy))
                rz = vz + 2.0 * (w_ * cz + (x_ * cy - y_ * cx))
                return rx, rz
            pxd, pzd = rotd(rloc)
            nxd, nzd = rotd(rloc_n)
            farp = (pxd > nxd)
            rimz = torch.where(farp, pos[:, 2] + pzd, pos[:, 2] + nzd)
            dwell_traj.append((rimz - SURF_Z).cpu().numpy() * 1000.0)
        if t > T_SET + T_DSC and s % 12 == 0:      # hold-phase sampling, 20 Hz
            washer.update(PHYS_DT)
            pos = washer.data.root_pos_w
            q = washer.data.root_quat_w            # (w,x,y,z)
            w_, x_, y_, z_ = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            # world offset of both ±x local rim points; keep the one nearer B
            def rot(v):
                vx, vy, vz = v[0], v[1], v[2]
                # quaternion rotate (batched, v constant per env)
                t2 = 2.0
                cx = y_ * vz - z_ * vy
                cy = z_ * vx - x_ * vz
                cz = x_ * vy - y_ * vx
                rx = vx + t2 * (w_ * cx + (y_ * cz - z_ * cy))
                ry = vy + t2 * (w_ * cy + (z_ * cx - x_ * cz))
                rz = vz + t2 * (w_ * cz + (x_ * cy - y_ * cx))
                return rx, ry, rz
            px, _, pz = rot(rloc)
            nx, _, nz = rot(rloc_n)
            far_is_p = (pos[:, 0] + px - origins[:, 0]) > (pos[:, 0] + nx - origins[:, 0])
            rim_z = torch.where(far_is_p, pos[:, 2] + pz, pos[:, 2] + nz)
            rim_samples.append((rim_z - SURF_Z).cpu().numpy() * 1000.0)

    R = np.stack(rim_samples, axis=0)             # (S, N) rim heights, mm
    mean_r, std_r = R.mean(axis=0), R.std(axis=0)

    if args.mode == "dwell":
        D = np.stack(dwell_traj, axis=0)          # (S, N) rim mm, 20 Hz
        SAMPLE_MS = 12 * PHYS_DT * 1000.0
        in_band = (D >= 4.0) & (D <= 6.0)
        dwell_ms = in_band.sum(axis=0) * SAMPLE_MS
        max_rim = D.max(axis=0)
        escape = (np.abs(D) > 60.0).any(axis=0)
        recs = []
        for e in range(N):
            recs.append(dict(env=e, **{k: mats[e][k] for k in
                                       ("stiffness", "ratio", "damping",
                                        "press", "a_over")},
                             dwell_ms=round(float(dwell_ms[e]), 1),
                             max_rim=round(float(max_rim[e]), 2),
                             hold_rim=round(float(mean_r[e]), 2),
                             escape=bool(escape[e])))
        recs.sort(key=lambda r: (-r["dwell_ms"], abs(r["hold_rim"] - 5)))
        json.dump(dict(recs=recs, traj=D.tolist()),
                  open(os.path.join(args.out, "dwell.json"), "w"))
        print("[dwell] top 12 by band dwell (4-6mm):")
        for r in recs[:12]:
            print(f"  k={r['stiffness']:4.0f} ratio={r['ratio']:4.1f} "
                  f"d={r['damping']:4.1f} press={1000*r['press']:.0f} "
                  f"ao={r['a_over']:.2f} dwell={r['dwell_ms']:6.1f}ms "
                  f"maxrim={r['max_rim']:6.2f} hold={r['hold_rim']:6.2f} "
                  f"esc={int(r['escape'])}")
        n_ok = sum(1 for r in recs if r["dwell_ms"] >= 150 and not r["escape"])
        print(f"[dwell] configs with >=150ms band dwell, no escape: {n_ok}")
    elif args.mode == "beside":
        D = np.stack(dwell_traj, axis=0)          # (S, N) rim mm, 20 Hz
        SAMPLE_MS = 12 * PHYS_DT * 1000.0
        dwell_ms = ((D >= 4.0) & (D <= 6.0)).sum(axis=0) * SAMPLE_MS
        max_rim = D.max(axis=0)
        esc = (np.abs(D) > 60.0).any(axis=0)
        for ci in range(N // 8):
            m = mats[ci * 8]
            dw = dwell_ms[ci * 8:(ci + 1) * 8]
            mr = max_rim[ci * 8:(ci + 1) * 8]
            ne = int(esc[ci * 8:(ci + 1) * 8].sum())
            print(f"[beside] press={1000*m['press']:.0f}mm "
                  f"ao={m['a_over']:.2f} "
                  f"dwell(slots)={[round(float(v)) for v in dw]} "
                  f"maxrim={[round(float(v), 1) for v in mr]} esc={ne}",
                  flush=True)
        json.dump(dict(dwell_ms=dwell_ms.tolist(), max_rim=max_rim.tolist(),
                       traj=D.tolist()),
                  open(os.path.join(args.out, "beside.json"), "w"))
    elif args.mode == "gate2":
        D = np.stack(dwell_traj, axis=0)          # (S, N) rim mm, 20 Hz
        SAMPLE_MS = 12 * PHYS_DT * 1000.0
        dwell_ms = ((D >= 4.0) & (D <= 6.0)).sum(axis=0) * SAMPLE_MS
        max_rim = D.max(axis=0)
        for ci in range(N // 8):
            block = mean_r[ci * 8:(ci + 1) * 8]
            dw = dwell_ms[ci * 8:(ci + 1) * 8]
            mr = max_rim[ci * 8:(ci + 1) * 8]
            m = mats[ci * 8]
            print(f"[gate2] k={m['stiffness']:.0f} r={m['ratio']:.0f} "
                  f"d={m['damping']:.0f} p={1000*m['press']:.0f} ao={m['a_over']:.2f} "
                  f"-> rim mean={block.mean():.2f} std={block.std():.2f} "
                  f"min={block.min():.2f} max={block.max():.2f}", flush=True)
            print(f"[dwell2]   dwell_ms per slot={[round(float(v)) for v in dw]} "
                  f"maxrim={[round(float(v), 1) for v in mr]}", flush=True)
        json.dump(dict(rim_mm=mean_r.tolist(), dwell_ms=dwell_ms.tolist(),
                       max_rim=max_rim.tolist(), traj=D.tolist()),
                  open(os.path.join(args.out, "gate2.json"), "w"))
    elif args.mode == "gate":
        ok = np.abs(mean_r - TARGET_MM) <= 1.0
        print(f"[gate] rim per jitter-env (mm): "
              f"{[round(float(v),2) for v in mean_r]}")
        print(f"[gate] mean={mean_r.mean():.2f} worst_err="
              f"{np.max(np.abs(mean_r-TARGET_MM)):.2f} pass={int(ok.sum())}/{N}")
        json.dump(dict(rim_mm=mean_r.tolist(), std=std_r.tolist(),
                       jitters=[list(j) for j in jit]),
                  open(os.path.join(args.out, "gate.json"), "w"), indent=1)
    elif args.mode == "parity":
        washer.update(PHYS_DT)
        pos = washer.data.root_pos_w.cpu().numpy()
        wz = (pos[:, 2] - SURF_Z) * 1000
        print(f"[parity] washer wz mm: mean={wz.mean():.2f} (pick_lab CPU ref ≈ -3.3)")
        print(f"[parity] far-rim mm:   mean={mean_r.mean():.2f} std_env={mean_r.std():.2f}")
        json.dump(dict(wz_mm=wz.tolist(), rim_mm=mean_r.tolist()),
                  open(os.path.join(args.out, "parity.json"), "w"), indent=1)
    else:
        recs = []
        for e in range(N):
            recs.append(dict(env=e, **{k: mats[e][k] for k in
                                       ("stiffness", "ratio", "damping",
                                        "press", "a_over")},
                             rim_mm=round(float(mean_r[e]), 2),
                             rim_std=round(float(std_r[e]), 3),
                             err_mm=round(abs(float(mean_r[e]) - TARGET_MM), 2)))
        recs.sort(key=lambda r: r["err_mm"] + 2.0 * r["rim_std"])
        json.dump(recs, open(os.path.join(args.out, "sweep.json"), "w"), indent=1)
        print("[sweep] top 10 by |rim-5mm| + 2*std:")
        for r in recs[:10]:
            print(f"  k={r['stiffness']:4.0f} ratio={r['ratio']:4.1f} "
                  f"d={r['damping']:4.1f} press={1000*r['press']:.0f} "
                  f"ao={r['a_over']:.2f} rim={r['rim_mm']:6.2f}mm "
                  f"std={r['rim_std']:.3f} err={r['err_mm']:.2f}")
        gate = [r for r in recs if r["err_mm"] <= 1.0 and r["rim_std"] < 0.5]
        print(f"[sweep] GATE-M candidates (err<=1mm, stable): {len(gate)}")

    print("[done]")
    simulation_app.close()


if __name__ == "__main__":
    main()
