#!/usr/bin/env python3
"""
resid_policy — Level 3: a CLOSED-LOOP residual policy for the washer pick.

A small numpy MLP reads real-sensor-equivalent observations every 20 ms
(B-spring deflection ≙ gripper motor current, command state, phase timing)
and outputs bounded RESIDUAL corrections around the env-tuned maneuver θ*:
    a0 → Δz on the finger height command      (± 3 mm)
    a1 → Δgap on the close/carry target       (± 1.5 mm)
    a2 → close-rate scale on the close power  (± 30 %)
The net's last layer starts at zero, so training starts exactly at θ*'s
behavior and learns corrections — a residual policy.

Trained with OpenAI-ES (antithetic pairs + rank shaping + Adam) under WIDE
domain randomization: placement ±1.5 mm, per-rig lattice roll ±20°, and —
the point of going closed-loop — MATERIAL noise (foundation stiffness /
damping / coupling and B-spring stiffness rescaled every episode via
SpringFoundation.retune) plus sensor noise on the deflection signal.
Antithetic pair members share the same rig, jitter, and material draw
(common random numbers), which cuts ES gradient variance enormously.

Runs in the DEPLOY env (GraspSortEnv — train where you deploy). Resumable:
ES state is saved atomically every update, so an outer loop can restart the
process to sidestep the ~96-rig-round-per-process memory leak.

Env (GS_RP2_*):
  MODE train|eval|baseline   ROUNDS (train: max rounds this process)
  THETA (θ*, 9 csv)  MAT (json)  STATE (state npz)  WEIGHTS (eval override)
  RIGS 6  COLS 3  POP 24  SIG 0.05  LR 0.03  SEED 0
  JIT_XY 0.0015  JIT_DZ 0.0005  ROLL 20  NOISE 0.00005
  DRK 0.70,1.35  DRC 0.80,1.25  DRCP 0.70,1.30  DRB 0.80,1.20
  OUT (dir)   SPAN (0.05)
"""
import datetime
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

MODE = os.environ.get("GS_RP2_MODE", "train")
RIGS = int(os.environ.get("GS_RP2_RIGS", "6"))
COLS = int(os.environ.get("GS_RP2_COLS", "3"))
SPACING = 0.15
ROUNDS = int(os.environ.get("GS_RP2_ROUNDS", "13" if MODE == "train" else "4"))
POP = int(os.environ.get("GS_RP2_POP", "24"))          # must be 2*RIGS*k
SIG = float(os.environ.get("GS_RP2_SIG", "0.05"))
LR = float(os.environ.get("GS_RP2_LR", "0.03"))
SEED = int(os.environ.get("GS_RP2_SEED", "0"))
JIT_XY = float(os.environ.get("GS_RP2_JIT_XY", "0.0015"))
JIT_DZ = float(os.environ.get("GS_RP2_JIT_DZ", "0.0005"))
ROLL = float(os.environ.get("GS_RP2_ROLL", "20"))
NOISE = float(os.environ.get("GS_RP2_NOISE", "0.00005"))
DRK = [float(v) for v in os.environ.get("GS_RP2_DRK", "0.70,1.35").split(",")]
DRC = [float(v) for v in os.environ.get("GS_RP2_DRC", "0.80,1.25").split(",")]
DRCP = [float(v) for v in os.environ.get("GS_RP2_DRCP", "0.70,1.30").split(",")]
DRB = [float(v) for v in os.environ.get("GS_RP2_DRB", "0.80,1.20").split(",")]
SNAP_PEN = float(os.environ.get("GS_RP2_SNAP_PEN", "4.0"))
os.environ["GS_PL_SPAN"] = os.environ.get("GS_RP2_SPAN", "0.05")
_now = datetime.datetime.now()
OUT = os.environ.get("GS_RP2_OUT", os.path.join(
    "data", _now.strftime("%Y-%m-%d"), _now.strftime("%H%M") + "-resid"))
STATE = os.environ.get("GS_RP2_STATE", os.path.join(OUT, "es_state.npz"))

DEF_MAT = ('{"name":"PARKER_d24","stiffness":300,"damping":24,'
           '"couple":3000,"couple_damp":8,"falloff":0,"cutoff":0.1,'
           '"leveling":true,"btip":"square","mode":"parallel"}')

OBS_DIM, H, ACT_DIM = 13, 32, 3
OBS_DEC = 12                       # policy tick every 12 physics steps (20 Hz)
A_DZ, A_DGAP, A_RATE = 0.003, 0.0015, 0.30


# ── tiny numpy MLP ───────────────────────────────────────────────────────────
SHAPES = [(OBS_DIM, H), (H,), (H, H), (H,), (H, ACT_DIM), (ACT_DIM,)]
N_PARAMS = sum(int(np.prod(s)) for s in SHAPES)


def unpack(w):
    out, i = [], 0
    for s in SHAPES:
        n = int(np.prod(s))
        out.append(w[i:i + n].reshape(s))
        i += n
    return out


def init_w(rng):
    W1 = rng.normal(0, 1.0 / math.sqrt(OBS_DIM), (OBS_DIM, H))
    W2 = rng.normal(0, 1.0 / math.sqrt(H), (H, H))
    W3 = np.zeros((H, ACT_DIM))            # zero last layer -> starts at θ*
    return np.concatenate([W1.ravel(), np.zeros(H), W2.ravel(), np.zeros(H),
                           W3.ravel(), np.zeros(ACT_DIM)]).astype(np.float64)


def policy_act(w, obs):
    W1, b1, W2, b2, W3, b3 = unpack(w)
    h1 = np.tanh(obs @ W1 + b1)
    h2 = np.tanh(h1 @ W2 + b2)
    return np.tanh(h2 @ W3 + b3)


# ── ES state ─────────────────────────────────────────────────────────────────
def load_state(rng):
    if os.path.exists(STATE):
        d = np.load(STATE)
        return (d["w"], d["m"], d["v"], int(d["upd"]),
                [float(x) for x in d["fit_hist"]])
    # cross-JOB resume: no local state, but weights injected (e.g. recovered
    # from a dead job's WSAVE log line via --inject-file). Adam restarts.
    iw = os.environ.get("GS_RP2_INIT_W", "")
    if iw and os.path.exists(iw):
        d = np.load(iw)
        w = d["w"].astype(np.float64)
        upd0 = int(d["upd"]) if "upd" in d else 0
        print(f"[rp2] init weights from {iw} (upd {upd0})", flush=True)
        return w, np.zeros_like(w), np.zeros_like(w), upd0, []
    w = init_w(rng)
    return w, np.zeros_like(w), np.zeros_like(w), 0, []


def save_state(w, m, v, upd, fit_hist):
    os.makedirs(os.path.dirname(STATE) or ".", exist_ok=True)
    tmp = STATE + ".tmp.npz"
    np.savez(tmp, w=w, m=m, v=v, upd=upd, fit_hist=np.array(fit_hist))
    os.replace(tmp, STATE)


def _dseed(*parts):
    """Deterministic cross-process seed (python's hash() is salted per run)."""
    import zlib
    return zlib.crc32(repr(parts).encode()) & 0x7FFFFFFF


def dr_draw(seed_tuple):
    """Deterministic domain draw: jitter + material scales for one episode."""
    r = np.random.default_rng(_dseed(*seed_tuple))
    jit = (r.uniform(-JIT_XY, JIT_XY), r.uniform(-JIT_XY, JIT_XY),
           r.uniform(-JIT_DZ, JIT_DZ))
    dr = dict(k=r.uniform(*DRK), c=r.uniform(*DRC),
              cp=r.uniform(*DRCP), bk=r.uniform(*DRB),
              nseed=int(r.integers(2 ** 31)))
    return jit, dr


def main():
    os.environ.setdefault("GS_SOFT", "0")
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    from graspsort.sim_env import GraspSortEnv
    env = GraspSortEnv(headless=True, width=320, height=240)
    env.world.get_physics_context().set_physics_dt(1.0 / 240.0)
    env.reset_world()
    stage = env.stage

    import tools.pick_lab as PL
    from pxr import PhysxSchema, UsdPhysics, UsdGeom
    wc = env._work_centre_from_stage()
    PL.SURF_Z = env.table_top_z + 0.030

    theta = np.array([float(v) for v in os.environ["GS_RP2_THETA"].split(",")])
    mat = json.loads(os.environ.get("GS_RP2_MAT", DEF_MAT))
    rows = (RIGS + COLS - 1) // COLS
    rolls = ([0.0] if RIGS == 1 else
             [round(float(v), 3) for v in np.linspace(-ROLL, ROLL, RIGS)])

    # ── PolicyRig: PickRig + sensors + residual actions ─────────────────────
    class PolicyRig(PL.PickRig):
        def set_policy(self, w):
            self._w = w

        def set_dr(self, dr):
            """Per-episode material randomization + sensor-noise stream."""
            self.found.retune(dr["k"], dr["c"], dr["cp"])
            drv = UsdPhysics.DriveAPI(
                self.stage.GetPrimAtPath(self.base + "/BSpring"), "linear")
            drv.GetStiffnessAttr().Set(8000.0 * dr["bk"])
            self._nrng = np.random.default_rng(dr["nseed"])

        def reset(self, th, jitter=(0.0, 0.0, 0.0)):
            super().reset(th, jitter=jitter)
            self._tick = 0
            self._act = np.zeros(ACT_DIM)
            self._dhist = [0.0, 0.0, 0.0]
            self._zcmd = self._bxcmd = None

        def _obs_and_act(self, t, a_c, bx0):
            xb = float(UsdGeom.XformCache().GetLocalToWorldTransform(
                self.stage.GetPrimAtPath(self.base + "/FingerB"))
                .ExtractTranslation()[0])
            defl = xb - (self._bxcmd if self._bxcmd is not None else bx0)
            defl += float(self._nrng.normal(0.0, NOISE)) if NOISE else 0.0
            d = float(np.clip(defl / 0.002, -2.0, 2.0))
            t2 = PL.T_SET + PL.T_DSC
            t_total = PL.T_TOTAL_P + PL.T_CARRY
            obs = np.array([
                t / t_total,
                1.0 if (self._ph < 2 and t < t2) else 0.0,
                1.0 if (self._ph < 2 and t >= t2) else 0.0,
                1.0 if self._ph >= 2 else 0.0,
                ((t - self._t3) / PL.T_LC) if self._t3 is not None else 0.0,
                d, self._dhist[0], self._dhist[1], self._dhist[2],
                ((self._zcmd - PL.SURF_Z) / 0.05
                 if self._zcmd is not None else 1.0),
                getattr(self, "_vc", 0.0),
                ((self._bxcmd - a_c) / 0.03
                 if self._bxcmd is not None else 1.0),
                1.0 if self._triggered else 0.0,
            ])
            self._dhist = [d, self._dhist[0], self._dhist[1]]
            self._act = policy_act(self._w, obs)

        def _drive_parallel(self, t):
            # pick_lab's maneuver verbatim, with policy residuals injected at
            # the command level (Δz, Δgap, close-rate) — the same interface a
            # real controller would expose.
            cx = self.c[0]
            a_over, g0, pd, close_pow, gap, rise_h, brace = self.theta[:7]
            gap_carry = self.theta[7] if len(self.theta) > 7 else gap
            pitch_deg = self.theta[8] if len(self.theta) > 8 else 0.0
            a_c = cx - PL.W_R - (a_over - 0.5) * PL.FW
            dzj = self._jit[2] if hasattr(self, "_jit") else 0.0
            hi = PL.SURF_Z + 0.06 + PL.FH / 2 + dzj
            zp = PL.SURF_Z - pd + PL.FH / 2 + dzj
            bx0 = a_c + g0
            t1, t2 = PL.T_SET, PL.T_SET + PL.T_DSC
            t3max = t2 + PL.T_WAIT

            if self._tick % OBS_DEC == 0:
                self._obs_and_act(t, a_c, bx0)
            self._tick += 1
            dz_r = A_DZ * float(self._act[0])
            dgap_r = A_DGAP * float(self._act[1])
            cp_eff = max(0.3, close_pow * (1.0 + A_RATE * float(self._act[2])))
            bx_close = a_c + PL.FW + gap + dgap_r

            if self._ph < 2:
                if t < t1:
                    z, bx = hi, bx0
                elif t < t2:
                    u = PL._ease((t - t1) / PL.T_DSC)
                    z, bx = hi + (zp - hi) * u + dz_r * u, bx0
                else:
                    z, bx = zp + dz_r, bx0
                    xb = float(UsdGeom.XformCache().GetLocalToWorldTransform(
                        self.stage.GetPrimAtPath(self.base + "/FingerB"))
                        .ExtractTranslation()[0])
                    if abs(xb - bx0) > PL.TRIG_DX:
                        self._trig_ct += 1
                        if self._trig_ct >= PL.TRIG_HOLD:
                            self._triggered = True
                    else:
                        self._trig_ct = 0
                    if self._triggered or t >= t3max:
                        self._ph = 2
                        self._t3 = t
                        self._zpress = z
            if self._ph >= 2:
                dt3 = t - self._t3
                if dt3 < PL.T_BRACE:
                    v = PL._ease(dt3 / PL.T_BRACE)
                    z = self._zpress - brace * v + dz_r
                    bx = bx0
                elif dt3 < PL.T_LC:
                    v = PL._ease(min((dt3 - PL.T_BRACE)
                                     / (PL.T_LC - PL.T_BRACE), 1.0))
                    vc = v ** cp_eff
                    self._vc = vc
                    z = self._zpress - brace + rise_h * v + dz_r
                    bx = bx0 + (bx_close - bx0) * vc
                else:
                    w_ = PL._ease(min((dt3 - PL.T_LC) / PL.T_CARRY, 1.0))
                    z_top = self._zpress - brace + rise_h
                    z_car = PL.SURF_Z + PL.CARRY_H + PL.FH / 2
                    z = z_top + (z_car - z_top) * w_ + dz_r
                    bx = bx_close + (gap_carry - gap) * w_
                    self._pitch = pitch_deg * w_
                    self._vc = 1.0
            pch = getattr(self, "_pitch", 0.0)
            self._zcmd, self._bxcmd = z, bx
            self._set_pos(self.primA, a_c, z, pch)
            self._set_pos(self.primC, bx, z, pch)
            if self._leveling:
                self._lvl_ct += 1
                if self._lvl_ct % 4 == 1:
                    bodies = [(a_c, self.c[1], PL.FW / 2, PL.FW / 2,
                               z - PL.FH / 2),
                              (bx, self.c[1], PL.FW / 2, PL.FW / 2,
                               z - PL.FH / 2)]
                    wb = self._washer_footprint()
                    if wb:
                        bodies.append(wb)
                    self.found.level_targets(bodies, self._ell)

    def rig_c(r):
        i, j = divmod(r, COLS)
        return (wc[0] + (j - (COLS - 1) / 2) * SPACING,
                wc[1] + (i - (rows - 1) / 2) * SPACING)

    rigs = []
    for r in range(RIGS):
        rig = PolicyRig(stage, r, rig_c(r), mat, roll_deg=rolls[r])
        rig.base = f"/World/RPRig{r}"
        rig.build()
        for pth in (rig.base + "/FingerB", rig.wpath):
            prb = PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(pth))
            prb.CreateSolverPositionIterationCountAttr().Set(32)
            prb.CreateSolverVelocityIterationCountAttr().Set(8)
        rigs.append(rig)
    env.world.reset()

    t_total = PL.T_TOTAL_P + PL.T_CARRY + 0.2
    steps = int(t_total / PL.PHYS_DT)
    rounds_done = 0

    def run_round(ws, seeds, tag):
        """One synchronized round: rig r runs weights ws[r] under DR seeds[r]."""
        nonlocal rounds_done
        import time as _tm
        tw0 = _tm.time()
        for r, rig in enumerate(rigs):
            jit, dr = dr_draw(seeds[r])
            rig.set_policy(ws[r])
            rig.set_dr(dr)
            rig.reset(theta, jitter=jit)
        for s in range(steps):
            t = s * PL.PHYS_DT
            for rig in rigs:
                rig.drive(t)
            env.step(render=False)
            if t > PL.T_SET and s % 12 == 0:
                for rig in rigs:
                    rig.observe(t)
        rounds_done += 1
        scores = [rig.final_score() for rig in rigs]
        rews = [x["reward"] for x in scores]
        print(f"[rp2] {tag}: succ={sum(1 for x in scores if x['success'])}"
              f"/{RIGS} snaps={sum(1 for x in scores if x.get('snap'))} "
              f"mean={np.mean(rews):.2f} ({_tm.time()-tw0:.0f}s)", flush=True)
        return scores

    print(f"[rp2] mode={MODE} params={N_PARAMS} pop={POP} sig={SIG} lr={LR} "
          f"rigs={RIGS} rolls={rolls} DRK={DRK} DRB={DRB}", flush=True)

    if MODE == "video":
        # render ONE deterministic-panel round of the policy: all-rigs camera,
        # grid skins (rotated lattices visible), mp4 + phase stills.
        from graspsort.soft_foundation import SurfaceSkin
        from pxr import UsdShade, Gf, Sdf
        from render_review import look_at_quat
        from isaacsim.sensors.camera import Camera
        import cv2
        wf = os.environ.get("GS_RP2_WEIGHTS", STATE)
        w = np.load(wf)["w"]
        print(f"[rp2] video weights {wf} |w|={np.linalg.norm(w):.2f}", flush=True)

        def solid(name, rgb, rough=0.5, emissive=None):
            p = f"/World/RPLooks/{name}"
            mm = UsdShade.Material.Define(stage, p)
            ss = UsdShade.Shader.Define(stage, p + "/Shader")
            ss.CreateIdAttr("UsdPreviewSurface")
            ss.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*rgb))
            ss.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
            if emissive:
                ss.CreateInput("emissiveColor",
                               Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
            mm.CreateSurfaceOutput().ConnectToSource(
                ss.ConnectableAPI(), "surface")
            return mm
        from pxr import UsdGeom as _UG
        _UG.Scope.Define(stage, "/World/RPLooks")
        red = solid("WasherRed", (0.95, 0.10, 0.06), emissive=(0.35, 0.02, 0.01))
        fa = solid("FingA", (0.16, 0.18, 0.22), 0.4)
        fb = solid("FingB", (0.15, 0.35, 0.90), 0.4)
        skins = []
        for r, rig in enumerate(rigs):
            sk = SurfaceSkin(rig.found, path=f"/World/RPSkin{r}").build()
            skins.append(sk)
            UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(
                rig.wpath)).Bind(red, UsdShade.Tokens.strongerThanDescendants)
            UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(
                rig.base + "/FingerA/geo")).Bind(fa)
            UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(
                rig.base + "/FingerB/geo")).Bind(fb)
        env.world.reset()
        gw, gh = (COLS - 1) * SPACING, (rows - 1) * SPACING
        diag = math.hypot(gw + 0.14, gh + 0.14)
        az = math.radians(215.0)
        dist = max(0.55, 1.35 * diag + 0.30)
        tgt = (wc[0], wc[1], PL.SURF_Z + 0.02)
        eye = (wc[0] + dist * math.cos(az), wc[1] + dist * math.sin(az),
               PL.SURF_Z + 0.58 * dist)
        cam = Camera(prim_path="/World/RPCam", position=np.array(eye),
                     frequency=30, resolution=(1920, 1080),
                     orientation=look_at_quat(eye, tgt))
        cam.initialize()
        for _ in range(20):
            env.step(render=True)
        for r, rig in enumerate(rigs):
            jit, dr = dr_draw(("panel", SEED, 0, r))
            rig.set_policy(w)
            rig.set_dr(dr)
            rig.reset(theta, jitter=jit)
        frames = []
        for s in range(steps):
            t = s * PL.PHYS_DT
            for rig in rigs:
                rig.drive(t)
            env.step(render=(s % 8 == 0))
            if t > PL.T_SET and s % 12 == 0:
                for rig in rigs:
                    rig.observe(t)
            if s % 8 == 0:
                for sk in skins:
                    sk.update()
                rgba = cam.get_rgba()
                if rgba is not None and getattr(rgba, "size", 0) > 0:
                    fr = np.asarray(rgba)[:, :, :3].astype(np.uint8).copy()
                    cv2.putText(fr, f"closed-loop residual policy  "
                                f"{RIGS} rigs  t={t:4.1f}s", (14, 34),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (245, 245, 245), 2, cv2.LINE_AA)
                    frames.append(fr)
        scores = [rig.final_score() for rig in rigs]
        print(f"[rp2] VIDEO succ="
              f"{sum(1 for x in scores if x['success'])}/{RIGS} "
              f"snaps={sum(1 for x in scores if x.get('snap'))}", flush=True)
        if frames:
            h_, w_ = frames[0].shape[:2]
            vw = cv2.VideoWriter(os.path.join(OUT, "policy.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 30, (w_, h_))
            for fr in frames:
                vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            vw.release()
            try:
                from graspsort.videoio import to_h264
                to_h264(os.path.join(OUT, "policy.mp4"))
            except Exception as e:
                print(f"[rp2] h264: {e}", flush=True)
            cv2.imwrite(os.path.join(OUT, "policy_final.png"),
                        cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR))
            print(f"[rp2] policy.mp4 ({len(frames)} frames) -> {OUT}",
                  flush=True)
        env.close()
        return

    if MODE in ("eval", "baseline"):
        if MODE == "eval":
            wf = os.environ.get("GS_RP2_WEIGHTS", STATE)
            w = np.load(wf)["w"]
            print(f"[rp2] eval weights from {wf} |w|={np.linalg.norm(w):.2f}",
                  flush=True)
        else:
            w = init_w(np.random.default_rng(0))  # zero residual = pure θ*
        tally = []
        for rnd in range(ROUNDS):
            seeds = [("panel", SEED, rnd, r) for r in range(RIGS)]
            tally += run_round([w] * RIGS, seeds, f"{MODE} rnd{rnd}")
        succ = sum(1 for sc in tally if sc["success"])
        snaps = sum(1 for sc in tally if sc.get("snap"))
        rews = [sc["reward"] for sc in tally]
        print(f"[rp2] ===== {MODE.upper()} {succ}/{len(tally)} "
              f"snaps={snaps} mean={np.mean(rews):.2f} "
              f"worst={np.min(rews):.2f} =====", flush=True)
        json.dump(dict(mode=MODE, success=succ, n=len(tally), snaps=snaps,
                       mean=float(np.mean(rews)), worst=float(np.min(rews))),
                  open(os.path.join(OUT, f"{MODE}_summary.json"), "w"), indent=1)
        env.close()
        return

    # ── train: OpenAI-ES, antithetic + common random numbers ────────────────
    w, m, v, upd, fit_hist = load_state(rng)
    print(f"[rp2] resume at update {upd}, |w|={np.linalg.norm(w):.3f}",
          flush=True)
    pairs = POP // 2
    assert pairs % RIGS == 0, "POP must be 2*RIGS*k"
    b1, b2, eps_adam = 0.9, 0.999, 1e-8
    while True:
        # (pairs*2 evals)/RIGS rounds + the center round after this update?
        need = (pairs * 2) // RIGS + (1 if (upd + 1) % 5 == 0 else 0)
        if rounds_done + need > ROUNDS:
            print(f"[rp2] round cap {ROUNDS} reached at update {upd} — "
                  f"clean exit for restart", flush=True)
            break
        erng = np.random.default_rng(_dseed("eps", SEED, upd))
        eps = erng.normal(0.0, 1.0, (pairs, N_PARAMS))
        fits = np.zeros(POP)
        # schedule: pair block p0..p{RIGS-1} sign + then -, next block...
        for blk in range(pairs // RIGS):
            base = blk * RIGS
            seeds = [("dr", SEED, upd, base + r) for r in range(RIGS)]
            for sgn, off in ((+1, 0), (-1, 1)):
                ws = [w + sgn * SIG * eps[base + r] for r in range(RIGS)]
                scores = run_round(ws, seeds, f"upd{upd} blk{blk}"
                                   f"{'+' if sgn > 0 else '-'}")
                for r, sc in enumerate(scores):
                    # hard anti-flip fitness: ES specification-gamed the smooth
                    # reward with violent rolls (snaps grew 2->13 over training)
                    fits[(base + r) * 2 + off] = (sc["reward"]
                                                  - SNAP_PEN * bool(sc.get("snap")))
        # rank-shaped gradient
        ranks = np.empty(POP)
        ranks[np.argsort(fits)] = np.arange(POP)
        shaped = ranks / (POP - 1) - 0.5
        g = np.zeros(N_PARAMS)
        for p in range(pairs):
            g += shaped[p * 2] * eps[p] - shaped[p * 2 + 1] * eps[p]
        g /= (POP * SIG)
        g -= 0.005 * w                       # weight decay
        upd += 1
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** upd)
        vh = v / (1 - b2 ** upd)
        w = w + LR * mh / (np.sqrt(vh) + eps_adam)
        fit_hist.append(float(np.mean(fits)))
        succ_n = int(np.sum(fits > 3.0))
        print(f"[rp2] UPDATE {upd}: fit mean={np.mean(fits):.3f} "
              f"max={np.max(fits):.2f} min={np.min(fits):.2f} "
              f"succ~{succ_n}/{POP} |w|={np.linalg.norm(w):.3f}", flush=True)
        save_state(w, m, v, upd, fit_hist)
        # weights → logs (base64 float32): the job's /tmp dies with the job,
        # so the log line IS the durable model artifact (~8.5 KB).
        import base64
        print(f"[rp2] WSAVE upd{upd} "
              + base64.b64encode(w.astype(np.float32).tobytes()).decode(),
              flush=True)
        if upd % 5 == 0:
            seeds = [("panel", 9999, upd // 5 % 4, r) for r in range(RIGS)]
            scs = run_round([w] * RIGS, seeds, f"CENTER upd{upd}")
            cs = sum(1 for x in scs if x["success"])
            print(f"[rp2] CENTER upd{upd}: {cs}/{RIGS}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
