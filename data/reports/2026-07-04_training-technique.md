# Training technique — how the pick is learned, and how Isaac Lab would scale it

*Written before the smooth-ramp scale-up (2026-07-04 evening). Companion to
[2026-07-04_robust-pick.md](2026-07-04_robust-pick.md).*

## 1. What we train

Not a neural policy (yet). The maneuver is a **parameterized motion primitive**
with one event branch:

```
descend(bound gripper, A over rim by a_over, gap g0)      # open-loop
  → press to press_depth                                  # open-loop
  → WAIT until B's spring deflects (touch sensor)         # event-driven
  → brace (A presses `brace` deeper, B holds)             # open-loop
  → pull up rise_h while closing to gap_final,            # open-loop
    close progress = rise_progress ** close_pow
```

θ ∈ R⁷ = (a_over, g0, press_depth, close_pow, gap_final, rise_h, brace).
The only feedback in the loop is the touch trigger — everything after it is
choreography. This is deliberate: 7 interpretable numbers, ~100s of
evaluations to learn, zero NN infrastructure, and every parameter maps to a
robot instruction we can hand to the UR10e controller.

## 2. How it's evaluated and optimized

- **Multi-rig batching**: K independent rigs (foundation + washer + gripper)
  spaced 0.3 m apart in ONE physics scene, stepped together. One `world.step`
  advances K evaluations — population parallelism without vectorized envs.
- **CEM (cross-entropy method)**: population = K rigs/round; elites (top ~34%)
  refit a diagonal Gaussian; σ floored at 8% of the bounds to keep exploring.
- **Jitter during search**: every eval gets placement ±0.7 mm / height ±0.3 mm
  noise, so the optimizer can't camp on knife-edge solutions.
- **Gauntlet selection**: top candidates face a fixed 8-jitter panel (one
  jitter per rig, same θ on all rigs); winner by **worst-2-mean**. Measured
  caveat: one panel is one SAMPLE (same θ + same panel re-run flipped the
  winner) → panels must be repeated 2–3× and averaged.
- **Determinism note**: PhysX in one scene is deterministic given identical
  boots, but scene composition and boot state shift outcomes — the same
  brittleness that shows up as the lab→env transfer gap. Treat every score as
  stochastic; select on distributions, not points.

## 3. Reward (smooth-ramp era, this phase)

Ordering dictated by cell reality: **a violent maneuver is worse than a
failed one** — high forces can kick the washer out of the work area, ending
the episode chain; a gentle failure leaves the part in place for a retry.

```
smooth   = 0.3·max_tilt/90 + 0.7·lift_tilt/90               (≤ 1)
capture  = captured && clearance>0 : 1 + 1.5·clip(clearance/8mm)
violence = 3.0·clip((max_dtilt − 20°)/30°, 0, 1)            (20°/sample grace)
escape   = washer flung/left the area : reward := −1 flat
reward   = smooth + capture − violence          (unless escape)
```

Consequences: gentle success ≈ 3.5 · gentle failure ≈ 0.7–1.0 · **violent
success ≈ 0.5 (below a gentle failure)** · violent failure ≈ −0.5 · escape −1.
Success alone is no longer what CEM climbs — *repeatable gentleness* is.
Metrics reported per eval: `success`, `clean` (success ∧ no snap), `escape`,
`max_dtilt` (20 Hz, pre-grip window).

## 4. Where the compute goes today (and the plan)

Profile of the current lab (8 rigs, 12×12 tiles + ~264 coupling drives each,
physics_dt 1/240, control in Python every step):

- **GPU ~idle** (0–3% util; ~600 MB): PhysX runs on CPU, rendering is off.
- Wall time ≈ 50 s per 8-eval round → **~9–10 evals/min**.
- Bottlenecks, in order: Python per-step control loop (240 Hz × K rigs of
  USD attribute writes), CPU solver on ~1.2k bodies / ~2.5k drives, per-tick
  XformCache queries.

Throughput bench (measured 2026-07-04 evening, 2 rounds each):

| config | s/round | **evals/hr** | GPU util | verdict |
|---|---|---|---|---|
| 8-rig CPU (baseline) | 42 | 686 | 0 % | — |
| 8-rig **GPU dynamics** | 35.5 | 811 | 4–16 % | +18 %, but the GPU solver produces **different dynamics** (snap counts and rewards shift) — a fresh transfer gap; rejected for training consistency |
| 24-rig CPU, 120 Hz control | 118 | 732 | 0 % | Python loop scales linearly with rigs — no win |
| 24-rig GPU, 120 Hz | 110 | 789 | 2–11 % | GPU doesn't rescue the Python loop either |
| **24-rig CPU, 60 Hz control, 10×10 bed** | ~65 | **~1300** | 0 % | **adopted** — decimation + 30 % fewer bodies/joints, same physics context |

Conclusion: the per-step Python USD-write loop is the wall (§5.2 — Isaac
Lab's tensor I/O removes it by construction). Within pick_lab, control-rate
decimation and a leaner bed bought ~2×; selection budget goes to **3×
repeated gauntlet panels**, which the eye-diagram rerun proved necessary.

## 5. Isaac Lab — recommendations for the next scale jump

The current rig maxes out around a few thousand evals/hour on one machine.
[Isaac Lab](https://isaac-sim.github.io/IsaacLab/) (NVIDIA's RL framework on
Isaac Sim) is built for exactly the jump past that:

1. **Vectorized GPU envs instead of hand-rolled multi-rig.** Isaac Lab clones
   one env design N× (hundreds–thousands) with `replicate_physics`, steps them
   in one GPU PhysX call, and exposes batched tensors. Our multi-rig trick is
   a manual version of this; Lab does it at 100× the width. The foundation
   (tile grid + joints) is a normal articulated asset — cloneable.
2. **Move state I/O off USD.** The single biggest porting win: Lab reads/writes
   body states through GPU tensors (`RigidObject.data`), not per-prim USD
   attributes. Our Python-per-step USD writes (the current #1 bottleneck)
   disappear by construction. (Interim tip even without Lab: use Isaac's
   tensorized `RigidPrim` views for the tiles/washer instead of XformCache.)
3. **Graduate the primitive to a closed-loop policy.** Keep θ-CEM as the
   curriculum starter, then train a small PPO policy (rsl_rl/skrl, built into
   Lab) whose observations are exactly what the robot will have: B-spring
   deflection (the touch sense), gripper pose, commanded phase — and whose
   actions perturb the primitive (Δz, Δgap, Δclose-rate per control tick).
   "Primitive + learned residual" keeps interpretability and adds the
   feedback that would fix pop-vs-roll on the fly. The smoothness reward
   transfers verbatim.
4. **Domain randomization as a manager, not hand-rolled jitter.** Lab's event
   manager randomizes material params (k, couple → our E′, t), friction, mass,
   placement per env per reset — our jitter panel becomes a config block, and
   the robust-selection problem largely dissolves into training-time DR.
5. **Consider the FEM deformable pad.** PhysX 5 supports GPU soft-body (FEM)
   volumes; Isaac Lab exposes deformable-body assets. A real deformable pad
   would replace 144 tiles + 264 coupling drives per env with one mesh — more
   realistic contact (true lateral compliance → square-tip drag "just works"),
   GPU-resident, and no α=4 solver-correction hack. Risks: contact tuning is
   its own art, deformables are GPU-pipeline-only, and env cloning with
   deformables is newer/less battle-tested — prototype with 1 env first.
6. **Practicalities**: Lab wants its own kit app + extension layout; our
   docker method carries over (same Isaac Sim base image). Keep pick_lab as
   the fast iteration harness and CI-style regression (it boots in ~90 s);
   use Lab for the wide training runs. Migration effort estimate: the
   foundation asset + Direct-workflow env ≈ 1–2 days; PPO residual training
   loop ≈ 1 day on top.

## 6. This phase's run plan

- Reward: §3. Bounds nudged toward the ramp regime (a_over capped below the
  flingy 0.65 bound-pin; press capped at 4.5 mm).
- Config from the throughput bench (§4), target ≥ 2× today's evals/hour.
- Budget: ~500–800 search evals on SOFT220 (+ MED as control if time allows),
  then 3× repeated 8-jitter panels on the top ~6 distinct candidates.
- Deliverables: eye diagrams of the winners (tight-ramp target), robust table
  with repeated-panel means ± spread, full-env replay of the winner, results
  appended to the robust-pick report.
