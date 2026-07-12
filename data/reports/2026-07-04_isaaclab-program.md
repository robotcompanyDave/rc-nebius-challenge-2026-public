# Isaac Lab program — co-optimizing material + technique

*The scale jump from the [training-technique doc](2026-07-04_training-technique.html) §5,
with David's system-level objective. Written as the prereqs install
(2026-07-04 night) so the rest can run unattended.*

## The reframe: optimize the SYSTEM, not just the maneuver

Until now the material was hand-designed and the technique learned on top of
it. New program: **the material parameters are decision variables too.**

### Stage M — material optimization (first gate)

David's objective, verbatim intent: *a finger press on one side shall raise
the opposite end of the part so it meets the opposing finger at a height that
allows a clean pickup — say 5 mm.*

Formalized (the **5 mm meet condition**):

```
Given the canonical press (A at ~55% over the near rim, pressed 3 mm,
held 0.5 s):
  target:  far-rim TOP height  = +5 mm above the rest surface,
           steady (|d(height)/dt| < 2 mm/s over the hold),
           at the opposing finger's x-station (so B touches it mid-face)
  search:  material θm = (k_cell, couple ratio, damping, couple_damp
           [, thickness t via the neoprene() mapping])
  score:   -|far_rim_height - 5mm|  - instability penalty (oscillation of
           the rim height during hold)  - realism penalty (ℓ outside the
           measured foam band, §materials report)
```

This inverts the earlier finding that the realistic dish *swallows* the
washer: instead of accepting whatever rim rise the material gives, we ask
which material presents the part to the second finger — the material becomes
part of the gripper. The washer is CAPTIVE at touch height by construction,
which is exactly the regime the smooth-ramp winner (early-close + tall rise)
wants.

**Gate M**: a material θm whose canonical press yields 5 ± 1 mm steady far-rim
rise across the 8-jitter panel (placement ±1 mm, height ±0.5 mm).

### Stage T — technique training on the achieved material

Only after Gate M: re-train the maneuver (CEM warm start from the smooth-ramp
winner, then the PPO residual once vectorized) on the optimized material.
Smoothness reward unchanged (violence < failure, escape worst). Expectation:
with the part *presented* at 5 mm, the technique's job collapses to
touch → pinch → rise; the 12/24-under-jitter frontier should move
substantially.

### Stage MT — alternation (optional, after both)

1–2 rounds of coordinate ascent: refit material under the trained technique's
actual press distribution, then fine-tune technique. Stop when neither moves.

## Why Isaac Lab for this

- Material search × technique search is a *product* of spaces — needs the
  1000-env width, not 24 rigs.
- Materials differ per env clone (DR over k/couple = the M-stage search is
  literally a domain-randomization sweep with per-env logging).
- Tensor state I/O removes our measured #1 bottleneck (Python USD writes).
- The env-vs-lab registration gap dissolves: ONE env definition serves both
  training and evaluation.

## Prereq checklist (the part needing David present)

| item | status |
|---|---|
| Disk (need ~40 GB) | ✓ 734 GB free |
| docker group (no sudo needed) | ✓ |
| NVIDIA driver 595.71 + runtime | ✓ (already runs Isaac Sim 6.0) |
| Version pairing | ✓ Isaac Lab **3.0.0-beta2** ↔ Isaac Sim 6.0.x (our stack) |
| Image `nvcr.io/nvidia/isaac-lab:3.0.0-beta2` | ✓ pulled (48.6 GB) |
| GPU smoke: bundled task trains headless | ✓ **Cartpole 50 PPO iters in 11.2 s**, GPU peak 32 % / 3 GB |
| Repo mount + runner | ✓ `docker/run_lab.sh` (entrypoint bypassed — the stock wrapper trips on a deprecated torch ext; training path unaffected) |

**ALL PREREQS GREEN (2026-07-05 00:20). Nothing required David's presence —
the remaining program runs unattended.**

No system-level sudo items identified — the docker-first method carries over
unchanged (`--runtime=nvidia`, `NVIDIA_DRIVER_CAPABILITIES=all`, root in
container + ownership reclaim, named cache volumes).

## Overnight results (2026-07-05, unattended)

**Port: WORKING, parity PASSED.** `lab/press_sweep.py` runs the full rig
(anchored foundation + washer + bound gripper) at N envs on `push-grasp:lab`.
CPU parity vs pick_lab: end-of-press washer sink −2.39 mm vs −2.8 mm
reference, env-std 0.22 mm. Six port lessons are recorded in the commit
(boot order, cloner APIs, Lab reset teleports, view-driven kinematics
producing no contact, name ordering) — each cost one debug boot.

**Sweep 1 (256 materials, fixed press): zero Gate-M candidates** — rims
cluster at 0–1.6 mm; the fixed 2.8 mm / 50 %-overhang press lacks leverage.

**Sweep 2 (material × press system grid): one apparent hit**
(k300 / ratio 10 / d12 / press 3 mm / a_over 0.70 → rim 4.90 ± 0.21 mm)…

**…which the validation panels refuted.** Jitter panel: 0/24. Worse, the
slot-averaged rerun (each config × 8 grid slots) showed identical configs
swinging **±10 mm by env slot alone** — the see-saw regime that lifts the rim
to +5 mm is chaotically bistable (pop vs swallow) in the discrete tile model;
higher damping worsens it; the only slot-stable config presents at −3.2 mm.

**Conclusion — Gate M as specified is a measured NEGATIVE result:** the
discrete spring-tile material has **no stable +5 mm static equilibrium**. The
stable branch presents *below* surface; the raised branch is a knife-edge.

### Visual diagnostics (charts, sections, videos)

![the test arena](../2026-07-05/charts/arena_heatmap.png)

*The arena: 256 simultaneous press tests (one system config per cell); the
blue-boxed cell is the lone apparent 5 mm hit.*

![sweep landscape](../2026-07-05/charts/sweep_landscape.png)

![bistability](../2026-07-05/charts/bistability.png)

*The refutation: identical configs, 8 scene slots each — the +5 mm branch is
a knife-edge; damping widens the spread; k420 d16 threw washers.*

![winner on the jitter panel](../2026-07-05/charts/gate_panel.png)

![cross-sections](../2026-07-05/charts/cross_sections.png)

*Cross-sections at end-of-press (parameters on each panel). Two NEW insights
only visible here: (1) **finger B pre-crushes the landing zone** — bound to
A's depth, it presses the right-side tiles down ~4 mm exactly where the far
rim should rise → a shorter B fingertip (staying above surface during the
press) is a concrete design fix; (2) the k420 "pop" branch is really an
EJECTION (washer to +90° or clean off the pad).*

Failure videos (full-env replays of the three characteristic behaviours):
[winner-config](../2026-07-05/video_gateM_winner/pick.mp4) — swallow (tilt
5.6°, rim never presents) ·
[ejection-config](../2026-07-05/video_gateM_eject/pick.mp4) — washer flung
airborne, lands vertical (88°) ·
[stable-swallow](../2026-07-05/video_gateM_stable/pick.mp4) — *context
caveat:* half-popped to 58° in the full env though it swallows cleanly in
the Lab scene — the same slot/context sensitivity the panels measured,
now on camera.

### Proposed redefinition: Gate M′ — the DYNAMIC meet

The system that works (M3e already demonstrated it) presents the rim
*transiently* during the press, and finger B **catches it with the touch
sensor** — presentation is an event, not an equilibrium. Reframe the material
objective as: **maximize dwell time of the far rim in the 4–6 mm band during
a slow press** (plus the existing smoothness constraints). Dwell time is a
robust, slot-averageable scalar; the technique's touch trigger converts the
dwell window into a grip. Stage T is unblocked under this framing — the
material search and technique training co-optimize dwell × catch.

*(Alternative long-road: FEM deformable pad — a continuum has no discrete
bistability; parked pending the above.)*

## Unattended work plan (after prereqs green)

1. **Lab smoke**: bundled Direct task (e.g. Cartpole) 100 iters headless →
   proves GPU PhysX + RL loop end-to-end. Record it/s.
2. **Foundation asset**: SpringFoundation as a Lab-cloneable asset (tile grid
   + prismatic drives + coupling); washer + two-finger gripper as articulation.
   Numeric parity check against pick_lab (dish profile, ℓ).
3. **`M3f-material` Direct env**: canonical press, per-env material params,
   the 5 mm meet objective; grid+CEM over θm at ~512 envs. → **Gate M**.
4. **`M3f-pick` env**: the parallel-gripper maneuver with the smoothness
   reward; CEM warm start, then PPO residual (rsl_rl). → success/clean rates
   vs the pick_lab frontier.
5. Reports + eye diagrams ported (tilt-vs-closure from tensor traces —
   trivially batched now).

Risks/watch: 3.0.0-beta2 is a beta (pin exact tag; keep pick_lab as
regression harness); deformable-FEM pad stays OUT of scope until the
spring-tile Lab port reproduces pick_lab numbers.

## Beta findings (2026-07-05, overnight port)

1. **The NGC isaac-lab:3.0.0-beta2 image's Kit is broken on this box** — every
   Kit-based script (including the bundled `create_empty.py` tutorial) dies
   ~15 s into extension loading: the deprecated `omni.isaac.ml_archive`
   prebundled torch fails (`undefined symbol: ncclDevCommCreate` under driver
   595) and Isaac Sim's torch stub exits the app cleanly (exit 0 — nasty to
   diagnose). `--disable-ext` / experience overrides didn't bite.
2. **The green "GPU smoke" was real but kitless** — Isaac Lab 3.0's Cartpole
   ran on the new Newton backend (that's why 50 iters took 11 s with no shader
   compile). RL infrastructure ✓; Kit PhysX in that image ✗.
3. **Resolution: `docker/Dockerfile.lab`** — Isaac Lab v3.0.0-beta2 cloned and
   pip-installed into our PROVEN `isaac-sim:6.0.0` image (the documented
   binary-install path; that Kit runs pick_lab daily). `push-grasp:lab`.
4. Port groundwork done meanwhile: `SpringFoundation(anchor=...)` makes the
   vertical springs clone-safe (env-local coords against a per-env anchor
   body instead of world-frame joints), and `lab/press_sweep.py` implements
   the Gate-M canonical press + parity + material-grid sweep.

## Gate M′ results (2026-07-05, dwell sweep + slot validation)

Implemented as `press_sweep.py --mode dwell` (256-env system grid, rim
sampled at 20 Hz through press+hold; dwell = time in the 4–6 mm band) and
`--mode gate2` (finalists × 8 scene slots, dwell per slot).

![dwell slots](../2026-07-05/charts/dwell_slots.png)

**The winner is a parker, not a transiter — and it's the heavy-damping
variant:**

| candidate | dwell per slot (ms) | behaviour |
|---|---|---|
| k300 r10 **d24** p3 ao.70 | **1000, 950, 0, 0, 0, 0, 1000, 0** | 3/8 slots ramp smoothly INTO the band and **park at ~5 mm for the entire hold**; the rest settle at −2.5 mm. Zero violence in either branch. |
| k300 r10 d12 p3 ao.70 | 0 × 8 | the 256-env sweep's "900 ms" cell — its slot was lucky. In this layout: never leaves −3 mm. |
| k220 r10 d12 p4 ao.70 | 0–100 | transits the band in ≤ 100 ms at ~20 mm overshoot — too fast for the 42 ms trigger debounce to guarantee a catch. |
| k420 r6 d8/d16 p4 ao.50 | ~0 | ejection branch: rim spikes off-scale (washer flung). |

Two sobering reproducibility notes, both now measured:

1. **Slot chaos persists under the dwell metric** — the same config parks in
   3/8 slots and stays down in 5/8. Dwell is smoother than the static gate
   (park vs stay-down, not pop-vs-eject), but the discrete tiles still
   bifurcate by grid phase.
2. **The bifurcation is layout-global**: re-running the identical 5 configs
   with 16 extra envs appended (56 total vs 40) *changed the per-slot
   pattern* — the d12 config went from (min −3.2 / max +10.4) to (uniform
   −2.8). Nothing about an env changed except how many neighbours the stage
   holds. This is the discreteness disease in its clearest form yet.

**Verdict**: the tile model's honest ceiling under Gate M′ is
**k300/r10/d24/p3/ao.70 with ~3/8 slot catch-probability** (parked rims are
trivially caught by the touch trigger; transits are not needed). That is
enough to unblock Stage T — the technique trains against the parked branch,
and the RL agent learns to *retry* on the stay-down branch (a press that
doesn't present costs one gentle re-press, not a lost part). In parallel, the
slot chaos is the strongest argument yet for the FEM re-evaluation
([proposal](2026-07-05_fem-proposal.html)) — a continuum cannot depend on
grid phase.

## Stage T results (2026-07-05 evening)

Technique trained on the Gate-M′ parker material (k300/r10/**d24**, square
tip, leveling, parallel-gripper mode): 24 rigs × 100 CEM rounds ≈ 2,400
evals at 60 Hz, search jitter on, smoothness-first reward, seeded from the
M3e early-close winner. Two preludes, one negative and one decisive:

- **Beside-press primitive: rejected.** With A fully on the pad the dent
  never reaches the washer (Pasternak coupling is local by design — the
  FEM lever came from the wide-bowl behaviour we rejected); with overlap
  it's the violent transit again. Corollary: in the parker's parked slots
  the rim is **leaning on the descending B** — B is the wall that makes
  parking stable, so the full parallel press stays the primitive.
- **Gauntlet: two candidates at 24/24 jittered successes** (M3e's best was
  12/24 on SOFT220) — the d24 parker + early-close family is simply a
  better-behaved system.

**Full-env exit gate: PASSED — first `clean` replay of the project.**

| candidate | gauntlet | env replay |
|---|---|---|
| cand3 (robust 2.71) | 24/24 | capture + roll-up, **slips at top of rise** (lifted 6.8 mm, ends flat on bed) |
| **cand0** (robust 2.65) | 24/24 | **captured, vertical 89.5°, clearance 3.97 mm, success + clean** |

θ(cand0) = [0.5566, 0.0267, 0.0042, 0.4606, 0.0016, 0.0269, 0.0023]
(a_over 0.56, g0 26.7 mm, press 4.2 mm, close_pow 0.46, gap 1.6 mm,
rise 26.9 mm, brace 2.3 mm).

The lab↔env offset still reorders finalists (the higher-robust candidate
was the one that slipped) — carrying multiple gauntlet survivors to the env
gate is now standing procedure, and the same-θ trace-diff diagnostic
remains the top infrastructure debt.

### Stage T showcase (grid-skin renders, 90 mm bed)

Workshop-style grid skin (10 mm texture from rc-remote-platform), bed
enlarged to 90 mm so the deformation halo is fully contained — the grid
lines make the dish extent readable out to where the coupling cuts off.
Replay is the exact cand0 θ; result reproduces clean (89.8°, 3.96 mm).

![press dish](../2026-07-05/stageT1_hero_grid2/hero_1.png)
![vertical hold](../2026-07-05/stageT1_hero_grid2/hero_2.png)

[full clip](../2026-07-05/stageT1_hero_grid2/pick.mp4)

**Carry-height margin (found while making these):** raising the pull-up
from the trained 27 mm to 35 or 45 mm drops the washer at the top of the
rise — the pinch after roll-up is real but marginal. The trained maneuver
is stable at its own rise; carry height is now a tracked robustness metric
for the next training round (see next steps).

## Stage T2 — carry phase + wrist pitch (2026-07-06)

Retrained with the carry extension (rise to a 45 mm hold) and two new θ
dims from the 6-DOF unlock: **gap_carry** (post-capture squeeze) and
**pitch_deg** (wrist pitch about the fingertips during carry).

- In-lab best carried **53 mm** (reward 4.50). CEM drove pitch to the
  18–20° bound in 3/4 finalists — tilting the pinch seats the washer —
  and restructured the maneuver (pull-up 27→8 mm; the carry does the
  lifting).
- Jitter robustness dropped vs T1 (11–17/24; the task got harder), and the
  gauntlet ranking again anti-predicted transfer:

| candidate | gauntlet | env gate |
|---|---|---|
| cand3 "winner" (robust 0.06) | 16/24 | slips, flat |
| cand0 / cand1 | 15, 11/24 | slip, flat |
| **cand2** (robust **−1.85**, lowest) | 17/24 | **captured, carried 41.5 mm, carry_ok, clean, pitch 17.6°** |

θ(cand2) = [0.5424, 0.0286, 0.0040, 0.9616, 0.0018, 0.0083, 0.0014,
0.0025, 17.59]. **Standing procedure reaffirmed**: every gauntlet finalist
goes through the env gate; three rounds running, the env gate has picked a
different winner than the gauntlet. (Trace-diff root cause: no static
offset; the env B-spring responds ~0.25 s faster —
[chart](../2026-07-05/charts/tracediff.png).)

### Progress reel (env replays of round-best θ, grid skin)

[progress reel — rounds 0→99](../2026-07-06/stageT2_carry/progress_reel.mp4)

Carry height in ENV replay creeps up over training: 26 mm (seed) → 29 mm
(r25) → 34 mm (r99), with a lab-overfit dip at r50 that fails outright —
the visible history of the process getting better that these reels are for.

[cand2 full carry](../2026-07-06/t2gate_c2/pick.mp4)
