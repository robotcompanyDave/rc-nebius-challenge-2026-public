# MILESTONES — UR10e pick+sort training (2026-07-02)

Goal: train grasping/sorting **techniques** (not just single grasps) in a slim,
scalable replica of the platform's UR10e site, with checkpoints David can
verify by eye (videos) and by number (AUC / sort-success).

Ground rules established today:
- All generated artifacts land in `data/<YYYY-MM-DD>/<HHMM>-<name>/` (see data/README.md).
- Local box: RTX 5080 laptop, Isaac Sim 6.0 at `~/TOOLS/isaac-sim`. Cloud: Nebius Serverless AI Jobs.

---

## M3b — Compliant elastic surface (the real root-cause fix)  ✅ tilt validated 2026-07-03

The washer pick (M3a) failed because the "soft" surface was **kinematic tiles
snapped to an analytic dent field on a rigid 5 mm hard stop** — it doesn't
respond to contact, so it's bistable (flat, or snap-to-vertical). David's
technique (press one side → gentle 10–20° tilt → parallel-finger roll-up; see
`data/2026-07-03/1029-press_design/01_pick_sequence_v2.svg`) needs a surface
that deflects IN PROPORTION to load.

**Built** `graspsort/soft_foundation.py` — a Winkler foundation: a grid of small
DYNAMIC rigid tiles, each on a vertical prismatic joint sprung to rest (linear
DriveAPI k/c, generous travel). PhysX integrates the springs — **no per-step
scripting** (reuses the gripper-finger prismatic+drive construct). Probe:
`tools/probe_soft_tilt.py` (standalone: bed + flat m12 washer + low-µ rigid
finger pressing off-centre; measures tilt/slip/deflection vs press depth).

**Result (k=300 N/m, 4 mm tiles, µ=0.9, press offset 6 mm):**
- Rest: washer flat & stable, sink 1.06 mm, lateral 0 — no jitter.
- The surface now tilts the washer **continuously** (5°→11°→21°), NOT a snap.
- **Stable controllable dwell**: hold at 2.2 mm → **13.3°**, 2.6 mm → **14.9°**
  (rock-steady over the whole hold, no creep), lateral slip ~1.4 mm.
- Tipping cliff at ~2.8 mm: press past it and the washer topples to vertical
  (a coin-tipping geometric instability) — useful later (that IS the roll-up),
  but it bounds the stable window to ~2.2–2.7 mm → 13–15°.

**Verdict:** frame 2 achieved. The compliant surface is the surface model going
forward. Next: widen/verify with force ("constant pressure") control, then swap
`soft_rig` → `soft_foundation` in `sim_env`, then build the parallel-finger
roll-up (v2 diagram) on it; the tipping tendency helps the roll-up.
Scale caveat: ~200 sprung tiles/cell — fine for single-cell; watch body-count
for parallel data-gen (coarsen tiles or shrink the active patch).

---

## M3c — Two-finger press→drag→roll-up pick LEARNED  ✅ 2026-07-04

The v2-diagram technique works end-to-end, learned per material by CEM in
`tools/pick_lab.py` (K rigs simultaneously in ONE standalone physics scene,
~40 s per 8-candidate round; finger A = kinematic low-µ press pad, finger B =
capsule-tip grip finger on a compliant spring prismatic). Replayed + rendered in
the full env by `tools/render_pick.py` (phase stills + mp4; `GS_RP_SKIN=1` for
the foam-skin look). Material realism additions: `couple` (Pasternak shear
layer), `couple_falloff`/`couple_cutoff` (neighbour-of-neighbour influence with
finite radius), `SurfaceSkin` (non-colliding smooth overlay following the dish).

**Results (8 rigs × 8 rounds each, data/2026-07-04/lab_mats2):**

| material | k / couple / falloff | success | best lift |
|---|---|---|---|
| M0 as-is        | 300 / 150 / 0   | 59% | 64.9 mm |
| M1 drop-off     | 300 / 150 / 0.5 | 56% | 59.0 mm |
| M2 firm+local   | 600 /  80 / 0.4 | 42% | 54.4 mm (replay brittle in full env — open Q) |
| M3 soft+local   | 200 /  40 / 0.3 | **86%** | 34.9 mm |

**Learnings.** (1) Success tracks how PROUD of the dish the washer's far edge
stays: soft gives depth contrast, LOCAL coupling keeps the dish narrow — soft+
local is easiest (8/8 perfect rounds); wide influence (drop-off) reads more like
a continuous medium but slightly hurts pickability. (2) Maneuver structure
mattered more than θ: the close must happen AT DEPTH before the lift (closing
while rising leaves the rolled-up washer standing in the foam at +11.7 mm =
OD/2 − sink); and the commanded gap must go BELOW washer thickness so B's
spring produces a real pinch. (3) B needs a ROUND tip to plow foam (a box
corner jams on the laterally-rigid tile sides). (4) PhysX trap: re-authoring a
dynamic body's ops as translate+rotateXYZ mid-session freezes ROTATION
writeback — author translate+orient (repose_part carries the same latent bug).

---

## M0 — Local training loop restored  ✅ DONE 2026-07-02

Resume exactly where the laptop left off.

- [x] Restore 22 chunks (2,200 records) + interim model from DAVIDUSB.
- [x] Render current-state videos (heuristic 6/6 vs interim scorer 5/6 — compare.mp4 sent).
- [x] Generate the 18 missing chunks → **4,000 records** (18/18 clean boots,
      ~10 min/chunk → `data/2026-07-02/1501-dataset_v2_merged/`).
- [x] Retrain scorer on the merged 4,000 → **val AUC 0.823** (interim was 0.734),
      acc 0.789, learning curve still rising (0.841 at n=3000) →
      `data/2026-07-02/1755-model_v2/`. Per-class success: washer-flat 2.4%
      (the wall), nut-flat 15.6%, bolt-standing 61.5%.
- [x] Re-render heuristic-vs-scorer with the new model (+ washers in scene).

## M1 — Nebius training lane  ⛔ BLOCKED (outage day 3)

- [x] 50-attempt smoke probe (`aijob-<redacted>`, seed 999) —
      **FAILED with the outage signature** (~30 min PROVISIONING → ERROR 13,
      `instances: null`). Same as every job since 2026-07-01.
- [x] 2026-07-03: **preemptible × cross-region sweep — ALL FAILED the same
      way** (preempt-eun1-l40s / preempt-euw1-h200 / preempt-usc1-h200, ~2 h
      PROVISIONING → ERROR, no instance). Status page: all-operational.
      Preemptible pricing tier makes no difference → conclusively
      **account/tenant-side (quota)**, not capacity.
- [ ] David: ask Nebius support to check the tenant **GPU quota** (every
      failure is pre-allocation, all regions × all GPU types × on-demand AND
      preemptible, while their status page shows no incident).
- [ ] When a probe COMPLETES: run the submit script on the build host (2 × 2,000
      attempts, seeds 200/300, ~$14–19) → 8,000+ total records.

## M2 — Slim-site environment parity (audit + close gaps)

The training env must be the platform slim site minus the room: **no
floor/ceiling/furniture/lamps, but decent lighting**. Most of this exists —
parametric nuts/bolts/washers m6–m12, 5 mm soft surface, minimal stage
(ground + platform + dome/distant light). Gaps to close:

- [x] Three sort lanes incl. **washers at +2·DY** — already ported
      (controller.py:74, zone_for_kind/zone_slot_world_xy).
- [x] Fingernail-tip parity — the parametric jaw's pads run to the TCP,
      functionally the platform's fingernail tips (robot.py docstring).
- [x] Bundle scenes (~40% touching clumps) — already the default
      (randomize.sample_scene_multi; jobs/gen_data.py uses it).
- [x] Review renders now stage washers too (tools/render_sort.py — was
      nut/bolt only, so the washer lane never appeared in evidence videos).
- [ ] "Proper gripper": see M4 — Robotiq is a separate, timeboxed rung.

**Verify:** the 3-lane render is in the post-retrain review video set
(2026-07-02); audit vs sites/ur10e/SORT_TRAINING.md found no other gaps.

## M3 — Behaviour tree + gripper controller (the "techniques" milestone)

Make retry/escalation explicit and learnable instead of welded into the sequencer:

- [x] `graspsort/behavior.py` — non-blocking BT (select target → propose
      candidates → pick/place via the force-feedback sequencer → verify zone),
      escalation `direct → tilt → lip` across retries, clutter-aware washer
      yaw, per-attempt JSONL trace. Non-blocking tick() so it drops into
      parallel_env per-cell (M5).
- [x] Scorer picks the best candidate WITHIN the ladder-forced strategy.
- [x] First live run (tools/smoke_bt.py, 2026-07-02 → data/2026-07-02/1757-bt_smoke):
      **mechanism verified** — nut+bolt sorted first-try, washer failures
      escalated direct→tilt→lip correctly, clean give-up, 2/4 in 8 picks,
      trace reads perfectly.
- [ ] **Washer pick reliability** — the remaining gap: flat washers (m12 AND
      m6) beat all three strategies with `dropped_on_lift` at ~40 N clamp
      (slips out of the jaws on lift). Next levers: per-strategy gripper
      params (deeper grasp_dz, slower CLOSE_SPEED for washers), lip-press
      tuning, pad friction. This IS the training target (2.4% in the dataset).

**Verify (remaining):** video of a tilt/lip retry rescuing a washer;
**M6-washer-flat > 0/3**.

## M3a — Press-lift washer pick (David, 2026-07-02) — STAGE 1 ✅ physics proven

One open-jaw finger presses the near rim of a flat washer into the soft pad;
does the far rim lift enough for the other finger to get under it?
`tools/probe_press_lift.py` (26 trials, data/2026-07-02/2338+2342-press_lift):

- **YES — sustained 10–12 mm far-rim lip** (underside +9–12 mm above the
  surface, vs the ≥0 mm needed) at **spread_gain 1.0–1.1 with the press point
  at 0.9–1.0 of the rim radius**. m12 washer (OD 24, 2.5 mm).
- Mechanism: the dent must stay LOCAL under the pressing finger (default
  spread 1.5 sinks the whole washer → only ~1.8 mm); the far shoulder must
  REFORM under the rim (slow reform 0.1 → transient only). Default depth 5 mm
  and reform 0.25 are right; deeper stops (7 mm) violently pop the washer
  (44 mm launch); spread ≤0.9 or pressing at/beyond the rim pops instead of
  holding. The stable pocket is narrow but real and repeatable.
- NOTE for the site: spread_gain is a soft-MATERIAL param (platform ships
  1.5). Options: adopt 1.0–1.1 as the site material, or keep 1.5 and localise
  the dent with a sharper fingertip foot (the platform's fingernail tip) —
  test the latter first, it needs no material change.

**Stage 2 (2026-07-03, 21 instrumented runs)** — built the full coordinated
pick (`press_lift` in controller.py: servo → gentle decel press → snap-wait →
flip⇒sweep / slide⇒regrip; feedforward TCP pin so the press finger stays put
while the closing finger sweeps at 2× joint rate). Findings that matter:

- **The "sustained lip" of stage 1 was a mirage**: full lift curves show NO
  static lip ever forms (a 5 mm dent caps static tilt at ~1.5 mm); instead the
  pressed rim SNAPS out at a discrete instant (energy release) and the washer
  flips — sometimes landing standing/LEANING on the press finger (that was
  the "sustained 11.5 mm"), sometimes flying clear. Identical trials pop or
  sustain — it is stochastic (A/B 0038/0039-press_lift).
- Run 20: 6 presses → 1 true flip / 1 slide / 4 static. The flip fell flat
  again the moment the press finger retreated → flips must be swept+clamped
  IN PLACE (routing added), never regripped.
- Run 22 (final): z-pump raises the release rate to 5/6; full-flip case got
  ALL the way to a 37 N clamp on the vertical washer (0051 screenshots show
  it pinched between the pads) and slipped on lift — i.e. the chain now ends
  at the SAME generic thin-washer friction slip every strategy hits. The
  fingernail tip addresses both ends (real lip + mechanical ledge).
- **Conclusion: with flat 10 mm pads and the kinematic-tile soft model, the
  quasi-static lip David described cannot form.** The platform reached the
  same place: `finger_tip: "fingernail"` is its "~29%→100% reliability
  lever". NEXT: add a fingernail tip to the parametric gripper (thin strip
  contact that presses the rim WITHOUT roofing the washer) and re-run the
  stage-1 probe — expect a true lever lip, then the coordinated sweep works
  as designed.
- Debug ledger (all verified by telemetry/screenshot): false stall-contact
  from the pinned finger; xy-chase shoving the washer; symmetric-close stroke
  halving (fix: feedforward TCP shift); level-pads trenching; runaway z-goal
  integrator; tilt corner slipping off the rim; PhysX contact-offset standoff
  blocking the press coupling (STRICT overlap gate is load-bearing — do not
  soften); hard-contact watermelon-seed squirt (fix: decel contact); closing
  finger dragging on the hard stop (fix: memory-foam dent + 4 mm raise before
  the sweep); capture threshold vs pad-thickness reality (tip-centre gap when
  gripping ≈ W + 16 mm).

## M3d/M3e — Neoprene material + parallel-gripper gentle pick  ✅ core 2026-07-04

Full story in `data/reports/2026-07-04_materials.md` §4–7. Highlights:
- **Material calibrated to physics**: `neoprene(E,t,ν)` → k_cell = E′c²/t,
  k_link = α·G·t with measured PhysX solver correction α≈4; ratio-10 preset
  reproduces the target dent ramp (52/35/16 % over three neighbour tiles).
- **Proximity leveling** (`level_targets`) — tile drive-targets conform ahead
  of contact → **square fingertip works** on every material (capsule was a
  gel-era artifact). Pure-drive springs; no kinematic snapping.
- **M3e parallel gripper (David's redesign)**: both fingers bound as one EE,
  descend together (A ~50–60 % over the rim touches first), B's spring
  deflection = touch sensor, tiny brace, then pull-up with power-law close
  (close lags rise — B's friction rolls the washer up; close_pow ≈ 1.9 won).
  Success = clearing the PRESS-DOWN height. In-lab: SOFT220 46 % success,
  13 GENTLE full picks (tilt rate 6–13°/sample vs 52° for the M3d pop-catch).
- **Open**: touch trigger under-fires (grips ride the wait-timeout); lab→env
  transfer keeps CAPTURE but sheds clearance (1.3–1.9 mm vs 3+ in-lab) →
  next: robustness-aware elite selection (jitter seeds/placement/height).

---

## M4 — Robotiq 2F-85 rung (timeboxed ~½ day)

Blocked: the five-bar mimic linkage locks at ~1° in this standalone harness
while the same USD closes on the live platform (probed exhaustively —
`graspsort/robot.py` docstring, `tools/probe_gripper.py`).

- [ ] Try in order: Isaac 6.x point-release bump → PhysX MimicJointAPI gearing
      review → drive all six linkage joints directly (bypass mimics).
- [ ] If still locked: STOP at the timebox. Fallback is already sound — the
      parametric jaw uses frame-invariant features (yaw deltas), so learned
      policies port to the real 2F-85 on the platform.

**Verify:** `probe_gripper.py` shows finger_joint tracking to full close; one
grasp video with the Robotiq. (Or a documented "still blocked, fallback holds".)

## M5 — Scale-out: many identical cells in one sim

`graspsort/parallel_env.py` + `jobs/gen_data_parallel.py` (K replicated
UR10e+gripper+part cells sharing one physics step, batched IK ready) are built
but **never executed**.

- [ ] Switch per-round part respawn → re-posing (the ~100-attempt heap-drift risk).
- [ ] K=4 local smoke → fix the expected 1–2 Isaac-API issues.
- [ ] K=16/64: measure attempts/s vs the ~4 s/attempt sequential baseline;
      confirm label parity (same seeds → same success stats as sequential).
- [ ] BT from M3 runs per-cell.

**Verify:** throughput table (attempts/s vs K); success spread still 50–90%;
short top-down video of the grid of cells all sorting at once.

## M6 — Train at scale + the headline number

- [ ] Retrain on the full dataset (8k+ records, bundles, strategies).
- [ ] `eval_sort` before/after: heuristic vs scorer+BT, incl. washer lanes and
      M6 sizes.

**Verify:** report.json sort-success delta (v1 was 0.806 → 0.875; v2 should beat
that and cover washers); final side-by-side video.

---

Dependency sketch: M0 → M3 → M5 → M6, with M1 (cloud) and M4 (Robotiq) in
parallel and strictly timeboxed. M2 is a short audit before M3.

## M4 — Placement + EE-roll robustness (Nebius)  🟠 2026-07-10

Refined the Stage-T2 champion pick to tolerate a non-homogeneous surface: washer
placement jitter widened to **±1.5 mm** and a new **EE-roll axis ±20°** (per-rig
lattice rotation in `soft_foundation.py` via `rotz_deg`; the symmetric disc makes
surface-rotation ≡ EE-roll). New knobs `GS_PL_JIT_XY`, `GS_PL_JIT_ROLL`; `pick_lab`
spreads rig rolls over ±JIT_ROLL and prints best-θ each round.

Trained on **Nebius H100** (image `rc-grasp-sort:roll`, `GS_PL_GPU=1 RIGS=6 ROUNDS=15`,
warm-started from the champion). Converged (rnd13) to **succ 6/6, snaps 0** across
−20…+20° roll. Best θ = `[0.3823,0.0271,0.0040,1.1093,0.0020,0.0080,0.0018,0.0027,20.0]`
(reward 4.48, no flip, lift 49.7 mm). Key change: `close_pow 0.62→1.11` (close lags →
roll-up on B-friction first = flip-resistant). Champion flipped ~9/24 at wide roll; the
trained policy drives snaps 9→0.

**Open (Phase 1.5):** a per-reset memory leak crashes runs at ~96 rig-rounds (24 rigs→
rnd4, 12→rnd7, 6→rnd16), so the selection gauntlet was cut and spend capped ~$5 (< the
$10-20 target). Fix the leak to run 24-rig / 50-round depth + full-budget training.
**Phase 2:** material-noise robustness + surface co-optimization. Report:
rc-spike-nebius-basic/docs/2026-07-10_washer-roll-placement-training.html

### M4 update — transfer gap RESOLVED via env-in-the-loop tuning ✅ 2026-07-10

The lab-trained roll-θ scored only 3/12 in the deploy env (lab↔env gap; bed-span
hypothesis tested negative — no effect). Fix: **tune where you deploy** —
`tools/arena_eval.py` runs CEM inside the full GraspSortEnv (no render, fast;
eval + tune modes), chained as short Nebius H100 jobs under the ~96 rig-round
leak ceiling (`rc-spike-nebius-basic/job/chain_env_tune.sh`, each job's MUθ
seeds the next). Chain hit 11/12 at link 2 (~$1.80).

**Env-tuned θ = `[0.4704,0.0288,0.0034,0.4000,0.0018,0.0106,0.0022,0.0022,-5.55]`**
— independent validation **17/18 (94%), snaps 0**, every roll angle ≥2/3
(champion baseline 10/18). Env prefers −5.5° carry pitch vs lab's +18°; succeeds
even without the touch-trigger firing (9/18 trig, 17/18 succ). Video (all-rigs
camera, `GS_RA_CAM=all`, roll spread visible in the grid skins):
`RC/media/08 washer soft surface/2026-07-10_envtuned_arena_6rig_5of6.mp4` (5/6 on camera).

## M5 — Level 3: closed-loop residual policy (overnight, Nebius+local)  🟠 2026-07-11

Built the full closed-loop stack: `tools/resid_policy.py` — 1,603-param numpy
MLP (13 obs → 32 → 32 → 3) reading **B-spring deflection history** (≙ gripper
motor current) + proprio/phase at 20 Hz, emitting residuals around θ\* (Δz ±3mm,
Δgap ±1.5mm, close-rate ±30%); OpenAI-ES (antithetic + CRN + Adam), trained IN
the deploy env under wide DR incl. **material noise** (`SpringFoundation.retune`
re-authors spring/couple gains per episode — Phase-2 axis delivered). Survived
Nebius killing every job at ~21 min via WSAVE-to-logs + auto-relaunch chains
(`rc-spike-nebius-basic/job/auto_relaunch_es.sh`); ~100 updates across 3 seeds
(~$28 GPU + free 5080).

**Honest verdict (216 identical episodes, wide DR):** θ\* open-loop **75.9%±5.7,
18 snaps** vs policy **76.9%±5.6, 31 snaps** → statistical parity on success,
WORSE on flips. ES specification-games the reward toward violent rolls (snaps
2→13 over training); a −4·snap fitness penalty did not fix it at this scale.
**θ\* stays the deployed pick.** Models + full analysis: `models/resid_policy/`.

Learnings: (1) env-tuned open-loop is remarkably DR-robust — the residual
policy has little headroom until DR gets extreme; (2) ES at pop 24 with 1-ep
noisy fitness needs validation-based checkpoint selection (small panels
mislead: a "39/48 vs 36/48 win" evaporated at n=216); (3) next real step is
vectorized PPO (Isaac Lab cloning; `soft_foundation` anchor mode is ready) or
10× ES population, plus richer obs (deflection may arrive too late to react
within the 2 s maneuver).

## M6 — Material × technique co-optimization  ✅ 2026-07-11

Searched SOURCEABLE passive surfaces (physical space: E_eff × thickness ×
damping via `soft_foundation.neoprene()`, per-material travel limits) jointly
with the maneuver (θ now 10-dim: added early-close yaw ±15°). 9 candidates →
breadth tunes (9 concurrent ~14-min Nebius jobs, all COMPLETED under the
20-min kill window) → depth chains → 216-episode verdicts with material
sourcing noise ×0.85–1.18 + roll ±20° + placement ±1.5mm always on.

**WINNER: soft PU foam ~60 kPa / 10 mm** + tuned θ (yaw zeroed):
**89.4%±4.1 success, 1.4% flips** — **+20pp over the incumbent** (PU_med_12 +
θ\*: 69.4%). Neighborhood forgiving: 60–80 kPa and 7.5–10 mm all fine; ≥13 mm
drops hard. Runner-up: neoprene sponge 12 mm (77.8%, 6× flips). Eliminated:
EVA (58 flips/216 at scale), gel (36%), rubber/thin-neoprene (rigid-flip
returns), thick pads (washer swallowed).

θ_prod = [0.43924,0.02833,0.00380,0.40000,0.00189,0.01270,0.00191,0.00240,0.97188,0.0]

**Yaw hypothesis resolved honestly:** optimizer used it, 48-ep panel said +8pp,
n=216 said −7pp → production θ has yaw=0. Only large identical-episode evals
decide (same lesson as M5). Report:
rc-spike-nebius-basic/docs/2026-07-11_material-cooptimization.html
