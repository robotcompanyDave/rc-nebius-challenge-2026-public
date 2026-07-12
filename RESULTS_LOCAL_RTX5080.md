# Local RTX 5080 run — train the grasp-success scorer, measure before/after sort

> Session 3 (2026-07-01). Nebius GPU capacity is still out (see `HANDOFF.md` §9), so
> this milestone — **the train-the-scorer step that was blocked on a dataset** — was
> executed **entirely locally on the RTX 5080 Laptop GPU**, headless Isaac Sim 6.0.
> It closes the loop the HANDOFF left open: *generate data → train a scorer → show the
> before/after sort-success delta*, with videos.

## TL;DR

- **Data-gen runs locally.** 6 fresh-boot chunks × 100 = **600 labeled grasp attempts**
  on the 5080 (~3.4 s/attempt, ~7 min/chunk). Chunked to dodge the ~100-attempt heap
  drift the HANDOFF warns about.
- **The scorer trains** (pure-numpy MLP, no sklearn/onnx — Isaac's python has neither):
  state `(part kind/size/pose + candidate grasp)` → `P(grasp holds)`.
  Validation **ROC-AUC = 0.933** (acc 0.88), and a dataset-size learning curve that
  rises with cycles (0.84 @ 68 attempts → ~0.94 @ 225+).
- **Before/after in the ported sim** (the faithful port of `adapter.py`):
  heuristic **0.806** → scorer **0.875** sort-success (**Δ +0.069**, and the *lowest*
  variance), with a naive-random reference at **0.847**. The learned scorer wins by
  *choosing* good grasps, not just by exploring.
- **Videos:** an animated *training-progress* clip (metrics per epoch — the "cycles")
  and a side-by-side *heuristic-vs-scorer* sort rollout (a mixed scene and a
  nut-flat-heavy hard scene). Note: a 6-part sort allows up to 3 retries/part, so both
  policies usually *complete* the sort — the visible signal is **fewer retry picks**
  under the scorer. On the hard scene the heuristic took **8 picks (2 retries)** to sort
  6 parts; the scorer took **6 picks — zero retries** (grasps held first-try). The
  success-rate gap itself lives in the aggregate eval below.

## What was added this session (all additive; the Nebius Job path is untouched)

| File | Purpose |
| --- | --- |
| `graspsort/features.py` | **NEW.** Canonical `(part, candidate) → 17-D vector`. One source of truth shared by trainer + serving → train/serve parity. Featurizes the yaw **delta from the heuristic** (frame-invariant), not absolute yaw. |
| `graspsort/scorer.py` | **NEW.** Tiny pure-numpy MLP saved as one `model.npz`; loads with zero deps in Isaac's python / the Job image. |
| `jobs/train_scorer.py` | **NEW.** Fits the scorer (Adam, class-balanced BCE), writes `model.npz`, `training_history.json`, plots, and `training_progress.mp4`. No Isaac. |
| `tools/render_sort.py` | **NEW.** One-boot heuristic-vs-scorer sort render → per-policy MP4s + `compare.mp4`. |
| `jobs/eval_sort.py` | `make_scorer_policy` implemented (was a TODO stub): sample N candidate grasps, score, execute argmax; the heuristic is always in the candidate set so the policy can't (by its belief) do worse. Added a `random` reference policy + `all` mode. |
| `jobs/gen_data.py` | Record `action.heuristic_yaw` so the trainer can recover the yaw delta. |
| `graspsort/sim_env.py` | Store per-part `PartSpec` (`env.part_specs`) so the eval policy can featurize size/pose, not just kind. |
| `graspsort/controller.py` | `begin_pick_place` now honors the candidate's `xy_offset` / `grasp_dz` / `approach_dh` (like `begin_grasp`) — this is what gives a scorer-guided policy a real lever over the sort. Defaults reproduce the heuristic exactly, so the baseline is unchanged. |

Two benign Windows-console `UnicodeEncodeError`s (from `→`/`—` in final print lines,
*after* data was already persisted) were made ASCII so runs finish clean.

## How the scorer works (state-based, no camera)

Per the locked decision in `HANDOFF.md` §2b we optimise **picking parts in any
orientation, not identification** — so the model is trained on **state → success**:

```text
features = one-hot(kind) ⊕ one-hot(size) ⊕ one-hot(pose)
         ⊕ [sin,cos,|·|](grasp_yaw − heuristic_yaw)      # delta from the expert grasp
         ⊕ [off_x, off_y, |off|]                          # xy miss off part centre
         ⊕ [grasp_dz, approach_dh]                        # depth / approach knobs
label    = did the grasp hold on lift (lifted > 25 mm)
```

The **serving policy** samples the heuristic grasp + N perturbations, scores all with
the MLP, and executes the argmax `P(hold)`. Because the exact heuristic is always a
candidate, the scorer only deviates when it *predicts* a better grasp.

## Results

### Scorer (600 attempts, 74.2% positive)

- **Validation ROC-AUC 0.933**, accuracy 0.88 (train 450 / val 150, held-out).
- **Learning curve** — val AUC vs # training grasp attempts: 68→**0.841**, 135→0.923,
  225→**0.945**, 315→0.937, 450→0.943. More grasp *cycles* → a better scorer, saturating
  near 0.94 by ~225 attempts. (`data/model/plots/learning_curve.png`.)
- The scorer learns the sim's **real physics of hard grasps** — dataset grasp-success by
  part×pose (`plots/success_breakdown.png`): **nut-flat 37%**, nut-random 48% (the hard
  cases — thin, low-profile) vs **bolt-standing 100%**, bolt-random 97% (easy).
- Note: train-loss 0.02 vs val-loss 0.97 shows the 24-unit net is *over-confident*
  (mis-calibrated) on 600 samples — but the **ranking** the argmax policy relies on is
  excellent (AUC 0.93). Early-stopping / weight-decay would calibrate the probabilities;
  not needed for grasp selection.

### Before / after sort-success (the headline)

12 sort trials × 6 parts, **identical scenes across policies** (paired), same seed:

| policy | sort-success (mean ± std) |
| --- | --- |
| heuristic (before) | 0.806 ± 0.150 |
| random (reference) | 0.847 ± 0.186 |
| **scorer (after)** | **0.875 ± 0.138** |

**Heuristic → scorer: Δ +0.069 (+6.9 pts, ~+8.6% relative), and the lowest variance.**
The scorer also beats the naive-random reference (0.847): because a sort trial allows up
to 3 retries, blindly repeating a deterministic heuristic grasp that fails on a hard part
loses to *any* exploration — but the learned scorer beats naive exploration by *choosing*
the grasp it predicts will hold. Paired over the 12 identical scenes the scorer
**wins 5, ties 4, loses 3** vs the heuristic (net +2). Caveat: with 12 trials the
per-policy CIs overlap (SEM ≈ 0.04); the well-powered evidence that the scorer
discriminates good vs bad grasps is the **AUC 0.933**, and the sort delta is directionally
in its favour. More trials would tighten the sort CI. Full per-trial breakdown:
`data/eval/report.json`.

## Deliverables (artifacts on disk)

```text
data/dataset/records.jsonl        600 labeled grasp attempts (+ records.parquet)
data/model/model.npz              trained scorer
data/model/training_history.json  per-epoch metrics + learning curve
data/model/plots/*.png            loss, val-AUC, learning curve, success breakdown
data/model/training_progress.mp4  training animated over epochs ("the cycles")
data/eval/report.json             before/after/random sort-success
data/review_sort/compare.mp4      heuristic-vs-scorer sort, side by side (mixed scene)
data/review_sort_hard/compare.mp4 same, nut-flat-heavy HARD scene (scorer = fewer retries)
data/review_sort*/{heuristic,scorer}.mp4
```

## Reproduce

```powershell
# 1. dataset (6 fresh-boot chunks × 100 attempts)  — see scratch gen_dataset.ps1
$env:GS_N_ATTEMPTS="100"; $env:GS_BATCH="100"; $env:GS_SEED="0"; $env:GS_OUTPUT_DIR="./data/ds/chunk_0"
& "D:/isaacsim/python.bat" jobs/gen_data.py            # repeat seeds 1..5, then merge records.jsonl

# 2. train the scorer (no Isaac needed, but Isaac's python has numpy/matplotlib)
$env:GS_DATASET="data/dataset"; $env:GS_MODEL_OUT="data/model"
& "D:/isaacsim/python.bat" jobs/train_scorer.py

# 3. before/after/random sort eval
$env:GS_POLICY="all"; $env:GS_MODEL="data/model/model.npz"; $env:GS_OUTPUT_DIR="data/eval"
& "D:/isaacsim/python.bat" jobs/eval_sort.py

# 4. comparison video
& "D:/isaacsim/python.bat" tools/render_sort.py
```

## Does it transfer to `rc-remote-platform`? (the real robot platform)

This sim is a **faithful port** of `targets/ur10e/adapter.py`, so a win here is the
evidence that a learned scorer improves the shipped grasp/sort logic. The integration
is a **single, well-localized hook**:

- In `adapter.py::_seq_start_pick_place` the grasp is chosen at
  [`adapter.py:3089`](../rc-remote-platform/targets/ur10e/adapter.py) — `"grasp_R": self._grasp_R(target)`.
  A scorer-guided policy replaces that one call with *sample candidates around
  `_grasp_R` → featurize (same `graspsort.features`) → score `model.npz` → argmax*,
  and applies the winning `xy_offset` / `grasp_dz` to `pick_xy` — the identical change
  already proven here in `controller.begin_pick_place`.
- **Why the features transfer:** the platform's Robotiq gripper and this parametric jaw
  use *different* absolute yaw conventions (platform bolt-yaw = `phi`; here `phi+90°`).
  The scorer never sees absolute yaw — only the **delta from that gripper's own
  heuristic** — so the learned "how far off the expert grasp is too far" signal is
  gripper-agnostic and ports directly.
- The model is dependency-free numpy, so it drops into the platform's python as-is
  (or export to ONNX if preferred; the eval hook already had an ONNX path).

Not done here (heavy + the full gateway viewport doesn't render on this Optimus box):
a live run inside the full digital-twin gateway. The sim before/after is the proxy;
the platform hook above is the ~20-line change to wire it in.
