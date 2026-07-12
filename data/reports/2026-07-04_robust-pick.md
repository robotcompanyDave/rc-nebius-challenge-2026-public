# Robust pick selection — closing the lab→env gap

*Phase goal: a parallel-gripper washer-pick policy that survives context
changes — selected under jitter in the lab, validated by replay in the full
environment with ≥ 3 mm clearance. Follow-up to
[2026-07-04_materials.md](2026-07-04_materials.md) §7 (M3e), where CAPTURE
transferred but clean clearance did not.*

## The problem

M3e's CEM selects winners on **single lab runs**. The elite that looks best is
often the one that got lucky: marginal candidates (gentle 3–5 mm-clearance
picks in-lab) fail outright when replayed in the full environment, while the
policy *family* clearly works (capture reproduces every time). Selection is
optimizing for peak, not for robustness.

## Method

1. **Noise during search** — every CEM evaluation gets light jitter (washer
   placement ±1 mm, random yaw, gripper height ±0.5 mm), so the optimizer
   can't camp on knife-edge solutions.
2. **Jitter gauntlet for selection** — the top candidates are each re-run on
   8 *different* jitters simultaneously (one per rig — the same trick that
   gives us a CEM population gives us a robustness panel). Final winner =
   best **worst-2-mean** across the panel, not best single run.
3. **Exit gate** — the winner must replay in the full `GraspSortEnv` with
   ≥ 3 mm clearance above the press-down height.

Also in this phase: touch-sensor debounce (the M3e trigger under-fired and
grips rode the wait-timeout), and a Nebius capacity probe on non-L40S GPUs
(L40S at capacity; trying H100) for faster ladder throughput if it provisions.

## Results

**Search under noise finds deeper basins.** With every eval jittered, CEM's
best single runs *improved* over the noise-free M3e ladder (SOFT220 best
clearance 7.2 mm vs 6.0; press pushed to the new 4.8 mm bound, close_pow ≈ 2.5).
Training on noise did not cost performance — it bought it.

**The gauntlet works exactly as designed.** SOFT220's panel
(`data/2026-07-04/lab_robust/results.json → gauntlet`):

| candidate | panel successes | mean | **worst-2-mean** |
|---|---|---|---|
| cand0 (winner) | 4/8 | 2.68 | **2.32** |
| cand1 | 3/8 | 2.47 | 1.60 |
| cand2 | **6/8** | 2.39 | **0.32** |
| cand3 | 4/8 | 2.44 | 1.43 |

cand2 is the cautionary tale: *most* successes on the panel, but its failures
are catastrophic — single-run (or even mean-based) selection would love it.
Worst-2-mean picks cand0, whose bad days still hold the washer. MED250's
winner scored only 1.61 → **SOFT220 confirmed as the pick surface**.

### What actually separates the candidates

The four are near-siblings — press identical (4.8 mm, at the bound),
brace/close_pow/g0 within a few percent. The discriminating knobs:

| param | cand0 ★ | cand1 | cand2 ⚠ | cand3 |
|---|---|---|---|---|
| a_over (rim overhang) | **0.622** | 0.650 *(bound)* | 0.650 *(bound)* | 0.626 |
| pinch gap (mm) | **2.03** | 2.04 | 2.21 | 2.20 |
| rise (mm) | 19.2 | 16.2 | 17.7 | 19.0 |

Panel per jitter — reward (clearance mm), `+` full success, `X` failure:

| jitter | cand0 | cand1 | cand2 | cand3 |
|---|---|---|---|---|
| nominal | 2.19 (1.0) | 2.42 (2.3) | **0.35 X** | 2.92 (4.9) + |
| x +1 / −1 mm | 2.95 + / 2.96 + | 2.37 / **0.90 X** | 2.94 + / 2.97 + | 2.49 / 2.44 |
| y +1 / −1 mm | 2.45 / 2.46 | 2.92 + / 3.28 + | **0.30 (−283!) X** / 3.30 + | 2.91 + / **0.42 X** |
| z +0.5 / −0.5 mm | 2.85 + / 2.56 | 2.30 / 3.02 + | 2.84 + / 3.38 + | 2.82 + / 3.01 + |
| combo | 2.99 + | 2.56 | 3.03 + | 2.55 |

Physical reading:

- **a_over = 0.65 (bound-pinned) is flingy.** Pressing 65 % off the rim gives
  the pop more energy: cand2's y+1 failure has clearance **−283 mm** — the
  washer was flung off the bed entirely; its nominal failure popped the washer
  7 mm up and never caught it. Six great runs, two violent ones.
- **Gap 2.2 mm is a loose pinch.** The washer is 2.5 mm thick: 2.2 leaves only
  0.3 mm of spring interference (vs 0.5 at 2.0 mm) — cand2/3's failures are
  mostly "popped but not held".
- **Off-centre-in-y is the killer jitter**: edge contact lands off the finger
  midline, the pop goes asymmetric, the washer escapes sideways.
- **cand0 wins by having no failure mode, not by brilliance** — tamer press
  position, real pinch, tallest rise. Its worst cells are *low lifts while
  still holding* (nominal 1.0 mm!), never fireworks.

Consistency check: cand0's worst panel cell is its **nominal** (1.0 mm) and
the full-env replay landed at 2.0 mm — the env behaves like a mid-grade
jitter, supporting the constant-offset theory over anything mysterious.

*(The replay videos below are cand0, replayed nominally in the full env.)*

**Exit gate: closer, not passed.** The robust winner replays in the full env
**captured and fully vertical (89.9°) at 2.0 mm clearance** — the best
transfer yet (M3e's hand-picked heroes managed 59° / 1.9 mm) — but short of
the 3 mm gate. The remaining gap is *systematic*: every env replay sheds
~3–5 mm of clearance relative to its lab twin, regardless of candidate. That
constant offset (bed-on-slab vs free-hanging tiles, or a floor-reference
mismatch) is now the single diagnostic to chase — instrument both sides with
the same θ and diff the washer z/tilt traces.

Replay renders: [tiles video](../2026-07-04/render_robust/pick.mp4) ·
[skin video](../2026-07-04/render_robust_skin/pick.mp4)

![robust winner lifted](../2026-07-04/render_robust/p3_lifted_zoom.png)

**Nebius: parked.** H100 (`gpu-h100-sxm`) probe hit the same
PROVISIONING→ERROR as every L40S attempt — the outage is **tenant-level
quota**, not GPU-type capacity. Escalate via support ticket; no more probes.

## Eye diagrams — roll smoothness per candidate

Washer tilt (90° = vertical target) vs **finger-closure fraction**, all 8
jitter-panel runs overlaid per candidate, bold line = mean (green = success,
red = fail). Like a telecom eye diagram: a tight bundle = a repeatable,
smooth roll; scattered traces = luck.

![eye all candidates](../2026-07-04/lab_eye/eye_all.png)

Individual: [cand0](../2026-07-04/lab_eye/eye_cand0.png) ·
[cand1](../2026-07-04/lab_eye/eye_cand1.png) ·
[cand2](../2026-07-04/lab_eye/eye_cand2.png) ·
[cand3](../2026-07-04/lab_eye/eye_cand3.png)

How to read them:

- Traces that jump straight to 90° at closure ≈ 0 reached vertical **during
  the press/brace** (the pop happened before any closing) and the fingers
  just collect the washer.
- The interesting traces are the **ramps**: a gradual climb across the
  closure axis is the true two-contact roll. **cand1 has the tightest ramp
  bundle** — three traces climbing 25°→90° almost on top of each other, the
  cleanest "open eye" of the four.
- Dead lines at 0° are misses (washer escaped before the grip).

**And an important honest finding:** this trace run re-executed the *same 4
candidates on the same 8-jitter panel* — and the scores moved (cand0
robust 2.32 → 0.57, cand1 1.60 → 2.48; the "winner" flipped). Same θ, same
jitters, fresh scene boot → different outcomes. The gauntlet measures a real
distribution, and one panel is one sample of it: **selection needs repeated
panels (2–3×) before the robust score is trustworthy.** This run-to-run
sensitivity is also the lab↔env transfer story in miniature — the maneuver
lives near a stability boundary, and cand0/cand1 are statistically
indistinguishable on current evidence (cand1's roll shape is the smoother).

## Smooth-ramp scale-up (late night) — violence bred out, strategy discovered

Directive: *rate sharp jumps and flips worse than failure* (high forces can
kick the part out of the area, ending the episode chain), run many more
iterations, and squeeze the hardware. Method + bench in the
[training-technique doc](2026-07-04_training-technique.html); config adopted:
24 rigs, 60 Hz control, 10×10 beds ≈ **2× evals/hour**, physics context
unchanged (GPU solver rejected — it *changes the dynamics*).

**Run 1 — 576 evals under the new reward.** Violence got bred out on schedule:
snaps 22/24 → 7/24 across rounds, population mean |Δtilt| 47°→21°/sample,
12 escapes in 576 (2 %). All top-6 candidates were **clean successes**
(Δtilt 9–14°/sample, clearance 4.5–5.5 mm) — under the old reward the top
was ~50°/sample. But the 3× repeated panel (24 jitters each) exposed the
frontier honestly: candidates were either *sometimes-successful with violent
tails* (10–12/24 with occasional escapes) or *never-violent but rarely
successful* (2/24). The eye diagrams also identified the dominant violence
mode under jitter: **chatter** — 30↔85° sawtooth oscillation of the washer
between the fingers during the close (not the press pop).

**Run 2 — refinement seeded at the gentlest shape.** The stall-ramps in the
eye diagram said "under-driven, not wrong", so a second CEM (288 evals) was
seeded there. It converged fast (15/24 successes with 1 snap by the final
round) and **discovered a cleaner strategy**: `close_pow → 0.4` (at the
bound) + `rise → 31.5 mm` (at the bound) — **pinch EARLY at low tilt, then a
tall slow rise rolls the captive washer up between the fingers**. The washer
is held the entire way; there is nothing left to fling.

**Final 3×-panel (24 jitters per candidate):**

| candidate | success | mean | worst-2 | escapes | character |
|---|---|---|---|---|---|
| **refined early-close (winner)** | **12/24** | **1.90** | 0.08 | **0/24** | worst cases are *picks with chatter* (clr 4.8–5.3 mm even in the bottom-5) |
| refined variant | 9/24 | 1.65 | 0.12 | 0 | similar |
| tall-rise variant | 6/24 | 0.36 | −2.09 | 1 | one escape tail |
| original gentle control | 2/24 | 0.66 | 0.29 | 0 | safe, under-driven |

The formal worst-2 metric ties the winner with the old control (0.08 vs 0.29,
within panel noise) — but at 6× the success rate with **zero part losses in
24 jittered attempts**, the early-close strategy is the deployment pick.
Winner θ = `a_over 0.56, g0 27.8, press 3.3 mm, close_pow 0.4, gap 1.8 mm,
rise 31.2 mm, brace 2.3 mm`.

### The final four, in parameters and in words

*(Naming note: cand numbers restart per phase — these are the FINAL-gauntlet
four, all on the same SOFT220 material; they differ only in maneuver
parameters. cand0–2 came from the refinement run, cand3 is the pre-refinement
gentle control.)*

| param | cand0 | cand1 | **cand2 ★** | cand3 (control) |
|---|---|---|---|---|
| a_over (rim overhang) | 0.584 | 0.603 | **0.557** | 0.584 |
| g0 approach spacing (mm) | 29.5 | 28.1 | **27.8** | 29.1 |
| press depth (mm) | 3.4 | **3.9** | 3.3 | 2.7 |
| close_pow (close timing) | 0.40 early | 0.61 mid | **0.40 early** | 1.01 simultaneous |
| pinch gap (mm) | 1.7 | 1.7 | 1.8 | 1.9 |
| rise (mm) | 31.5 | 30.2 | **31.2** | 12.5 |
| panel: success / escapes | 6/24 · 1 | 9/24 · 0 | **12/24 · 0** | 2/24 · 0 |

In words:

- **cand2 (winner) is the *snug, patient* one** — fingers start closest
  together, presses nearest the washer's centre-side of the rim, pinches
  earliest, then takes the tallest, slowest ride up. Nothing about it is
  fast; it holds the washer before anything interesting can happen.
- **cand0 is cand2's twin, shifted outward** — same early pinch and tall
  rise, but it presses ~3 mm further out on the rim and approaches with
  wider-set fingers. That small extra leverage is enough to make it
  *flingier*: it lost the part once on the panel (the only escape among the
  four) and its floor is much worse (−2.09 vs 0.08).
- **cand1 is the *assertive* one** — presses ~0.6 mm deeper than the winner
  and grips later (close_pow 0.61): more energy into the pad, more washer
  motion before capture. Middle of the pack: more successes than cand0,
  rougher rolls than cand2, no escapes.
- **cand3 is the *timid* one** — the shallowest press and less than half the
  rise, closing while rising. It almost never finishes the roll (2/24) but
  also never breaks anything; its eye diagram is the tightest bundle at the
  *bottom* of the plot. cand2 is what cand3 became once it was given enough
  rise to finish the job.

One-line ranking: **assertiveness cand1 > cand0 > cand2 > cand3; safety
cand2 ≈ cand3 > cand1 > cand0; usefulness = cand2**, because it's the only
one that combines cand3's floor with a real success rate.

![final eye diagrams](../2026-07-04/lab_gaunt_final/eye_all.png)

**Full-env replay (nominal): a GENTLE failure — the reward's definition of
acceptable.** The winner did not capture in the full env (washer ends flat,
in place, clearance intact, zero violence, no escape) — and notably the
**touch sensor fired in the env for the first time** (every earlier phase
rode the timeout). In-lab 12/24 vs env-nominal miss re-confirms the
systematic lab↔env offset as **the** gating diagnostic (§Next) — but the
failure mode is now exactly what was asked for: when this policy misses, the
part stays put and the cell can simply try again.

[winner replay — tiles](../2026-07-04/render_final/pick.mp4) ·
[skin](../2026-07-04/render_final_skin/pick.mp4)

## Next

1. **Trace-diff the lab↔env offset** (one instrumented run each side, same θ)
   — likely a constant, fixable at the source.
2. Then the M3 integration: `soft_foundation` + this maneuver into
   `GraspSortEnv` proper, press-roll as a behavior-tree strategy, measured
   against the 2.4 % washer-flat baseline.

## Data & code

- Lab: `tools/pick_lab.py` (parallel mode + jitter machinery, this phase)
- Runs: `data/2026-07-04/lab_robust*/`
- Replay: `tools/render_pick.py` → `data/2026-07-04/render_robust*/`
