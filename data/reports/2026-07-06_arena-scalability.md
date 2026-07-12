# The parallel arena, rendered — and did rig count taint training?

*2026-07-06. Companion to [the Isaac Lab program](2026-07-04_isaaclab-program.html)
(Stage T/T2 results) and [the training technique doc](2026-07-04_training-technique.html).*

Two things happened in one investigation: David asked for a screenshot of the
many-rig training arena, and building it exposed a scalability question that
deserved a straight answer — **if the render scene shows reliability issues at
16 rigs, did the same apply while we were training?** (Spoiler: no — measured
below.)

## The arena

`tools/render_arena.py` — a grid of PickRigs (default 16, 0.15 m spacing)
running the Stage T2 champion θ in parallel inside the full-env boot, each
with a gauntlet-panel jitter, grid skins, arm hidden. This run: **12/16
successful picks, 6/16 full carries.**

![carry](../2026-07-06/arena/arena_hero_carry.png)

*Mid-carry: washers hanging from the grippers across the arena. A rolled-up
washer is a 2.5 mm edge-on disc — the corner camera keeps the discs facing
the lens; wide overhead angles hide them behind the finger pillars.*

![press](../2026-07-06/arena/arena_hero_press.png)

[full run video](../2026-07-06/arena/arena.mp4)

## What it took: the solver-budget cliff

The first arena run scored 0/16 with bizarre symptoms: B fingers followed
their carriers down but **stayed on the beds during the rise**, triggers never
fired. Per-rig telemetry pinned it: with ~4,500 joints in the env scene
(16 rigs × ~280 + the arm articulation), the B-spring prismatic joints
starve — the constraint holds when gravity helps and yields when it doesn't.

**Fix**: per-body solver budgets — `solverPositionIterationCount = 32` on the
sprung finger and washer. With it, triggers fire and the maneuver completes.
Two more render-craft rules landed in the tool: drive the fingers **every
physics step** in the env (the training-side dec-4 cadence loses the catch
there), and shoot from ~35° off the grip axis.

## The training-integrity A/B

Training runs in pick_lab's **own standalone World**, not the env scene — but
"24 rigs was fine" had never been isolated from jitter and slot effects. So:
same θ (Stage T cand0), same zero-jitter panel slot, same bed position,
**RIGS=1 vs RIGS=24**, traces on.

| | rig0 alone | rig0 with 23 neighbours |
|---|---|---|
| reward | 4.04 | 4.00 |
| success | yes | yes |
| lift | 31.0 mm | 28.9 mm |
| trajectory diff | — | **mean Δz 0.13 mm** (max 1.33 mm, mid-roll-up transient) |

**Verdict: rig count does not degrade the lab's physics — training was
clean.** The solver cliff is a property of the env scene (arm articulation +
scene solver config), which training never used; single-rig env replays were
always fine because one rig's ~300 joints don't strain it.

Two footnotes the same A/B re-confirmed:

1. The 24-rig run still failed 6/24 — at specific **bed positions**, including
   two zero-jitter rigs, while identically-jittered rigs elsewhere succeeded.
   That is the tile model's known slot-phase chaos (present throughout
   training; the reason selection uses worst-2-mean gauntlets), not solver
   load.
2. The env scene under-solving compliant joints plausibly contributes to the
   lab↔env B-spring timing lead found in the
   [trace-diff](../2026-07-05/charts/tracediff.png) — aligning scene solver
   settings (task open) may close the transfer gap and the env-side
   reliability cliff in one move.
