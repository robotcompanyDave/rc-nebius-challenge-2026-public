# FEM deformable pad — proposal to re-evaluate (with POCs before commitment)

*Companion to [the Isaac Lab program](2026-07-04_isaaclab-program.html). Status:
PROPOSAL — nothing is committed to FEM until the POC gates below pass.*

## Why this was ruled out before (the record)

`spikes/soft-table/README.md`, decisions locked **2026-06-25**:

> *"a FEM deformable pad runs (deforms) but **rigid parts tunnel through it**;
> … PhysX soft bodies [have] **no static friction** (the part would creep on
> it), and rigid–deformable contact is the fragile/expensive path with limited
> contact-force readout. Revisit only if the pin-board result is promising
> **and** someone needs the continuous-foam answer."*

Both revisit-conditions are now met: the pin-board (→ SpringFoundation) result
IS promising — it carried M3b through M3e — and Gate M just demonstrated that
we **need the continuous-foam answer**: the discrete model's +5 mm see-saw
regime is chaotically bistable (±10 mm by scene slot), the shear layer needs
an α=4 solver-correction hack, square tips jam on tile walls without the
leveling workaround, and slot-nondeterminism is the same disease as our
lab→env transfer gap. These are all **discreteness artifacts**; a continuum
has none of them *by construction*. The question is whether the June
failure modes are fixable in the current stack.

## What has changed since June

1. **Two engine generations**: the June tests ran the pre-6.0 stack; we now
   have Isaac Sim 6.0.x + Isaac Lab 3.0 (with a maintained `DeformableObject`
   asset class, tutorials, and GPU-resident deformable state tensors).
2. **Our part is the worst case, and we now know its numbers.** A 2.5 mm-thin
   rigid washer against coarse tetrahedra is precisely the tunneling-prone
   configuration. Tunneling is governed by knowable knobs: simulation-mesh
   resolution vs. part thinness, collision/rest offsets, solver iterations /
   substeps, and stiffness. The POCs sweep exactly these.
3. **GPU-only is no longer a constraint** — the Lab program trains and
   evaluates in one GPU context anyway (which also dissolves our CPU↔GPU
   dynamics mismatch for this path).
4. The static-friction limitation may or may not persist in the current
   PhysX deformable model — **measured, not assumed**, in POC-3.

## What FEM would buy (if the POCs pass)

- No pop/swallow bistability → Gate M′ dwell becomes a smooth, optimizable
  objective (possibly even static Gate M revives).
- True lateral compliance → square-tip drags work with **no proximity-leveling
  workaround**, and no α solver-correction — the material parameters become
  the *real* E, ν of neoprene.
- One mesh replaces ~100 bodies + ~300 joints per env.

## POC ladder (kill-fast; each gate has a hard criterion)

| # | test | stack | pass gate | kills the path if |
|---|---|---|---|---|
| POC-1 **rest** | m12 washer resting 60 s on a 60×60×10 mm FEM pad; matrix: tet resolution {8,12,16 per edge} × Young's {0.1, 0.3, 1 MPa} × contact offset {1, 2 mm} | Isaac Sim 6.0 (our image) | penetration ≤ 0.5 mm, **zero fall-through** in all viable cells | washer tunnels at every viable resolution (the June failure reproduced) |
| POC-2 **press** | canonical A-press at 70 % overhang; rim-height trajectory + dwell in 4–6 mm band; repeat at 4 pad positions | Isaac Sim 6.0 | monotone rim response; position-invariant (< 1 mm spread) — i.e. **no bistability** | rim response is chaotic like the tiles (then discreteness wasn't the cause) |
| POC-3 **friction** | washer creep at rest (10 s, tilted pad 5°) + square-tip drag across the pad | Isaac Sim 6.0 | creep < 1 mm; drag smooth, no tunnel | static-friction hole persists → parts skate |
| POC-4 **Lab scale** | `DeformableObject` × 8 cloned envs, POC-1 repeated | Isaac Lab 3.0 (push-grasp:lab) | parity with POC-1; clean cloning | deformable cloning broken in beta → single-env only (path survives but scales worse) |
| POC-5 **render** | one POC-2 press rendered in the full env (skin-quality visuals free — the FEM mesh IS the skin) | render pipeline | visible smooth dish, video | — |

Budget: POC-1..3 ≈ one session; POC-4..5 ≈ a half session. **Decision point
after POC-3**: if rest+press+friction all pass, Gate M′ re-runs on the FEM pad
and the tile model becomes the fast-iteration/CPU-parity harness; any hard
fail → FEM stays descoped with fresh evidence, and Gate M′ continues on tiles
with the shortened-B fix.

## Relationship to the main line (no blocking)

Gate M′ (dwell-time) proceeds on the tile model NOW — the FEM POCs run beside
it, not instead of it. They meet at Stage T: whichever surface passes its
gates with the better dwell/catch statistics carries the technique training.

## POC-1 log (2026-07-05, live)

`tools/poc_fem_rest.py`, Isaac Sim 6.0, GPU pipeline. Five runs to a
confirmed pass — each failure diagnostic in itself:

| run | change | result |
|---|---|---|
| 1 | old June-era API (`add_physx_deformable_body`) | API gone in 6.0 — new path is `create_auto_volume_deformable_hierarchy` + `add_deformable_material` (which now HAS a first-class `static_friction` parameter — June's "no static friction" may be fixed upstream) |
| 2 | new API, material bound to **collision** mesh | all 9 cells identical, washer through to the ground. PhysX warns: *materials on collision/graphics meshes are ignored* — the pad ran on the DEFAULT material with no contact. **Likely a component of the June failure.** |
| 3 | material on **sim** mesh | cells now differ (E and resolution bite) but the pad floats **+20 mm** above the ground on its default contact-offset cushion, washer tunnels past it |
| 4 | explicit offsets on the pad collision mesh (contact 2 mm / rest 0) | **first PASS**: res 8, E 0.3 MPa — washer rests at +0.9 mm, penetration 0.31 mm, pad top stable, 3 s hold |
| 5 | 10 s hold, E ∈ {0.2, 0.3, 0.5} at res 8 | **PASS ×2 confirmed**: E 0.2 and 0.3 MPa rest at ~+0.9 mm for the full 10 s with **zero creep** (E 0.5 marginal at pen 0.70 mm, slight wobble) |

The run-4 matrix, and what it teaches:

| | E 0.1 MPa | E 0.3 MPa | E 1.0 MPa |
|---|---|---|---|
| res 8 | creeps in (−4 mm @ 3 s) | **PASS (pen 0.31 mm)** | contact jitter (±4 mm bounce) |
| res 12 | fell | fell | fell |
| res 16 | fell | fell | fell |

1. **Contact offset is the entire game for thin parts.** June's fall-through
   is fully explained by (material-on-wrong-prim) + (default offsets), not by
   a fundamental engine limit.
2. **Collider TYPE matters vs deformables**: the 10 mm control cube tunnels in
   every cell *even with explicit 4 mm offsets* (run 5) — it uses a
   **primitive** cube collider. The washer's **convex-hull mesh** collider
   contacts fine. Working hypothesis: rigid↔deformable contact requires
   mesh-based colliders. Carried to POC-2: the press finger must be a hull
   mesh, not a primitive capsule/box.
3. **The PASS band sits exactly in the physically-relevant regime** —
   0.2–0.3 MPa is foam-neoprene territory, precisely our target material.
4. **Open risk carried to POC-2**: higher sim-mesh resolution (12/16) still
   tunnels — the coarse mesh is currently load-bearing. Needs collision-mesh
   (not sim-mesh) refinement and/or solver-iteration sweep before the press
   test can trust fine meshes.

**POC-1 verdict: gate OPEN** — penetration ≤ 0.35 mm, zero fall-through, zero
10 s creep in the target-E band. The flat-rest half of POC-3 is already
green as a side effect. Next: POC-2 (canonical press at 4 pad positions,
bistability check).

## POC-2 log (2026-07-05, live)

`tools/poc_fem_press.py` — canonical press (p3, ao 0.70, dwell-winner
timing), E {0.2, 0.3} × 4 sub-cell washer positions. Four runs:

| run | change | result |
|---|---|---|
| 1 | POC-1 recipe as-is (8-vertex box source, hex res 8) | violent: rim pops +22…+44 mm at press bottom-out, hold rings ±12–16 mm, one fall-through |
| 2 | tighter washer offsets, 480 Hz, 32 solver iters | WORSE (−96 mm fling; tightening offsets re-enabled tunneling — the thin washer needs its 4 mm cushion) |
| 3 | finger friction like the tile rig (A 0.10 / B 1.20) | still pops (10–64 mm), all cells end buried |
| 4 | **tessellated source mesh** (16×16×3 surface grid → ~5 mm collision cells) + hex res 16 | **PASS — all 8 cells: hold std 0.00 mm, position spread 0.07 mm** (gate < 1 mm) |

![poc2 press](../2026-07-05/charts/poc2_press.png)

The run-1..3 violence was never material physics: an 8-vertex box source
cooks into giant collision tets regardless of the hex sim resolution, so the
washer sat on 1–2 vertices — we had accidentally rebuilt DISCRETENESS with
~10 mm cells, coarser than the 5 mm tiles. **The source-mesh tessellation is
the real collision-resolution knob**, and it also retroactively explains
POC-1's "res 12/16 tunnel" oddity (sim res changed, collision stayed coarse).

With it fixed, the contrast with the tile model could not be sharper:

| | spring tiles (gate4) | FEM pad (POC-2 v4) |
|---|---|---|
| hold noise | ±4–16 mm ring / chaotic branches | **0.00 mm** |
| position sensitivity | 12–16 mm by slot, layout-global | **0.07 mm** |
| trajectory | pop / swallow / eject bistability | monotone ease-in, flat hold |

Note the canonical p3/ao.70 press on this pad presses the far rim DOWN
(−2.9 mm, no see-saw pop) — the +5 mm presentation regime hasn't been found
YET; that's precisely the Gate M′ material/press co-optimization, and on the
FEM pad it is now a **smooth, deterministic objective** instead of a
coin-flip. E 0.2 vs 0.3 differ by only 0.07 mm here — geometry dominates at
this press depth; the optimizer will need the depth/overhang axes.

**POC-2 verdict: gate PASSED** (monotone, position-invariant, zero violence).
Working recipe: tessellated watertight source grid (~5 mm), hex sim res 16,
material on sim mesh, pad col offsets 2/0 mm, parts keep their 4 mm contact
offset, fingers = convex-hull meshes with real friction materials.
Next: POC-3 tilted-creep + drag (flat-rest creep already 0 mm/10 s), then
Gate M′ dwell sweep ON the FEM pad, POC-4 Lab cloning.

## POC-2b — penetration under load (2026-07-05, David's catch)

David spotted in the render video that the washer sinks INTO the surface
during the press. Quantified (`tools/poc_fem_pen.py`, washer probes vs the
local deformed surface) and swept every legitimate knob:

| knob | press-pen (mm) | verdict |
|---|---|---|
| baseline (div32, E0.2, ν0.3, 240 Hz) | 4.7–4.9 | — |
| mesh 5.0→2.5 mm cells | 5.0→2.8→4.9* | weak, noisy |
| E 0.2→0.5 MPa | 4.89→4.89 | **no effect** |
| ν 0.3→0.45 | no change | no effect |
| 240→480 Hz | no change | no effect |
| washer solver iters 32/64 + depen-vel 10 | 4.98 | no effect |
| pad density 300→1200 | 4.97 | no effect |
| thin pad 4–5 mm (bottom-out fulcrum idea) | 6.5–7.5 (punch-through!) | worse |
| washer mass 5 g→20 g | 4.4 | marginal |
| washer mass 5 g→**100 g** | **1.8** | works — but non-physical |

Control: the bare finger pressing the pad = 0.02 mm penetration, clean dish.
The failure is specific to the **light rigid part under load** — contact
holds at rest (gravity ~0.05 N) and collapses under press forces,
**independent of material stiffness**. Only 20× fake mass restores it.

![sections](../2026-07-05/charts/fem_pen_sections.png)

The cross-sections add the second, deeper finding: even where contact
roughly holds, the linear-elastic continuum forms a **wide ±20 mm bowl**, not
a local dimple. The washer slides into the bowl and the far rim NEVER rises
above rest height in any canonical-press config (`far_rim_max` = rest value
everywhere). The tile model's see-saw came from a LOCAL dent with a sharp
shoulder — which is what real foam does too, because real foam localizes
via densification (a nonlinearity PhysX linear FEM does not model). PhysX
FEM is the right tool for a solid rubber sheet; the workshop mat behaves
foam-like.

## VERDICT (2026-07-05): FEM re-descoped, with precise evidence

1. **June's fall-through**: fixed (config, not engine) — POC-1 stands.
2. **NEW kill #1**: rigid↔deformable contact cannot support a ~5 g thin part
   under press loads (E/ν/dt/iterations/density-independent; only fake mass
   helps). One-file repro: `tools/poc_fem_pen.py`.
3. **NEW kill #2 (mechanism)**: linear-elastic continuum ⇒ wide bowls, no
   local fulcrum, no rim presentation — the flip technique premise breaks
   even if kill #1 were fixed.
4. The **pin-grid (Winkler/Pasternak) model is retroactively validated**: its
   locality knobs (k, couple, falloff, travel limit) are exactly the foam
   nonlinearity FEM lacks. Its remaining disease (slot bistability) is a
   training-robustness problem, not a modelling one.

Revisit triggers: PhysX ships hyperelastic/densifying volume materials, or a
release note touches rigid–deformable contact for light bodies (retest =
one command), or the target surface becomes solid rubber rather than foam.

One transferable observation from the FEM runs: the slow deep press BESIDE
the washer levered the far rim to +8 mm (see-saw via pad, not part). Worth
porting to the tiles as a candidate presentation primitive.
