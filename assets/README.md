# assets/

## `ur10.usd` — UR10e arm (required, not committed)

The sim references `assets/ur10.usd` for the arm. It is **not** checked in: the
geometry derives from NVIDIA's URDF importer output and we keep this repo's
licensing clean. The **gripper** is built procedurally in `graspsort/gripper.py`
(original to this repo) and the **parts** in `graspsort/parts.py` — only the arm
body is external.

To produce a clean, redistributable `ur10.usd`:

1. Take Universal Robots' open UR10/UR10e description (BSD-licensed `ur_description`
   ROS package — `urdf/ur10.urdf` / xacro).
2. Import it to USD with Isaac Sim's URDF importer (GUI: *Isaac Utils → Workflows →
   URDF Importer*, or the `isaacsim.asset.importer.urdf` python API), with a **fixed
   base** and articulation root on `base_link`.
3. Save as `assets/ur10.usd`. Confirm the link named `wrist_3_link` exists — the
   gripper mounts onto it and `graspsort/kinematics.py` is calibrated to that chain.

The DH/joint constants in `graspsort/kinematics.py` already match the official UR10
URDF, so FK/IK line up with the imported arm.

> For private iteration the arm USD from the internal `rc-remote-platform`
> (`targets/ur10e/assets/virtual/imports/ur10/ur10.usd`) can be dropped in here;
> do **not** commit it to this public repo.

## `virtual/` — gripper-baked arm + Robotiq 2F-85 (optional, not committed)

The **robotiq gripper rung** (`GS_GRIPPER=robotiq`) needs the platform's
gripper-baked arm copy and the Robotiq asset, mirroring the platform layout so
the baked relative reference resolves:

    assets/virtual/imports/ur10/ur10.usd     <- targets/ur10e/assets/virtual/imports/ur10/
    assets/virtual/robotiq_2f85/**           <- targets/ur10e/assets/virtual/robotiq_2f85/

⚠ This rung is currently **blocked** in the standalone harness — the 2F-85
five-bar mimic linkage locks at ~1° here while the same USD closes fine in the
live platform gateway. Full diagnosis: `graspsort/robot.py` docstring and
`tools/probe_gripper.py` (probes: drive gains, tips, PhysicsScene parity, GPU
pipeline, self-collisions, solver iterations, gearing, both physics variants,
the platform's own scene). Default is the parametric jaw.

## `platform/` — platform scene mirror (optional, not committed)

`GS_SCENE=platform` opens the platform's slim ur10e site scene (the exact
composed robot + sort platform) instead of the in-code stage. Copy, preserving
relative references (drop the `.blend` sources):

    assets/platform/sites/ur10e/virtual/**      <- sites/ur10e/virtual/
    assets/platform/sites/workshop/virtual/**   <- sites/workshop/virtual/
    assets/platform/targets/ur10e/assets/**     <- targets/ur10e/assets/
