"""
graspsort — a slim, headless Isaac Sim environment for grasp + sort data
generation and evaluation, packaged to run to completion as a Nebius Serverless
AI Job.

Scene: a UR10e arm + a parametric parallel-jaw gripper + procedurally generated
M12 nut/bolt parts on a flat work surface. No dependency on the NVIDIA Factory
asset pack and no live teleop gateway — everything is built in-code so the whole
environment is reproducible from this repo alone.

Import note: modules that touch Isaac/USD (`sim_env`, `robot`, `gripper`,
`parts`, `observe`) must only be imported AFTER a `SimulationApp` has been
constructed (the Omniverse python bindings fail to load otherwise). `kinematics`
and `logging_schema` are pure-python and safe to import anywhere.
"""

__version__ = "0.1.0"
