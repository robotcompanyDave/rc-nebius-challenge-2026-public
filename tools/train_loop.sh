#!/usr/bin/env bash
# In-job restart loop for resid_policy training: re-runs the trainer in fresh
# python processes (each capped at GS_RP2_ROUNDS rounds) to sidestep the
# ~96-rig-round-per-process memory leak. ES state resumes from GS_RP2_STATE,
# which lives in the container FS and survives process restarts within a job.
# Invoked as the single --args token of a Nebius job:
#   --container-command /bin/bash --args /workspace/grasp-sort/tools/train_loop.sh
set -u
N="${RP2_LOOPS:-40}"
for i in $(seq 1 "$N"); do
  echo "[loop] === process $i/$N $(date -u +%H:%M:%S) ==="
  /isaac-sim/python.sh /workspace/grasp-sort/tools/resid_policy.py
  rc=$?
  echo "[loop] process $i exited rc=$rc"
  [ "$rc" -ne 0 ] && sleep 5
done
echo "[loop] done"
