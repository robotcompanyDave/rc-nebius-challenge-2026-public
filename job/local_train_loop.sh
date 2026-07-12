#!/usr/bin/env bash
# Free local ES seed on the 5080: fresh docker container per process
# (leak-proof), state persists on disk, logs appended. Seed 7.
set -u
SPIKE="$HOME/RC/rc-spike-soft-surface/spikes/push-grasp"
cd "$SPIKE" || exit 1
TUNED="0.47040,0.02877,0.00340,0.40000,0.00176,0.01063,0.00220,0.00221,-5.55072"
mkdir -p data/rp2_local
N="${1:-45}"
for i in $(seq 1 "$N"); do
  echo "[loop-local] === process $i/$N $(date +%H:%M:%S) ==="
  GS_RP2_MODE=train GS_RP2_THETA="$TUNED" GS_RP2_RIGS=6 GS_RP2_COLS=3 \
  GS_RP2_ROUNDS=13 GS_RP2_POP=24 GS_RP2_SEED=7 GS_RP2_LR=0.03 GS_RP2_SIG=0.05 \
  GS_RP2_DRK=0.5,2.0 GS_RP2_DRC=0.7,1.4 GS_RP2_DRCP=0.5,1.8 GS_RP2_DRB=0.7,1.4 \
  GS_RP2_JIT_XY=0.0025 GS_RP2_JIT_DZ=0.001 GS_RP2_NOISE=0.0001 \
  GS_RP2_OUT=data/rp2_local GS_RP2_STATE=data/rp2_local/es_state.npz \
  docker/run.sh tools/resid_policy.py 2>&1 | grep -E '^\[rp2\]'
done
echo "[loop-local] done"
