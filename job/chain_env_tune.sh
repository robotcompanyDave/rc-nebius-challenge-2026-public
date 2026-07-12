#!/usr/bin/env bash
# Chained env-in-the-loop tuning of the washer pick on Nebius.
#
# Each link = one short H100 job running arena_eval.py in TUNE mode (6 rigs,
# 10 CEM rounds + a 2-round deterministic MU panel), kept under the ~96
# rig-round-per-process leak ceiling. The CEM mean (MUθ) from each job seeds
# the next, so the chain gives unlimited optimization depth at ~$1/job.
# Stops early when the MU panel hits the target.
set -uo pipefail
export HOME="${HOME:-/tmp}"
export PATH="$HOME/.nebius/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

CHAINS="${CHAINS:-4}"
TARGET_NUM="${TARGET_NUM:-11}"          # stop when MU panel >= TARGET_NUM/12
SEED_THETA="${SEED_THETA:?set SEED_THETA}"
PARENT="${NEBIUS_PROJECT_ID:?set NEBIUS_PROJECT_ID}"
IMAGE="${NEBIUS_REGISTRY:?set NEBIUS_REGISTRY}/rc-grasp-sort:roll"
OUT="${OUT:-$HOME/RC/rc-spike-nebius-basic/data/$(date +%F)/envtune}"
mkdir -p "$OUT"
LOG="$OUT/chain.log"
SIGS=(0.08 0.06 0.05 0.04 0.04 0.04)

say(){ echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }
jstate(){ nebius ai job get "$1" --format json 2>/dev/null \
  | sed 's/\x1b\[[0-9;]*m//g' \
  | python3 -c "import json,sys;print(json.load(sys.stdin).get('status',{}).get('state','?'))" 2>/dev/null; }

seed="$SEED_THETA"
say "chain start: $CHAINS links, target MU panel >= $TARGET_NUM/12"
say "seed θ: $seed"

for i in $(seq 1 "$CHAINS"); do
  SIG="${SIGS[$((i-1))]}"
  NAME="washer-envtune-c${i}-$(date +%H%M)"
  say "── link $i/$CHAINS: sig=$SIG name=$NAME"
  CREATE=$(nebius ai job create \
    --parent-id "$PARENT" --name "$NAME" \
    --platform gpu-h100-sxm --preset 1gpu-16vcpu-200gb \
    --restart-policy never --timeout 1h30m --disk-size 300Gi --shm-size 8Gi \
    --env GS_AE_MODE=tune --env GS_AE_RIGS=6 --env GS_AE_COLS=3 \
    --env GS_AE_ROUNDS=10 --env GS_AE_SIG="$SIG" --env GS_AE_SEED="$i" \
    --env GS_AE_JIT_XY=0.0015 --env GS_AE_ROLL=20 \
    --env GS_AE_THETA="$seed" --env GS_AE_OUT=/tmp/out \
    --image "$IMAGE" \
    --container-command /isaac-sim/python.sh \
    --args /workspace/grasp-sort/tools/arena_eval.py \
    --async --format json 2>&1 | sed 's/\x1b\[[0-9;]*m//g')
  J=$(echo "$CREATE" | grep -oE 'aijob-[a-z0-9]+' | head -1)
  if [ -z "$J" ]; then say "  submit FAILED: $CREATE"; break; fi
  say "  job=$J — waiting"
  t0=$(date +%s)
  while :; do
    s=$(jstate "$J")
    case "$s" in COMPLETED|ERROR|FAILED|CANCELLED) break;; esac
    if [ $(( $(date +%s)-t0 )) -gt 4200 ]; then
      say "  70min watchdog — cancelling $J"
      nebius ai job cancel "$J" >/dev/null 2>&1; s="TIMEOUT"; break
    fi
    sleep 60
  done
  nebius ai job logs "$J" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' > "$OUT/$J.log"
  rounds=$(grep -cE '\[ae\] tune rnd[0-9]+' "$OUT/$J.log" || true)
  MU=$(grep -oE 'MUθ=\[[0-9eE+,.-]+\]' "$OUT/$J.log" | tail -1 | sed 's/MUθ=\[//; s/\]//')
  BEST=$(grep -oE 'BESTθ r=[0-9.-]+ \[[0-9eE+,.-]+\]' "$OUT/$J.log" | tail -1 \
         | grep -oE '\[[0-9eE+,.-]+\]' | tr -d '[]')
  PANEL=$(grep -oE 'MU panel [0-9]+/[0-9]+' "$OUT/$J.log" | tail -1 | grep -oE '[0-9]+/[0-9]+')
  say "  state=$s tune-rounds=$rounds panel=${PANEL:-n/a}"
  if [ -n "$MU" ]; then
    seed="$MU"; say "  next seed = MUθ [$MU]"
  elif [ -n "$BEST" ]; then
    seed="$BEST"; say "  no MUθ; next seed = BESTθ [$BEST]"
  else
    say "  no θ recovered — keeping previous seed"
  fi
  if [ -n "$PANEL" ]; then
    num=${PANEL%%/*}
    if [ "$num" -ge "$TARGET_NUM" ]; then
      say "TARGET HIT: MU panel $PANEL at link $i — stopping chain"
      break
    fi
  fi
done
echo "$seed" > "$OUT/final_theta.txt"
say "chain done. final θ: $seed"
say "final θ written to $OUT/final_theta.txt"
