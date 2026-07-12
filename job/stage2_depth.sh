#!/usr/bin/env bash
# Stage 2 depth: 2 chained tune waves for the 4 surviving materials, each link
# warm-started from the previous MUθ. Links run CONCURRENTLY across materials.
set -u
export PATH="$HOME/.nebius/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT:-$REPO/data/runs/materials}"
LOG="$OUT/stage2.log"
GRID="${GRID:-$REPO/configs/materials_grid.json}"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

FINALISTS="PU_soft_10 SIL_gel_15 EVA_firm_10 NEO_sponge_12"

matjson(){ python3 -c "
import json
for m in json.load(open('$GRID')):
    if m['name'] == '$1':
        core = {k: m[k] for k in ('name','stiffness','damping','couple','couple_damp',
                                  'falloff','cutoff','ell','travel','leveling','btip','mode')}
        print(json.dumps(core, separators=(',',':')))"; }

seed_of(){ # latest MUθ for material from stage1/stage2 logs
  cat "$OUT/$1.log" "$OUT/$1.d"*.log 2>/dev/null \
    | grep -oE 'MUθ=\[[0-9eE+,.-]+\]' | tail -1 | sed 's/MUθ=\[//; s/\]//'; }

for WAVE in 1 2; do
  declare -A JOB=()
  for NAME in $FINALISTS; do
    SEED_TH=$(seed_of "$NAME")
    [ -z "$SEED_TH" ] && { say "no seed for $NAME, skip"; continue; }
    MJSON=$(matjson "$NAME")
    J=$(nebius ai job create \
      --parent-id ${NEBIUS_PROJECT_ID:?set NEBIUS_PROJECT_ID} --platform gpu-h100-sxm \
      --preset 1gpu-16vcpu-200gb --restart-policy never --disk-size 300Gi \
      --shm-size 8Gi --timeout 1h --name "matd${WAVE}-${NAME//_/-}-$(date +%H%M)" \
      --image ${NEBIUS_REGISTRY:?set NEBIUS_REGISTRY}/rc-grasp-sort:roll \
      --container-command /isaac-sim/python.sh \
      --args /workspace/grasp-sort/tools/arena_eval.py \
      --env GS_AE_MODE=tune --env GS_AE_THETA="$SEED_TH" \
      --env GS_AE_MAT="$MJSON" \
      --env GS_AE_RIGS=6 --env GS_AE_COLS=3 --env GS_AE_ROUNDS=10 \
      --env GS_AE_SIG=0.07 --env GS_AE_SNAP_PEN=4.0 --env GS_AE_SEED=$((WAVE+1)) \
      --env GS_AE_JIT_XY=0.0015 --env GS_AE_ROLL=20 \
      --env GS_AE_MATK=0.85,1.18 --env GS_AE_MATC=0.9,1.12 --env GS_AE_MATCP=0.85,1.18 \
      --env GS_AE_OUT=/tmp/out \
      --async --format json 2>&1 | sed 's/\x1b\[[0-9;]*m//g' \
      | grep -oE 'aijob-[a-z0-9]+' | head -1)
    say "wave$WAVE $NAME -> ${J:-FAILED}"
    [ -n "$J" ] && JOB[$NAME]="$J"
  done
  t0=$(date +%s)
  while :; do
    pending=0
    for NAME in "${!JOB[@]}"; do
      J="${JOB[$NAME]}"
      [ -f "$OUT/$NAME.d$WAVE.done" ] && continue
      s=$(nebius ai job get "$J" --format json 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' \
          | python3 -c "import json,sys;print(json.load(sys.stdin).get('status',{}).get('state','?'))" 2>/dev/null)
      nebius ai job logs "$J" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' > "$OUT/$NAME.d$WAVE.log"
      case "$s" in
        COMPLETED|ERROR|FAILED|CANCELLED)
          PANEL=$(grep -oE 'MU panel [0-9]+/[0-9]+' "$OUT/$NAME.d$WAVE.log" | tail -1)
          SNAPS=$(grep -E 'mu-eval' "$OUT/$NAME.d$WAVE.log" | grep -oE 'snaps=[0-9]+' | awk -F= '{s+=$2} END{print s+0}')
          echo "$NAME|wave$WAVE|$s|${PANEL:-n/a}|snaps=$SNAPS|$(seed_of "$NAME")" >> "$OUT/results2.txt"
          touch "$OUT/$NAME.d$WAVE.done"
          say "wave$WAVE $NAME $s ${PANEL:-n/a} snaps=$SNAPS"
          ;;
        *) pending=$((pending+1));;
      esac
    done
    [ "$pending" -eq 0 ] && break
    [ $(( $(date +%s)-t0 )) -gt 2700 ] && { say "wave$WAVE watchdog"; break; }
    sleep 90
  done
done
say "stage 2 depth done"
