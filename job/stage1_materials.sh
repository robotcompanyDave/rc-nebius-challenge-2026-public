#!/usr/bin/env bash
# Stage 1 breadth: tune θ per sourceable material — all candidates as
# CONCURRENT Nebius jobs (each ~14 min, under the ~20-min infra-kill window).
# Harvests per-material MUθ + mu-panel score from the log windows.
set -u
export PATH="$HOME/.nebius/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT:-$REPO/data/runs/materials}"
mkdir -p "$OUT"
LOG="$OUT/stage1.log"
GRID="${GRID:-$REPO/configs/materials_grid.json}"
TUNED="0.47040,0.02877,0.00340,0.40000,0.00176,0.01063,0.00220,0.00221,-5.55072"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

mapfile -t MATS < <(python3 -c "
import json
for m in json.load(open('$GRID')):
    core = {k: m[k] for k in ('name','stiffness','damping','couple','couple_damp',
                              'falloff','cutoff','ell','travel','leveling','btip','mode')}
    print(m['name'] + '|' + json.dumps(core, separators=(',',':')))")

declare -A JOB
for entry in "${MATS[@]}"; do
  NAME="${entry%%|*}"; MJSON="${entry#*|}"
  J=$(nebius ai job create \
    --parent-id ${NEBIUS_PROJECT_ID:?set NEBIUS_PROJECT_ID} --platform gpu-h100-sxm \
    --preset 1gpu-16vcpu-200gb --restart-policy never --disk-size 300Gi \
    --shm-size 8Gi --timeout 1h --name "mat-${NAME//_/-}-$(date +%H%M)" \
    --image ${NEBIUS_REGISTRY:?set NEBIUS_REGISTRY}/rc-grasp-sort:roll \
    --container-command /isaac-sim/python.sh \
    --args /workspace/grasp-sort/tools/arena_eval.py \
    --env GS_AE_MODE=tune --env GS_AE_THETA="$TUNED" \
    --env GS_AE_MAT="$MJSON" \
    --env GS_AE_RIGS=6 --env GS_AE_COLS=3 --env GS_AE_ROUNDS=10 \
    --env GS_AE_SIG=0.10 --env GS_AE_SNAP_PEN=4.0 --env GS_AE_SEED=1 \
    --env GS_AE_JIT_XY=0.0015 --env GS_AE_ROLL=20 \
    --env GS_AE_MATK=0.85,1.18 --env GS_AE_MATC=0.9,1.12 --env GS_AE_MATCP=0.85,1.18 \
    --env GS_AE_OUT=/tmp/out \
    --async --format json 2>&1 | sed 's/\x1b\[[0-9;]*m//g' \
    | grep -oE 'aijob-[a-z0-9]+' | head -1)
  say "submitted $NAME -> ${J:-FAILED}"
  [ -n "$J" ] && JOB[$NAME]="$J"
done

t0=$(date +%s)
while :; do
  pending=0
  for NAME in "${!JOB[@]}"; do
    J="${JOB[$NAME]}"
    [ -f "$OUT/$NAME.done" ] && continue
    s=$(nebius ai job get "$J" --format json 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' \
        | python3 -c "import json,sys;print(json.load(sys.stdin).get('status',{}).get('state','?'))" 2>/dev/null)
    nebius ai job logs "$J" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' > "$OUT/$NAME.log"
    case "$s" in
      COMPLETED|ERROR|FAILED|CANCELLED)
        MU=$(grep -oE 'MUθ=\[[0-9eE+,.-]+\]' "$OUT/$NAME.log" | tail -1 | sed 's/MUθ=\[//; s/\]//')
        PANEL=$(grep -oE 'MU panel [0-9]+/[0-9]+' "$OUT/$NAME.log" | tail -1)
        SNAPS=$(grep -E 'mu-eval' "$OUT/$NAME.log" | grep -oE 'snaps=[0-9]+' | awk -F= '{s+=$2} END{print s+0}')
        echo "$NAME|$s|${PANEL:-n/a}|snaps=$SNAPS|$MU" >> "$OUT/results.txt"
        touch "$OUT/$NAME.done"
        say "$NAME $s ${PANEL:-n/a} snaps=$SNAPS"
        ;;
      *) pending=$((pending+1));;
    esac
  done
  [ "$pending" -eq 0 ] && break
  [ $(( $(date +%s)-t0 )) -gt 3600 ] && { say "1h watchdog"; break; }
  sleep 90
done
say "stage 1 done"
