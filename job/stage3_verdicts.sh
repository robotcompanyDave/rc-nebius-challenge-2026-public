#!/usr/bin/env bash
# Stage 3: large-n final verdicts. 5 candidate (material × θ) combos, each
# evaluated on 216 identical episodes (3 chunk-jobs × 12 rounds × 6 rigs),
# all 15 jobs CONCURRENT. Material noise + placement + roll always on.
set -u
export PATH="$HOME/.nebius/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
OUT=~/RC/rc-spike-nebius-basic/data/2026-07-11/materials
LOG="$OUT/stage3.log"
GRID=~/RC/rc-spike-soft-surface/spikes/push-grasp/configs/materials_grid.json
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

matjson(){ python3 -c "
import json
for m in json.load(open('$GRID')):
    if m['name'] == '$1':
        core = {k: m[k] for k in ('name','stiffness','damping','couple','couple_damp',
                                  'falloff','cutoff','ell','travel','leveling','btip','mode')}
        print(json.dumps(core, separators=(',',':')))"; }

# candidate|material|theta
CANDS=$(cat <<'EOF'
PUsoft10_s1|PU_soft_10|0.43924,0.02833,0.00380,0.40000,0.00189,0.01270,0.00191,0.00240,0.97188,3.91437
NEO12_w2|NEO_sponge_12|0.46138,0.02594,0.00426,1.82816,0.00206,0.02470,0.00210,0.00236,-10.39887,-13.36207
EVA10_w2|EVA_firm_10|0.53949,0.02786,0.00397,1.69561,0.00171,0.02308,0.00228,0.00228,-15.06188,-3.09154
SILgel_w1|SIL_gel_15|0.46827,0.02995,0.00436,0.67040,0.00162,0.01661,0.00155,0.00174,-15.30875,-11.64977
INCUMBENT|PU_med_12|0.47040,0.02877,0.00340,0.40000,0.00176,0.01063,0.00220,0.00221,-5.55072,0.0
EOF
)

declare -A JOB
while IFS='|' read -r CID MATN TH; do
  MJSON=$(matjson "$MATN")
  for CH in 801 802 803; do
    J=$(nebius ai job create \
      --parent-id ${NEBIUS_PROJECT_ID:?set NEBIUS_PROJECT_ID} --platform gpu-h100-sxm \
      --preset 1gpu-16vcpu-200gb --restart-policy never --disk-size 300Gi \
      --shm-size 8Gi --timeout 1h --name "v-${CID//_/-}-$CH-$(date +%H%M)" \
      --image ${NEBIUS_REGISTRY:?set NEBIUS_REGISTRY}/rc-grasp-sort:roll \
      --container-command /isaac-sim/python.sh \
      --args /workspace/grasp-sort/tools/arena_eval.py \
      --env GS_AE_MODE=eval --env GS_AE_THETA="$TH" --env GS_AE_MAT="$MJSON" \
      --env GS_AE_RIGS=6 --env GS_AE_COLS=3 --env GS_AE_ROUNDS=12 \
      --env GS_AE_SEED=$CH --env GS_AE_JIT_XY=0.0015 --env GS_AE_ROLL=20 \
      --env GS_AE_MATK=0.85,1.18 --env GS_AE_MATC=0.9,1.12 --env GS_AE_MATCP=0.85,1.18 \
      --env GS_AE_OUT=/tmp/out \
      --async --format json 2>&1 | sed 's/\x1b\[[0-9;]*m//g' \
      | grep -oE 'aijob-[a-z0-9]+' | head -1)
    say "submitted $CID chunk$CH -> ${J:-FAILED}"
    [ -n "$J" ] && JOB["$CID.$CH"]="$J"
  done
done <<< "$CANDS"

t0=$(date +%s)
while :; do
  pending=0
  for KEY in "${!JOB[@]}"; do
    J="${JOB[$KEY]}"
    [ -f "$OUT/v_$KEY.done" ] && continue
    s=$(nebius ai job get "$J" --format json 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' \
        | python3 -c "import json,sys;print(json.load(sys.stdin).get('status',{}).get('state','?'))" 2>/dev/null)
    case "$s" in
      COMPLETED|ERROR|FAILED|CANCELLED)
        nebius ai job logs "$J" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' > "$OUT/v_$KEY.log"
        RES=$(grep -oE 'EVAL [0-9]+/[0-9]+ success, snaps=[0-9]+' "$OUT/v_$KEY.log" | tail -1)
        echo "$KEY|$s|${RES:-n/a}" >> "$OUT/verdicts.txt"
        touch "$OUT/v_$KEY.done"
        say "$KEY $s ${RES:-n/a}"
        ;;
      *) pending=$((pending+1));;
    esac
  done
  [ "$pending" -eq 0 ] && break
  [ $(( $(date +%s)-t0 )) -gt 3000 ] && { say "watchdog"; break; }
  sleep 90
done
say "stage 3 done"
python3 - "$OUT/verdicts.txt" <<'PY'
import re, sys, math
from collections import defaultdict
agg = defaultdict(lambda: [0, 0, 0])
for line in open(sys.argv[1]):
    m = re.match(r'(\w+)\.\d+\|\w+\|EVAL (\d+)/(\d+) success, snaps=(\d+)', line.strip())
    if m:
        a = agg[m.group(1)]
        a[0] += int(m.group(2)); a[1] += int(m.group(3)); a[2] += int(m.group(4))
print("\n===== STAGE 3 VERDICTS =====")
for cid, (su, n, sn) in sorted(agg.items(), key=lambda kv: -(kv[1][0] / max(kv[1][1], 1))):
    p = su / max(n, 1); se = math.sqrt(p * (1 - p) / max(n, 1))
    print(f"{cid:12} {su}/{n} = {100*p:.1f}% ±{100*1.96*se:.1f}  snaps={sn}")
PY