#!/usr/bin/env bash
# Auto-resume ES training across Nebius infra kills: harvest the latest WSAVE
# weights from each dead job's log window, relaunch resuming from them.
# Caps: MAX_JOBS relaunches, MAX_GPU_S cumulative seconds, TARGET_UPD updates.
set -u
export PATH="$HOME/.nebius/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT:-$REPO/data/runs/resid}"
LOG="${RLOG:-$OUT/relaunch.log}"
RAW="${RAW:-$OUT/chainA.progress.raw}"
INITW="${INITW:-$OUT/initw_next.npz}"
TUNED="0.47040,0.02877,0.00340,0.40000,0.00176,0.01063,0.00220,0.00221,-5.55072"
MAX_JOBS="${MAX_JOBS:-6}"
MAX_GPU_S="${MAX_GPU_S:-14400}"
TARGET_UPD="${TARGET_UPD:-60}"
mkdir -p "$OUT"; touch "$RAW"
say(){ echo "[$(date '+%H:%M:%S')] $*" >> "$LOG"; }

harvest(){ # $1 = job log window file -> update RAW + INITW; echo latest upd
  grep -E '\[rp2\] (UPDATE|CENTER|WSAVE|init weights)' "$1" >> "$RAW" 2>/dev/null
  sort -u "$RAW" -o "$RAW"
  python3 - "$RAW" "$INITW" <<'PY'
import re, sys, base64, numpy as np
raw, out = sys.argv[1], sys.argv[2]
m = re.findall(r'WSAVE upd(\d+) ([A-Za-z0-9+/=]+)', open(raw).read())
if not m:
    print(-1); raise SystemExit
m.sort(key=lambda x: int(x[0]))
upd, b64 = m[-1]
w = np.frombuffer(base64.b64decode(b64), dtype=np.float32).astype(np.float64)
if w.size != 1603:
    print(-1); raise SystemExit
np.savez(out, w=w, upd=int(upd))
print(upd)
PY
}

total_s=0
for i in $(seq 1 "$MAX_JOBS"); do
  upd=$(harvest /dev/null)
  say "── relaunch $i/$MAX_JOBS (resume upd $upd, gpu_used ${total_s}s)"
  [ "$upd" -ge "$TARGET_UPD" ] && { say "target updates reached"; break; }
  [ "$total_s" -ge "$MAX_GPU_S" ] && { say "gpu budget reached"; break; }
  J=$(nebius ai job create \
    --parent-id ${NEBIUS_PROJECT_ID:?set NEBIUS_PROJECT_ID} --platform gpu-h100-sxm \
    --preset 1gpu-16vcpu-200gb --restart-policy never --disk-size 300Gi \
    --shm-size 8Gi --timeout 3h \
    --image ${NEBIUS_REGISTRY:?set NEBIUS_REGISTRY}/rc-grasp-sort:roll \
    --container-command /bin/bash --args /workspace/grasp-sort/tools/train_loop.sh \
    --inject-file "$INITW:/workspace/initw.npz" \
    --name resid-${CHAIN:-chainA}-$i-$(date +%H%M) \
    --env GS_RP2_MODE=train --env GS_RP2_THETA="$TUNED" \
    --env GS_RP2_RIGS=6 --env GS_RP2_COLS=3 --env GS_RP2_ROUNDS=13 \
    --env GS_RP2_POP=24 --env GS_RP2_SEED=0 --env GS_RP2_LR="${LR:-0.03}" --env GS_RP2_SNAP_PEN="${SNAP_PEN:-4.0}" \
    --env GS_RP2_SIG=0.05 \
    --env GS_RP2_DRK=0.5,2.0 --env GS_RP2_DRC=0.7,1.4 \
    --env GS_RP2_DRCP=0.5,1.8 --env GS_RP2_DRB=0.7,1.4 \
    --env GS_RP2_JIT_XY=0.0025 --env GS_RP2_JIT_DZ=0.001 \
    --env GS_RP2_NOISE=0.0001 \
    --env GS_RP2_OUT=/tmp/out --env GS_RP2_STATE=/tmp/out/es_state.npz \
    --env GS_RP2_INIT_W=/workspace/initw.npz --env RP2_LOOPS=40 \
    --async --format json 2>&1 | sed 's/\x1b\[[0-9;]*m//g' \
    | grep -oE 'aijob-[a-z0-9]+' | head -1)
  [ -z "$J" ] && { say "submit failed"; sleep 120; continue; }
  say "  job=$J"
  t0=$(date +%s); started=""
  while :; do
    s=$(nebius ai job get "$J" --format json 2>/dev/null \
        | sed 's/\x1b\[[0-9;]*m//g' \
        | python3 -c "import json,sys;print(json.load(sys.stdin).get('status',{}).get('state','?'))" 2>/dev/null)
    [ "$s" = "RUNNING" ] && [ -z "$started" ] && started=$(date +%s)
    nebius ai job logs "$J" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' > "$OUT/$J.win"
    case "$s" in COMPLETED|ERROR|FAILED|CANCELLED)
      upd2=$(harvest "$OUT/$J.win")
      [ -n "$started" ] && total_s=$(( total_s + $(date +%s) - started ))
      say "  terminal $s at upd $upd2"
      break;;
    esac
    [ $(( $(date +%s) - t0 )) -gt 11400 ] && { say "  3.2h watchdog"; nebius ai job cancel "$J" >/dev/null 2>&1; break; }
    sleep 90
  done
done
upd=$(harvest /dev/null)
say "chain done: final upd $upd, gpu ${total_s}s"
