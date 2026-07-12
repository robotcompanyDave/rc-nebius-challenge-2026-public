#!/usr/bin/env bash
# ============================================================================
# Isaac Lab on Nebius Serverless AI Jobs — short "make something work" test.
#
# Runs a SHORT, HEADLESS Isaac Lab RL training (Cartpole, PhysX GPU, no
# rendering) as a Nebius AI Job, to demonstrate a robot-simulation batch job
# on Nebius for the Serverless AI Builders Challenge.
#
REPO="$(cd "$(dirname "$0")/.." && pwd)"
# Capacity has been jammed at peak (L40S + H100 on-demand both failed/stalled
# on 2026-07-05), so this is meant to run OFF-PEAK (~04:30 CEST) and it tries
# several GPU / allocation combos until one actually provisions and runs.
#
# Self-contained for cron: sets PATH/HOME, uses the non-interactive `sa-sim`
# service-account profile (no browser auth). Writes everything under
# data/<date>/lab/ and regenerates the HTML report at the end.
# ============================================================================
set -uo pipefail

# --- environment (cron has a minimal env) -----------------------------------
export HOME="${HOME:-/tmp}"
export PATH="$HOME/.nebius/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export NEBIUS_PROFILE="sa-sim"
PROFILE_ARG="--profile sa-sim"

SPIKE="${SPIKE:-$REPO}"
PARENT="${NEBIUS_PROJECT_ID:?set NEBIUS_PROJECT_ID}"                       # eu-north1 (image lives here)
IMAGE="${NEBIUS_REGISTRY:?set NEBIUS_REGISTRY}/isaac-lab-test:v3"   # v3 = clean ENTRYPOINT + refreshed rsl_rl

DATE="$(date +%Y-%m-%d)"; STAMP="$(date +%H%M)"
OUT="$SPIKE/data/$DATE/lab"
mkdir -p "$OUT"
RUNLOG="$OUT/runner.log"

# short, headless, GPU-physics training — the actual Isaac Lab workload
# Isaac Lab v3 train invocation. NOTE Nebius arg quirks (learned the hard way):
#  • --args is a SINGLE space-split string, NOT repeatable (repeat = last wins).
#  • --container-command REPLACES the image ENTRYPOINT (v3 has a clean/empty one).
TRAIN='train --rl_library rsl_rl --task Isaac-Cartpole-v0 --headless --num_envs 256 --max_iterations 30'

# combos to try, in order: "platform|preset|allocation"  (all eu-north1)
COMBOS=(
  "gpu-h100-sxm|1gpu-16vcpu-200gb|on-demand"
  "gpu-h200-sxm|1gpu-16vcpu-200gb|on-demand"
  "gpu-l40s-d|1gpu-16vcpu-96gb|on-demand"
  "gpu-h100-sxm|1gpu-16vcpu-200gb|preemptible"
  "gpu-l40s-d|1gpu-16vcpu-96gb|preemptible"
)

PROVISION_WAIT=1080      # sec to reach RUNNING (covers 47GB image pull + sim init)
RUN_WAIT=2400            # sec to finish once RUNNING
DEADLINE=$(( $(date +%s) + 7200 ))   # give up after 2h total

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$RUNLOG"; }

jstate(){ nebius $PROFILE_ARG ai job get "$1" --format json 2>/dev/null \
  | python3 -c "import json,sys;print(json.load(sys.stdin).get('status',{}).get('state','?'))" 2>/dev/null; }

log "=== Isaac Lab / Nebius test start (host $(hostname)) ==="
log "image=$IMAGE  parent=$PARENT"

WINNER=""; WIN_JOB=""; WIN_COMBO=""
round=0
while [ -z "$WINNER" ] && [ "$(date +%s)" -lt "$DEADLINE" ]; do
  round=$((round+1))
  for combo in "${COMBOS[@]}"; do
    [ "$(date +%s)" -lt "$DEADLINE" ] || break
    IFS='|' read -r PLAT PRESET ALLOC <<<"$combo"
    NAME="isaaclab-${PLAT##gpu-}-${ALLOC}-${STAMP}-r${round}"
    NAME="$(echo "$NAME" | tr -cd 'a-z0-9-' | cut -c1-40)"
    PREEMPT=(); [ "$ALLOC" = "preemptible" ] && PREEMPT=(--preemptible)

    log "round $round: submit $PLAT / $PRESET / $ALLOC  (name=$NAME)"
    CREATE="$(nebius $PROFILE_ARG ai job create \
        --parent-id "$PARENT" --name "$NAME" \
        --platform "$PLAT" --preset "$PRESET" "${PREEMPT[@]}" \
        --image "$IMAGE" --restart-policy never --timeout 2h \
        --container-command /workspace/isaaclab/isaaclab.sh \
        --disk-size 400Gi --shm-size 8Gi \
        --args "$TRAIN" \
        --async --format json 2>&1 | sed 's/\x1b\[[0-9;]*m//g')"
    echo "$CREATE" >>"$RUNLOG"
    JOB="$(echo "$CREATE" | grep -oE 'aijob-[a-z0-9]+' | head -1)"
    if [ -z "$JOB" ]; then log "  submit failed, next combo"; continue; fi
    log "  job=$JOB — waiting to reach RUNNING (<= ${PROVISION_WAIT}s)"

    # phase 1: wait for RUNNING (or terminal / timeout)
    t0=$(date +%s); state="?"; reached_running=""
    while :; do
      state="$(jstate "$JOB")"
      case "$state" in
        RUNNING|COMPLETED) reached_running=1; break;;
        ERROR|FAILED|CANCELLED) break;;
      esac
      [ $(( $(date +%s) - t0 )) -lt "$PROVISION_WAIT" ] || { log "  provisioning timed out (state=$state)"; break; }
      sleep 20
    done

    if [ -z "$reached_running" ]; then
      log "  $PLAT/$ALLOC did not provision (state=$state) — cancel + next"
      nebius $PROFILE_ARG ai job cancel "$JOB" >/dev/null 2>&1
      nebius $PROFILE_ARG ai job delete "$JOB" >/dev/null 2>&1
      continue
    fi

    # phase 2: it's running — let it finish
    log "  RUNNING on $PLAT/$ALLOC — training (<= ${RUN_WAIT}s)"
    t0=$(date +%s)
    while :; do
      state="$(jstate "$JOB")"
      case "$state" in COMPLETED|ERROR|FAILED|CANCELLED) break;; esac
      [ $(( $(date +%s) - t0 )) -lt "$RUN_WAIT" ] || { log "  run timed out (state=$state)"; break; }
      sleep 25
    done
    log "  final state=$state"
    nebius $PROFILE_ARG ai job get  "$JOB" --format json 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' >"$OUT/$JOB.json"
    nebius $PROFILE_ARG ai job logs "$JOB" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' >"$OUT/$JOB.log"

    if [ "$state" = "COMPLETED" ]; then
      WINNER="$state"; WIN_JOB="$JOB"; WIN_COMBO="$combo"
      log "  SUCCESS on $combo (job $JOB)"; break
    else
      log "  ran but ended $state — keeping logs, trying next combo"
      # if it actually produced training output, treat as usable evidence too
      if grep -qiE 'Iteration|reward|Learning iteration|Mean reward' "$OUT/$JOB.log" 2>/dev/null; then
        WINNER="$state"; WIN_JOB="$JOB"; WIN_COMBO="$combo"
        log "  training output present despite $state — accepting as evidence"; break
      fi
    fi
  done
  [ -z "$WINNER" ] && { log "round $round: nothing landed; sleeping 300s before retry"; sleep 300; }
done

# --- summary + report -------------------------------------------------------
python3 - "$OUT" "$WINNER" "$WIN_JOB" "$WIN_COMBO" "$IMAGE" <<'PY'
import json,sys,os
out,winner,job,combo,image=sys.argv[1:6]
summary=dict(winner=winner or "NONE", job=job, combo=combo, image=image, dir=out)
json.dump(summary, open(os.path.join(out,"summary.json"),"w"), indent=2)
print("summary:", summary)
PY

log "regenerating report"
python3 "$SPIKE/job/build_lab_report.py" >>"$RUNLOG" 2>&1 || log "report build failed (see runner.log)"

if [ -n "$WINNER" ]; then
  log "=== DONE: Isaac Lab ran on Nebius ($WIN_COMBO, state=$WINNER, job=$WIN_JOB) ==="
else
  log "=== DONE: no combo provisioned within the window — see runner.log; report notes the attempts ==="
fi

# one-shot: disable the cron line after this run so it doesn't repeat nightly
( crontab -l 2>/dev/null | grep -v 'isaac_lab_nebius_test.sh' ) | crontab - 2>/dev/null || true
log "cron entry removed (one-shot)."
