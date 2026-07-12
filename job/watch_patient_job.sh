#!/usr/bin/env bash
# Lightweight watcher for the "patient" Isaac Lab job left queued on Nebius.
# Runs from cron every ~15 min. While the job is still queued/running it just
# refreshes the report's heartbeat; when the job reaches a terminal state it
# captures the logs, regenerates the report, and removes its own cron line.
set -uo pipefail
export HOME="${HOME:-/tmp}"
export PATH="$HOME/.nebius/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

SPIKE="${SPIKE:-$REPO}"
# locate the newest patient-job id
PJ="$(ls -1t "$SPIKE"/data/*/lab/patient-job.txt 2>/dev/null | head -1)"
[ -n "$PJ" ] || exit 0
OUT="$(dirname "$PJ")"
JOB="$(sed 's/.*=//' "$PJ" | tr -d '[:space:]')"
[ -n "$JOB" ] || exit 0
W="$OUT/watch.log"

state="$(nebius ai job get "$JOB" --format json 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' \
  | python3 -c "import json,sys;print(json.load(sys.stdin).get('status',{}).get('state','?'))" 2>/dev/null)"
echo "[$(date '+%F %H:%M')] $JOB -> $state" >>"$W"

nebius ai job get  "$JOB" --format json 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' >"$OUT/$JOB.json"
case "$state" in
  COMPLETED|ERROR|FAILED|CANCELLED)
    nebius ai job logs "$JOB" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' >"$OUT/$JOB.log"
    python3 - "$OUT" "$state" "$JOB" <<'PY'
import json,sys,os
out,state,job=sys.argv[1:4]
s=json.load(open(os.path.join(out,"summary.json")))
s["winner"]= state if state=="COMPLETED" else None
s["terminal_state"]=state
json.dump(s, open(os.path.join(out,"summary.json"),"w"), indent=2)
PY
    python3 "$SPIKE/job/build_lab_report.py" >>"$W" 2>&1
    ( crontab -l 2>/dev/null | grep -v 'watch_patient_job.sh' ) | crontab - 2>/dev/null || true
    echo "[$(date '+%F %H:%M')] terminal ($state) — report updated, watcher cron removed" >>"$W"
    ;;
  *)
    # still queued or running: refresh heartbeat in the report
    python3 - "$OUT" "$state" <<'PY'
import json,sys,os
out,state=sys.argv[1:3]
s=json.load(open(os.path.join(out,"summary.json")))
s["last_state"]=state
json.dump(s, open(os.path.join(out,"summary.json"),"w"), indent=2)
PY
    python3 "$SPIKE/job/build_lab_report.py" >>"$W" 2>&1
    ;;
esac
