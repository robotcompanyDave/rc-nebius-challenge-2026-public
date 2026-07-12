#!/usr/bin/env bash
# Watches the deployed graspsvc CPU endpoint. While it provisions it refreshes
# the report; when it goes live it curls /health + /score, writes the live proof,
# regenerates the report, and removes its own cron line.
set -uo pipefail
export HOME="${HOME:-/tmp}"
export PATH="$HOME/.nebius/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SPIKE="${SPIKE:-$REPO}"; DATA="$SPIKE/data"

EID="$(grep -oE 'aiendpoint-[a-z0-9]+' "$DATA/graspsvc-endpoint.txt" 2>/dev/null | head -1)"
[ -n "$EID" ] || exit 0
JSON="$(nebius ai endpoint get "$EID" --format json 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g')"
[ -n "$JSON" ] || exit 0
echo "$JSON" > "$DATA/graspsvc-endpoint-state.json"
STATE="$(echo "$JSON" | python3 -c "import json,sys;print(json.load(sys.stdin).get('status',{}).get('state','?'))" 2>/dev/null)"
echo "[$(date '+%F %H:%M')] $EID -> $STATE" >> "$DATA/watch-endpoint.log"

case "$STATE" in
  ACTIVE|RUNNING|READY)
    URL="$(echo "$JSON" | python3 -c "import json,sys,re;print((re.findall(r'https://[A-Za-z0-9._/-]+',json.dumps(json.load(sys.stdin).get('status',{})))or[''])[0])" 2>/dev/null)"
    if [ -n "$URL" ]; then
      D="$DATA/$(date +%F)"; mkdir -p "$D"
      { echo "### LIVE on Nebius: $URL"; echo "### GET /health"; curl -s -m 15 "$URL/health"; echo
        echo "### POST /score"; curl -s -m 15 -X POST "$URL/score" -H 'content-type: application/json' \
          -d '{"part":{"kind":"washer","size":"m12","pose":"flat"},"action":{"strategy":"tilt","tilt_deg":14,"xy_offset":[0.002,0.0]},"scene":{"n_close":1,"nearest_mm":30}}'; echo
      } > "$D/endpoint-live-proof.txt" 2>&1
    fi
    python3 "$SPIKE/job/build_endpoint_report.py" >> "$DATA/watch-endpoint.log" 2>&1
    ( crontab -l 2>/dev/null | grep -v 'watch_endpoint.sh' ) | crontab - 2>/dev/null || true
    echo "[$(date '+%F %H:%M')] LIVE — report updated, watcher removed" >> "$DATA/watch-endpoint.log"
    ;;
  ERROR|FAILED|CANCELLED)
    python3 "$SPIKE/job/build_endpoint_report.py" >> "$DATA/watch-endpoint.log" 2>&1
    ( crontab -l 2>/dev/null | grep -v 'watch_endpoint.sh' ) | crontab - 2>/dev/null || true
    ;;
  *)
    python3 "$SPIKE/job/build_endpoint_report.py" >> "$DATA/watch-endpoint.log" 2>&1
    ;;
esac
