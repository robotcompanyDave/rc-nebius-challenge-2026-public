#!/usr/bin/env bash
# Linux chunked local data-gen for the v2 (washers/soft/strategies) dataset.
# Fresh Isaac boot per 100-attempt chunk (native heap corruption ~150 attempts
# into a single app), resumable (chunks with records.jsonl anywhere in the
# dated layout are skipped), zombie-kit kill + 45s cooldown after a failed
# boot, 8s settle between kit exits.
#
# Dated layout (see data/README.md): chunks land in
#   data/<YYYY-MM-DD>/<HHMM>-ds_v2_chunks/chunk_<seed>/
# and the merge of ALL chunks (any date) lands in
#   data/<YYYY-MM-DD>/<HHMM>-dataset_v2_merged/records.jsonl
#
#   bash tools/gen_dataset_v2.sh
set -u
ISAAC_PYTHON="${ISAAC_PYTHON:-$HOME/TOOLS/isaac-sim/python.sh}"
CHUNK="${CHUNK:-100}"
N_CHUNKS="${N_CHUNKS:-40}"
SEED0="${SEED0:-1000}"
# Measured 2026-07-02 (RTX 5080 laptop, 24 cores): ONE headless gen worker uses
# GPU 0% / ~1.7 GiB VRAM / load ~7 — the box fits ~3 workers for ~3x throughput.
N_WORKERS="${N_WORKERS:-3}"
DAY="${DAY:-$(date +%Y-%m-%d)}"
HHMM="${HHMM:-$(date +%H%M)}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
NEW_DIR="data/$DAY/${HHMM}-ds_v2_chunks"
MERGE_OUT="data/$DAY/${HHMM}-dataset_v2_merged"
T0=$SECONDS

export ACCEPT_EULA=Y PRIVACY_CONSENT=Y PYTHONUTF8=1

run_worker() {  # $1 = worker index; workers stride the seed list
    local w=$1
    for ((s = w; s < N_CHUNKS; s += N_WORKERS)); do
        local seed=$((SEED0 + s))
        if compgen -G "data/*/*ds_v2_chunks*/chunk_$seed/records.jsonl" >/dev/null; then
            echo "==== CHUNK seed=$seed already done, skipping ===="
            continue
        fi
        local dir="$NEW_DIR/chunk_$seed"
        echo "==== CHUNK seed=$seed -> $dir  [w$w] (t=$((SECONDS - T0))s) ===="
        GS_N_ATTEMPTS="$CHUNK" GS_BATCH="$CHUNK" GS_SEED="$seed" \
        GS_OUTPUT_DIR="$dir" GS_HEADLESS=1 GS_GRIPPER=parametric GS_SOFT=1 \
            "$ISAAC_PYTHON" jobs/gen_data.py 2>&1 | grep -E '\[gen\]|Traceback'
        if [[ -f "$dir/records.jsonl" ]]; then
            echo "---- chunk seed=$seed wrote $(wc -l < "$dir/records.jsonl") records [w$w] ----"
            sleep 8    # settle between kit exits and boots (a 3s gap crashed native)
        else
            echo "---- chunk seed=$seed FAILED (no records) [w$w] ----"
            # cool down; do NOT pkill here — other workers' kits are alive
            sleep 45
        fi
    done
}

for ((w = 0; w < N_WORKERS; w++)); do
    run_worker "$w" &
    sleep 20    # stagger boots (simultaneous kit boots are the crashy case)
done
wait

mkdir -p "$MERGE_OUT"
merged="$MERGE_OUT/records.jsonl"
: > "$merged"
total=0
for f in data/*/*ds_v2_chunks*/chunk_*/records.jsonl; do
    [[ -f "$f" ]] || continue
    cat "$f" >> "$merged"
    total=$((total + $(wc -l < "$f")))
done
echo "==== MERGED $total records -> $merged  ($((SECONDS - T0))s) ===="
