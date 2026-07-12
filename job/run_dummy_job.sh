#!/usr/bin/env bash
# Simplest possible Nebius AI "job" smoke test.
#
# Launches a single-GPU container that runs `nvidia-smi` and prints its host
# info, then exits. Purpose: prove the Nebius Jobs pipeline works end-to-end
# (submit -> schedule -> pull image -> run on a real GPU -> collect logs).
#
# Usage:
#   ./run_dummy_job.sh                       # default: L40S in eu-north1
#   PLATFORM=gpu-h100-sxm PRESET=1gpu-16vcpu-200gb PARENT_ID=<project> ./run_dummy_job.sh
#
# Region is selected by PARENT_ID (each Nebius region is its own project).
set -euo pipefail

PARENT_ID="${PARENT_ID:-${NEBIUS_PROJECT_ID:?set NEBIUS_PROJECT_ID}}"   # eu-north1 (home)
PLATFORM="${PLATFORM:-gpu-l40s-d}"                          # Ada, RT cores -> Isaac Sim capable
PRESET="${PRESET:-1gpu-16vcpu-96gb}"                        # smallest 1-GPU preset
IMAGE="${IMAGE:-nvidia/cuda:12.4.1-base-ubuntu22.04}"       # public Docker Hub base image
NAME="${NAME:-nebius-smoke-$(date +%H%M%S)}"

echo "Submitting dummy job '$NAME'  platform=$PLATFORM preset=$PRESET parent=$PARENT_ID"

nebius ai job create \
  --parent-id "$PARENT_ID" \
  --name "$NAME" \
  --platform "$PLATFORM" \
  --preset "$PRESET" \
  --image "$IMAGE" \
  --restart-policy never \
  --timeout 1h \
  --container-command 'bash' \
  --args '-c' \
  --args 'echo "=== Nebius Jobs smoke test ==="; echo "host: $(hostname)"; echo "date: $(date -u)"; nvidia-smi; echo "=== OK ==="' \
  --format json
