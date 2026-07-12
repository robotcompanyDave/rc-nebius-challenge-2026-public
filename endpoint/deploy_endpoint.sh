#!/usr/bin/env bash
# Build, push, and deploy the soft-surface grasp micro-service as a CPU Nebius
# Serverless AI Endpoint. CPU-only (cpu-d3) — no GPU, so it uses the healthy
# cloudapps CPU quota instead of the capacity-blocked GPU pool.
set -euo pipefail
export PATH="$HOME/.nebius/bin:$PATH"

HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${NEBIUS_REGISTRY:?set NEBIUS_REGISTRY}/graspsvc:latest"
PARENT="${NEBIUS_PROJECT_ID:?set NEBIUS_PROJECT_ID}"   # eu-north1

if [ "${1:-}" = "--build" ]; then
  docker build -t graspsvc:latest "$HERE"
  docker tag graspsvc:latest "$IMAGE"
  docker push "$IMAGE"
fi

nebius ai endpoint create \
  --parent-id "$PARENT" --name graspsvc \
  --platform cpu-d3 --preset 4vcpu-16gb \
  --image "$IMAGE" \
  --container-port 8080/http --public \
  --async --format json

echo "Deployed (async). Watch: nebius ai endpoint get <id> ; then curl its public https URL /health"
