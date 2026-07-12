#!/usr/bin/env bash
# Run a command inside the Isaac Lab container (3.0.0-beta2, pairs with the
# Isaac Sim 6.0.x generation we train against). Mirrors docker/run.sh: nvidia
# runtime + all caps (else renderer/PhysX GPU is blank), root in container,
# named cache volumes for fast 2nd+ boots, spike mounted live at
# /workspace/push-grasp, ownership of data/ reclaimed afterwards.
#
# Default image: push-grasp:lab — Isaac Lab v3.0.0-beta2 installed into our
# PROVEN isaac-sim:6.0.0 (the NGC isaac-lab image's Kit is broken on this box;
# see docker/Dockerfile.lab header).
#
#   docker/run_lab.sh "cd /workspace/isaaclab && ./isaaclab.sh -p scripts/... --headless"
set -uo pipefail

SPIKE="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${PUSH_GRASP_LAB_IMAGE:-push-grasp:lab}"
UIDGID="$(id -u):$(id -g)"
[ "$#" -ge 1 ] || { echo "usage: $0 <cmd...>" >&2; exit 2; }

env_args=()
while IFS= read -r kv; do env_args+=(-e "$kv"); done < <(env | grep -E '^GS_' || true)

docker run --rm --runtime=nvidia --user 0:0 \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e OMNI_KIT_ALLOW_ROOT=1 -e PYTHONUNBUFFERED=1 "${env_args[@]}" \
  -v isaac_lab_kit_cache:/isaac-sim/kit/cache \
  -v isaac_lab_ov_cache:/root/.cache/ov \
  -v isaac_lab_pip_cache:/root/.cache/pip \
  -v isaac_lab_gl_cache:/root/.cache/nvidia/GLCache \
  -v isaac_lab_compute_cache:/root/.nv/ComputeCache \
  -v isaac_lab_logs:/root/.nvidia-omniverse/logs \
  -v isaac_lab_data:/root/.local/share/ov/data \
  -v "$SPIKE":/workspace/push-grasp \
  --entrypoint /bin/bash \
  "$IMAGE" -c "$*"
rc=$?

if [ -d "$SPIKE/data" ]; then
  docker run --rm --user 0:0 -v "$SPIKE/data":/d --entrypoint chown "$IMAGE" \
    -R "$UIDGID" /d >/dev/null 2>&1 || true
fi
exit "$rc"
