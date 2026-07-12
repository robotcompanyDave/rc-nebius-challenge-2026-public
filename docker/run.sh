#!/usr/bin/env bash
# Run a push-grasp script inside the Isaac Sim 6.0.0 container.
#
# Reuses the SAME image the rc-remote-platform gateway runs
# (nvcr.io/nvidia/isaac-sim:6.0.0) with our extra deps layered on top
# (push-grasp:dev, built from docker/Dockerfile.dev). The spike dir is MOUNTED,
# so edits are live — no rebuild between runs.
#
# Why headless AND renders: `--runtime=nvidia` + NVIDIA_DRIVER_CAPABILITIES=all
# inject the GL/Vulkan/RTX libs (rc-remote-platform compose note: without this the
# renderer comes up blank).
#
# Runs as ROOT: the image's /isaac-sim/python.sh is owned by the isaac-sim user
# (uid 1234, mode 750) so no other uid can exec it, and root can also write to
# your data/ mount. Anything root writes under data/ is chown'd back to you at
# the end (via a root container — no host sudo needed). Shader/GL caches persist
# in named volumes so 2nd+ boots are fast.
#
#   docker/run.sh tools/probe_soft_tilt.py [args...]
#   GS_ST_VIDEO=1 docker/run.sh tools/probe_soft_tilt.py   # GS_* env forwarded
#
# Needs a free GPU (check `nvidia-smi`; one Isaac fits on the 16 GB 5080).
set -uo pipefail

SPIKE="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${PUSH_GRASP_IMAGE:-push-grasp:dev}"
UIDGID="$(id -u):$(id -g)"
[ "$#" -ge 1 ] || { echo "usage: $0 <script.py> [args...]" >&2; exit 2; }

# forward any GS_* env vars from the caller
env_args=()
while IFS= read -r kv; do env_args+=(-e "$kv"); done < <(env | grep -E '^GS_' || true)

docker run --rm --runtime=nvidia --user 0:0 \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e PYTHONUNBUFFERED=1 "${env_args[@]}" \
  -v isaac_kit_cache:/isaac-sim/kit/cache \
  -v isaac_ov_cache:/root/.cache/ov \
  -v isaac_pip_cache:/root/.cache/pip \
  -v isaac_gl_cache:/root/.cache/nvidia/GLCache \
  -v isaac_compute_cache:/root/.nv/ComputeCache \
  -v isaac_logs:/root/.nvidia-omniverse/logs \
  -v isaac_ov_data:/root/.local/share/ov/data \
  -v isaac_ov_docs:/root/.local/share/ov/documents \
  -v "$SPIKE":/workspace/push-grasp -w /workspace/push-grasp \
  --entrypoint /isaac-sim/python.sh \
  "$IMAGE" "$@"
rc=$?

# reclaim ownership of anything root wrote under data/ (fast — just metadata)
if [ -d "$SPIKE/data" ]; then
  docker run --rm --user 0:0 -v "$SPIKE/data":/d --entrypoint chown "$IMAGE" -R "$UIDGID" /d \
    >/dev/null 2>&1 || true
fi
exit "$rc"
