# push-grasp in Docker (Isaac Sim)

We run the sim in the **official Isaac Sim container** instead of the native
`~/TOOLS/isaac-sim` install — the same method (and the same image) that
`rc-remote-platform`'s gateway uses
([`deploy/compose/docker-compose.yml`](../../../../rc-remote-platform/deploy/compose/docker-compose.yml)).

**Why Docker.** The container path gives a clean, reproducible Isaac runtime and —
critically — a **working RTX renderer**: with `--runtime=nvidia` +
`NVIDIA_DRIVER_CAPABILITIES=all` the GL/Vulkan/RTX libs are injected into the
container (the platform compose notes that without this "the renderer comes up
blank"). That's the leading candidate to fix the blank-render problem we hit with
the bare native scene.

## Reusing the image

Yes — we reuse `nvcr.io/nvidia/isaac-sim:6.0.0` directly (it's already pulled
locally, ~32 GB; it's public on nvcr.io, no NGC login). We don't bake our code
into it; instead:

- **`Dockerfile.dev`** layers only the deps the base lacks (opencv, pyarrow,
  imageio-ffmpeg — the base already ships numpy/matplotlib/boto3/imageio/pillow)
  and mounts the code at run time, so edits are live with no rebuild.
- **`Dockerfile`** (spike root) is the *cloud-job* image — it bakes the code and
  writes to a bucket; used for Nebius, not local iteration.

## Setup (once)

```bash
cd ~/RC/rc-spike-soft-surface/spikes/push-grasp
docker build -f docker/Dockerfile.dev -t push-grasp:dev .   # GPU-free; ~1–2 min
```

## Run (needs a free GPU)

```bash
# check the GPU is free first — only one Isaac fits on the 16 GB 5080
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

docker/run.sh tools/probe_soft_tilt.py                      # headless probe
GS_ST_VIDEO=1 docker/run.sh tools/probe_soft_tilt.py        # GS_* env forwarded
```

`run.sh` mounts the spike dir, forwards `GS_*` env, and persists the shader/GL
caches in named volumes so the 2nd+ boot is fast (first boot still compiles
shaders — minutes).

## Gotchas / TODO on first run

- **GPU sharing:** `rc-remote-platform`'s gateway may already hold the GPU
  (`docker ps` → `rc-isaac-gateway`). Two Isaac processes won't fit in 16 GB —
  wait for it to stop, or stop it.
- **File ownership:** the container runs as root, so outputs under `data/` come
  out root-owned. Reclaim: `sudo chown -R "$(id -u):$(id -g)" data/`.
- **mp4 encoding:** `opencv-python-headless` may lack an mp4 encoder; if
  `cv2.VideoWriter` produces empty/odd files, re-encode with the bundled
  `imageio-ffmpeg` (already installed). Verify on first video run.
- **Non-root run:** switching to `--user $(id -u):$(id -g)` (NVIDIA's non-root
  pattern, with a writable `$HOME` + host-dir caches) avoids the ownership issue
  but needs testing — left as a follow-up once we can run.
