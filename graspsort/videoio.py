"""H.264 finalization for the MP4 deliverables.

The OpenCV wheel in the Isaac-bundled python ships no H.264 encoder, so the
render/train scripts write MPEG-4 Part 2 ("mp4v") — which browsers, Slack and
Notion refuse to play. This re-encodes a finished mp4 in place to H.264 +
yuv420p using the ffmpeg binary bundled with imageio-ffmpeg.

Pure python, safe to import before SimulationApp.
"""
import os
import subprocess


def to_h264(path: str, crf: int = 20) -> str:
    """Re-encode an mp4 in place to H.264/yuv420p; returns path unchanged on failure."""
    try:
        import imageio_ffmpeg
    except ImportError:
        print(f"[videoio] imageio-ffmpeg not installed — {os.path.basename(path)} "
              "left as mp4v (won't play in browsers)", flush=True)
        return path
    tmp = path + ".h264.tmp.mp4"
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-i", path,
        # yuv420p requires even dimensions; pad by one pixel if needed
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        tmp,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, OSError) as e:
        err = getattr(e, "stderr", "") or str(e)
        print(f"[videoio] h264 re-encode failed for {path}: {err.strip()}", flush=True)
        if os.path.exists(tmp):
            os.remove(tmp)
        return path
    os.replace(tmp, path)
    return path
