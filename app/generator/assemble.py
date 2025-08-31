from pathlib import Path
from .utils import run_cmd, ensure_ffmpeg

def images_to_video(image_glob: str, fps: int, out_video: Path):
    ensure_ffmpeg()
    cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-pattern_type", "glob", "-i", image_glob,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_video)]
    run_cmd(cmd)
    return out_video

def add_audio_to_video(video_in: Path, audio_in: Path, video_out: Path):
    ensure_ffmpeg()
    cmd = ["ffmpeg", "-y", "-i", str(video_in), "-i", str(audio_in),
           "-c:v", "copy", "-c:a", "aac", "-shortest", str(video_out)]
    run_cmd(cmd)
    return video_out
