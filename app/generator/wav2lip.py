import os
from pathlib import Path
from ..config import WAV2LIP_PATH
from .utils import run_cmd

def apply_wav2lip(video_in: Path, audio_in: Path, out_path: Path):
    repo = Path(WAV2LIP_PATH)
    if not repo.exists():
        raise FileNotFoundError(f"Wav2Lip repo not found at {repo}")
    ckpt = os.environ.get("WAV2LIP_CKPT", str(repo / "checkpoints" / "wav2lip_gan.pth"))
    cmd = ["python", "inference.py", "--checkpoint_path", ckpt, "--face", str(video_in), "--audio", str(audio_in), "--outfile", str(out_path)]
    run_cmd(cmd, cwd=str(repo))
    return out_path
