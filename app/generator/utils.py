import subprocess, shutil
from pathlib import Path

def run_cmd(cmd, cwd=None):
    print("RUN:", " ".join(str(c) for c in cmd))
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = p.stdout.decode(errors="ignore")
    err = p.stderr.decode(errors="ignore")
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(str(c) for c in cmd)}\nSTDOUT:\n{out}\nSTDERR:\n{err}")
    return out

def ensure_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError("ffmpeg is required inside the container.")
