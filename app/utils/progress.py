import json, time

def _progress_path(job_dir: Path) -> Path:
    return job_dir / "progress.json"

def progress_init(job_dir: Path, total: float = 100.0):
    data = {"percent": 0.0, "stage": "starting", "detail": "", "ts": time.time(), "total": total}
    _progress_path(job_dir).write_text(json.dumps(data))

def progress_update(job_dir: Path, percent: float, stage: str, detail: str = ""):
    percent = max(0.0, min(100.0, float(percent)))
    data = {"percent": percent, "stage": stage, "detail": detail, "ts": time.time()}
    _progress_path(job_dir).write_text(json.dumps(data))

def progress_read(job_dir: Path) -> dict:
    p = _progress_path(job_dir)
    if not p.exists():
        return {"percent": 0.0, "stage": "pending", "detail": "", "ts": time.time()}
    return json.loads(p.read_text())
