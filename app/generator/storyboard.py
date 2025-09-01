# app/generator/storyboard.py
from typing import List, Tuple
import re
import numpy as np
import soundfile as sf
import librosa

def split_script(script: str) -> List[str]:
    s = script.replace("\r", "").strip()
    if not s:
        return []
    # prioridad: doble salto; si no, por oraciones básicas
    parts = [p.strip() for p in s.split("\n\n") if p.strip()]
    if not parts:
        parts = re.split(r'(?<=[\.\?\!])\s+', s)
        parts = [p.strip() for p in parts if p.strip()]
    return parts

def allocate_durations(parts: List[str], total_audio_sec: float) -> List[float]:
    if total_audio_sec <= 0 or not parts:
        return [2.0 for _ in parts]  # fallback 2s por escena
    lengths = np.array([max(1, len(p)) for p in parts], dtype=float)
    weights = lengths / lengths.sum()
    return (weights * total_audio_sec).tolist()

def audio_energy_envelope(wav_path: str, sr_target: int = 22050, hop_ms: int = 50) -> np.ndarray:
    sig, sr = sf.read(wav_path)
    if sig.ndim > 1:
        sig = sig.mean(axis=1)
    if sr != sr_target:
        sig = librosa.resample(sig, orig_sr=sr, target_sr=sr_target)
        sr = sr_target
    hop_length = max(1, int(sr * hop_ms / 1000))
    frame = librosa.util.frame(sig, frame_length=hop_length*2, hop_length=hop_length, axis=0)
    energy = (frame**2).mean(axis=1)
    energy = energy / (energy.max() + 1e-9)
    return energy  # 0..1

def sample_energy_for_frames(energy: np.ndarray, total_frames: int) -> List[float]:
    if energy.size == 0 or total_frames <= 1:
        return [0.5]*total_frames
    xs = np.linspace(0, len(energy)-1, total_frames)
    idxs = np.clip(np.round(xs).astype(int), 0, len(energy)-1)
    return energy[idxs].tolist()
