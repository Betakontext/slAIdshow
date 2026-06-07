#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
verify_runtime.py

Purpose:
- Compare critical audio/Whisper settings between your UI (.env used by FastAPI app)
  and this test script.
- Perform a 3s probe recording using the configured device/sample rate.
- Print recommended .env values to align the UI behavior with the test setup.

Notes:
- This script does not start any network services and stays local.
- It is safe to run multiple times.
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import sounddevice as sd


def _env_load_dotenv_inline(dotenv_path: Path) -> None:
    """
    Lightweight .env loader:
    - Reads KEY=VALUE pairs.
    - Ignores comments and blank lines.
    - Does not override values already set in the environment.
    """
    if not dotenv_path.is_file():
        return
    for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        # Support leading `export KEY=VALUE`
        if s.lower().startswith("export "):
            s = s[7:].strip()
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip("'").strip('"')
        os.environ.setdefault(k, v)


def env_str(k: str, d: str) -> str:
    """Read string from env with default."""
    v = os.getenv(k)
    return v.strip() if v is not None else d


def env_int(k: str, d: int) -> int:
    """Read int from env with default and safe casting."""
    v = os.getenv(k)
    try:
        return int(v) if v is not None else d
    except Exception:
        return d


def env_float(k: str, d: float) -> float:
    """Read float from env with default and safe casting."""
    v = os.getenv(k)
    try:
        return float(v) if v is not None else d
    except Exception:
        return d


def resample_to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    """
    Very simple linear resampler to 16 kHz mono float32.
    - Assumes x is 1D float waveform in [-1, 1].
    - For quick diagnostics (not high-fidelity SRC).
    """
    if x.size == 0:
        return np.array([], dtype=np.float32)
    if sr == 16000:
        return x.astype(np.float32, copy=False)
    # Compute target length proportional to sample rate ratio
    tgt = int(round(x.shape[0] * (16000.0 / float(sr))))
    if tgt <= 0:
        return np.array([], dtype=np.float32)
    # Map input to [0,1), then interpolate to new grid
    t_in = np.linspace(0.0, 1.0, num=x.shape[0], endpoint=False, dtype=np.float64)
    t_out = np.linspace(0.0, 1.0, num=tgt, endpoint=False, dtype=np.float64)
    y = np.interp(t_out, t_in, x.astype(np.float64, copy=False))
    return y.astype(np.float32, copy=False)


def pick_input_device(prefer_name: Optional[str]) -> int:
    """
    Select a suitable input device index for recording.
    Strategy:
    1) Exact name match with input channels > 0.
    2) Substring match (case-insensitive).
    3) Prefer PulseAudio/PipeWire devices on Linux.
    4) First device with input channels > 0.
    Raises RuntimeError if none found.
    """
    devs = sd.query_devices()
    if prefer_name:
        for i, d in enumerate(devs):
            if d.get("name") == prefer_name and d.get("max_input_channels", 0) > 0:
                return i
        pl = prefer_name.lower()
        for i, d in enumerate(devs):
            if pl in (d.get("name", "").lower()) and d.get("max_input_channels", 0) > 0:
                return i
    # Heuristics for Linux desktop
    for i, d in enumerate(devs):
        nm = (d.get("name") or "").lower()
        if ("pulse" in nm or "pipewire" in nm) and d.get("max_input_channels", 0) > 0:
            return i
    # Any input-capable device
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0:
            return i
    raise RuntimeError("No input device with input channels found")


def main() -> None:
    # Try to load .env from current working directory if not already exported
    dotenv = Path(".env")
    _env_load_dotenv_inline(dotenv)

    # Collect critical parameters (matching the UI/env expectations)
    cfg: Dict[str, Any] = {
        "APP_AUDIO_DEVICE": env_str("APP_AUDIO_DEVICE", ""),
        "APP_SAMPLE_RATE": env_int("APP_SAMPLE_RATE", 48000),
        "APP_FRAME_DURATION_MS": env_int("APP_FRAME_DURATION_MS", 20),
        "APP_DISABLE_VAD": env_int("APP_DISABLE_VAD", 0),
        "APP_RMS_VAD_THRESHOLD": env_float("APP_RMS_VAD_THRESHOLD", 0.035),
        "APP_SNAPSHOT_SEC": env_float("APP_SNAPSHOT_SEC", 3.5),
        "APP_MAX_SILENCE_MS": env_int("APP_MAX_SILENCE_MS", 300),
        "APP_WHISPER_MODEL_PATH": env_str("APP_WHISPER_MODEL_PATH", ""),
        "APP_WHISPER_LANGUAGE": env_str("APP_WHISPER_LANGUAGE", "de"),
        "APP_WHISPER_THREADS": env_int("APP_WHISPER_THREADS", 4),
        "APP_WHISPER_TEMPERATURE": env_float("APP_WHISPER_TEMPERATURE", 0.0),
    }

    print("[RUNTIME] Current .env / ENV values:")
    for k, v in cfg.items():
        print(f"  - {k} = {v}")

    # Sanity check: whisper.cpp model file present?
    mp = Path(cfg["APP_WHISPER_MODEL_PATH"])
    if not mp.is_file():
        print(f"[ERR] Whisper model not found: {mp}")
        print("      Set APP_WHISPER_MODEL_PATH to a valid ggml/gguf model file (e.g., ggml-base.bin).")
        return

    # 3 s probe recording with the configured device and sample rate
    prefer = cfg["APP_AUDIO_DEVICE"] or None

    try:
        idx = pick_input_device(prefer)
    except Exception as e:
        print(f"[ERR] Could not select an input device: {e}")
        print("      Tip: Set APP_AUDIO_DEVICE to the exact device name from sounddevice.query_devices().")
        return

    try:
        name = sd.query_devices()[idx]["name"]
    except Exception:
        name = f"idx:{idx}"

    sd.default.device = (idx, None)
    sr = int(cfg["APP_SAMPLE_RATE"])

    print(f"[AUDIO] Probe record 3s @ {sr} Hz, device {idx}:{name!r}")
    try:
        rec = sd.rec(int(3.0 * sr), samplerate=sr, channels=1, dtype="float32")
        sd.wait()
    except Exception as e:
        print(f"[ERR] Recording failed: {e}")
        print("      Hints:")
        print("       - Check microphone permissions (OS privacy settings).")
        print("       - On Linux, ensure ALSA/PulseAudio/PipeWire is configured.")
        print("       - Try a different device or sample rate (e.g., 16000 or 48000).")
        return

    x = rec[:, 0] if rec.size else np.array([], dtype=np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    print(f"[AUDIO] Peak={peak:.3f} RMS={rms:.3f} samples={x.shape[0]}")

    if peak < 0.01:
        print("[HINT] Very low signal. Increase mic gain or lower APP_RMS_VAD_THRESHOLD (e.g., 0.02–0.03).")

    # Quick resample to 16 kHz to emulate whisper.cpp input constraints
    x16 = resample_to_16k(x, sr)
    if x16.size == 0:
        print("[ERR] Resampling produced an empty buffer.")
        return

    # Derive simple recommendations to align UI behavior to this recording
    print("\n[RECOMMEND] Set these values to approximate UI ≈ test script behavior:")
    recs: Dict[str, Any] = {
        "APP_DISABLE_VAD": 0,
        "APP_RMS_VAD_THRESHOLD": 0.035 if peak >= 0.1 else 0.02,
        "APP_SNAPSHOT_SEC": 3.0,
        "APP_MAX_SILENCE_MS": 300,
        "APP_FRAME_DURATION_MS": 20,
        "APP_WHISPER_TEMPERATURE": 0.0,
    }
    for k, v in recs.items():
        print(f"  {k}={v}")

    # Compact checklist to verify end-to-end audio path in the UI
    print("\n[CHECKLIST]")
    print("  1) UI logs should show: '[AUDIO] open OK' and NOT 'switching to polling read() loop'.")
    print("  2) VAD active (disable_vad=False / webrtc_vad=True if installed).")
    print("  3) '[WHISPER] text:' should emit only for segments with Peak>=0.01; quiet frames filtered.")
    print("  4) SNAPSHOT_SEC 3.0–3.5; first snapshot deadline ~1.0–1.2s.")

    # Optional: JSON summary for CI usage (enable with ENV VERIFY_JSON=1)
    if os.getenv("VERIFY_JSON", "0").strip() in {"1", "true", "yes"}:
        out: Dict[str, Any] = {
            "config": cfg,
            "audio": {"device_index": idx, "device_name": name, "sr": sr, "peak": peak, "rms": rms, "n": int(x.shape[0])},
            "recommend": recs,
        }
        print(json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()
