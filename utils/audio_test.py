#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
audio_test.py

Purpose:
- Quick microphone sanity check using sounddevice.
- Records a short clip and reports peak/RMS levels.
- Helps to validate APP_AUDIO_DEVICE and APP_SAMPLE_RATE settings from .env.

Features:
- Robust device selection (exact/partial match, Pulse/PipeWire preference).
- Clear error messages and simple recommendations.
- Optional JSON summary for CI/logging (AUDIO_TEST_JSON=1).
- Optional WAV dump of the recording (AUDIO_TEST_SAVE_WAV=path.wav).

Environment variables:
- APP_AUDIO_DEVICE: preferred device name (string). Example: "pipewire", "Built-in Microphone"
- APP_SAMPLE_RATE: sample rate (int), default 16000
- APP_TEST_DURATION: duration in seconds (float), default 3.0
- AUDIO_TEST_JSON: "1"/"true" to print a JSON result in addition to human logs
- AUDIO_TEST_SAVE_WAV: file path to save the raw recording as 16-bit PCM WAV
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd


def env_str(k: str, d: str) -> str:
    v = os.getenv(k)
    return v.strip() if v is not None else d


def env_int(k: str, d: int) -> int:
    v = os.getenv(k)
    try:
        return int(v) if v is not None else d
    except Exception:
        return d


def env_float(k: str, d: float) -> float:
    v = os.getenv(k)
    try:
        return float(v) if v is not None else d
    except Exception:
        return d


def pick_input_device(prefer_name: Optional[str]) -> Tuple[int, str]:
    """
    Select a suitable input device index and name for recording.
    Strategy:
    1) Exact name match with input channels > 0.
    2) Substring match (case-insensitive).
    3) Prefer PulseAudio/PipeWire devices on Linux.
    4) First device with input channels > 0.

    Returns (index, name) or raises RuntimeError if none found.
    """
    devs = sd.query_devices()
    if prefer_name:
        for i, d in enumerate(devs):
            if d.get("name") == prefer_name and d.get("max_input_channels", 0) > 0:
                return i, d.get("name", str(i))
        pl = prefer_name.lower()
        for i, d in enumerate(devs):
            if pl in (d.get("name", "").lower()) and d.get("max_input_channels", 0) > 0:
                return i, d.get("name", str(i))
    # Linux desktop preference
    for i, d in enumerate(devs):
        nm = (d.get("name") or "").lower()
        if ("pulse" in nm or "pipewire" in nm) and d.get("max_input_channels", 0) > 0:
            return i, d.get("name", str(i))
    # Any input-capable device
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0:
            return i, d.get("name", str(i))
    raise RuntimeError("No input device with input channels found")


def save_wav_int16_mono(path: Path, x: np.ndarray, sr: int) -> None:
    """
    Save a mono float32 waveform [-1, 1] as 16-bit PCM WAV.
    Uses Python's built-in wave module to avoid extra deps.
    """
    import wave
    # Clip to [-1,1], scale to int16
    x16 = np.clip(x, -1.0, 1.0)
    x16 = (x16 * 32767.0).astype(np.int16, copy=False)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(int(sr))
        wf.writeframes(x16.tobytes())


def main() -> None:
    # Read env
    device_pref = env_str("APP_AUDIO_DEVICE", "pipewire")
    sr = env_int("APP_SAMPLE_RATE", 16000)
    dur = env_float("APP_TEST_DURATION", 3.0)
    want_json = env_str("AUDIO_TEST_JSON", "0").lower() in {"1", "true", "yes"}
    wav_out = env_str("AUDIO_TEST_SAVE_WAV", "").strip()
    wav_path = Path(wav_out) if wav_out else None

    # Select device
    try:
        idx, name = pick_input_device(device_pref or None)
    except Exception as e:
        print(f"[ERR] Could not select an input device: {e}")
        print("      Tips:")
        print("       - Set APP_AUDIO_DEVICE to an exact device name from sounddevice.query_devices().")
        print("       - On Linux, ensure PulseAudio/PipeWire is running; try 'pipewire' or 'pulse'.")
        print("       - On macOS/Windows, check microphone privacy permissions.")
        if want_json:
            print(json.dumps({"ok": False, "error": str(e)}))
        sys.exit(2)

    print(f"Testing microphone: device='{name}' (idx={idx}), sr={sr}, duration={dur}s")
    print("Please speak …")

    # Configure sounddevice defaults explicitly
    sd.default.device = (idx, None)
    sd.default.samplerate = int(sr)
    sd.default.channels = 1

    try:
        audio = sd.rec(int(dur * sr), samplerate=sr, channels=1, dtype="float32")
        sd.wait()
    except Exception as e:
        print(f"[ERR] Recording failed: {e}")
        print("      Hints:")
        print("       - Check mic permissions in OS privacy settings.")
        print("       - Try a different sample rate (16000 or 48000).")
        print("       - Try another device name or remove APP_AUDIO_DEVICE to auto-pick.")
        if want_json:
            print(json.dumps({"ok": False, "error": f"recording_failed: {e}"}))
        sys.exit(3)

    x = audio[:, 0] if audio.size else np.array([], dtype=np.float32)
    if x.size == 0:
        print("[ERR] Empty buffer recorded.")
        if want_json:
            print(json.dumps({"ok": False, "error": "empty_buffer"}))
        sys.exit(4)

    # Basic level analysis
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x ** 2)))

    print(f"Peak={peak:.3f}, RMS={rms:.3f}")
    if peak < 0.05:
        print("Warning: Very low level. Check pavucontrol / mic gain or move closer to the mic.")
    else:
        print("OK: Level is sufficient.")

    # Optional WAV dump for debugging
    if wav_path:
        try:
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            save_wav_int16_mono(wav_path, x, sr)
            print(f"[INFO] Saved recording to: {wav_path}")
        except Exception as e:
            print(f"[WARN] Could not save WAV: {e}")

    if want_json:
        print(json.dumps({
            "ok": True,
            "device": {"index": idx, "name": name},
            "sr": sr,
            "duration": dur,
            "peak": peak,
            "rms": rms
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
