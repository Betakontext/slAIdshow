#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
audio_test.py

Purpose:
- Quick microphone sanity check using sounddevice.
- Records a short clip and reports peak/RMS levels.
- Helps validate APP_AUDIO_DEVICE and APP_SAMPLE_RATE from .env.

Features:
- Robust .env autoload (supports inline '#'-comments and 'export KEY=VALUE').
- Robust device selection (exact/partial, Pulse/PipeWire preference).
- Optional JSON summary (AUDIO_TEST_JSON=1) and optional WAV dump (AUDIO_TEST_SAVE_WAV=path.wav).

Environment variables:
- APP_AUDIO_DEVICE (str)
- APP_SAMPLE_RATE (int, default 16000)
- APP_TEST_DURATION (float, default 3.0)
- AUDIO_TEST_JSON ("1"/"true" for JSON)
- AUDIO_TEST_SAVE_WAV (path to save WAV)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd


# ---------- .env loader and helpers (shared logic) ----------

def _strip_inline_comment(s: str) -> str:
    """Strip inline comments after '#', unless inside quotes."""
    s = s.strip()
    out = []
    in_single = False
    in_double = False
    for ch in s:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        if ch == "#" and not in_single and not in_double:
            break
        out.append(ch)
    return "".join(out).strip()


def _load_dotenv_inline(p: Path) -> None:
    """
    Load .env safely:
    - Supports 'export KEY=VALUE'
    - Strips inline comments after '#'
    - Does not override already-set env vars
    """
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        k, v = line.split("=", 1)
        k = k.strip()
        v = _strip_inline_comment(v).strip().strip("'").strip('"')
        if k and v != "":
            os.environ.setdefault(k, v)


def env_str(k: str, default: str) -> str:
    v = os.getenv(k)
    return v.strip() if v is not None else default


def env_int(k: str, default: int) -> int:
    v = os.getenv(k)
    if v is None:
        return default
    try:
        vv = _strip_inline_comment(v)
        return int(vv)
    except Exception:
        return default


def env_float(k: str, default: float) -> float:
    v = os.getenv(k)
    if v is None:
        return default
    try:
        vv = _strip_inline_comment(v)
        return float(vv)
    except Exception:
        return default


# Load .env from current working directory
_load_dotenv_inline(Path(".env"))

# ---------- Config ----------

DEVICE_PREF = env_str("APP_AUDIO_DEVICE", "pipewire")
SR = env_int("APP_SAMPLE_RATE", 16000)
DUR = env_float("APP_TEST_DURATION", 3.0)
WANT_JSON = env_str("AUDIO_TEST_JSON", "0").lower() in {"1", "true", "yes"}
WAV_OUT = env_str("AUDIO_TEST_SAVE_WAV", "").strip()


# ---------- Audio utilities ----------

def pick_input_device(prefer_name: Optional[str]) -> Tuple[int, str]:
    """
    Select an input device (index, name).
    Strategy: exact name → substring → Pulse/PipeWire → first input-capable.
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
    for i, d in enumerate(devs):
        nm = (d.get("name") or "").lower()
        if ("pulse" in nm or "pipewire" in nm) and d.get("max_input_channels", 0) > 0:
            return i, d.get("name", str(i))
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0:
            return i, d.get("name", str(i))
    raise RuntimeError("No input device with input channels found")


def save_wav_int16_mono(path: Path, x: np.ndarray, sr: int) -> None:
    """Save mono float32 [-1,1] as 16-bit PCM WAV without extra deps."""
    import wave
    x16 = np.clip(x, -1.0, 1.0)
    x16 = (x16 * 32767.0).astype(np.int16, copy=False)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(x16.tobytes())


# ---------- Main ----------

def main() -> None:
    try:
        idx, name = pick_input_device(DEVICE_PREF or None)
    except Exception as e:
        print(f"[ERR] Could not select an input device: {e}")
        print("      Tips:")
        print("       - Set APP_AUDIO_DEVICE to an exact device name from sounddevice.query_devices().")
        print("       - On Linux, ensure PulseAudio/PipeWire is running; try 'pipewire' or 'pulse'.")
        print("       - On macOS/Windows, check microphone privacy permissions.")
        if WANT_JSON:
            print(json.dumps({"ok": False, "error": str(e)}))
        sys.exit(2)

    print(f"Testing microphone: device='{name}' (idx={idx}), sr={SR}, duration={DUR}s")
    print("Please speak …")

    sd.default.device = (idx, None)
    sd.default.samplerate = int(SR)
    sd.default.channels = 1

    try:
        audio = sd.rec(int(DUR * SR), samplerate=SR, channels=1, dtype="float32")
        sd.wait()
    except Exception as e:
        print(f"[ERR] Recording failed: {e}")
        print("      Hints:")
        print("       - Check mic permissions in OS privacy settings.")
        print("       - Try a different sample rate (16000 or 48000).")
        print("       - Try another device name or remove APP_AUDIO_DEVICE to auto-pick.")
        if WANT_JSON:
            print(json.dumps({"ok": False, "error": f"recording_failed: {e}"}))
        sys.exit(3)

    x = audio[:, 0] if audio.size else np.array([], dtype=np.float32)
    if x.size == 0:
        print("[ERR] Empty buffer recorded.")
        if WANT_JSON:
            print(json.dumps({"ok": False, "error": "empty_buffer"}))
        sys.exit(4)

    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x ** 2)))
    print(f"Peak={peak:.3f}, RMS={rms:.3f}")

    if peak < 0.05:
        print("Warning: Very low level. Check pavucontrol / mic gain or move closer to the mic.")
    else:
        print("OK: Level is sufficient.")

    if WAV_OUT:
        try:
            p = Path(WAV_OUT)
            p.parent.mkdir(parents=True, exist_ok=True)
            save_wav_int16_mono(p, x, SR)
            print(f"[INFO] Saved recording to: {p}")
        except Exception as e:
            print(f"[WARN] Could not save WAV: {e}")

    if WANT_JSON:
        print(json.dumps({
            "ok": True,
            "device": {"index": idx, "name": name},
            "sr": SR,
            "duration": DUR,
            "peak": peak,
            "rms": rms
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
