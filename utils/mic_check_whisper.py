#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mic_check_whisper.py

- Robust .env parsing that strips inline comments after '#'
- Safe int/float parsing from ENV
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd


def _strip_inline_comment(s: str) -> str:
    """
    Strip inline comments starting with '#', unless inside quotes.
    Handles simple cases: 'value # comment' -> 'value'
    """
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
    Load .env file:
    - Supports 'export KEY=VALUE'
    - Strips inline comments after '#'
    - Does not override already-set environment variables
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
        # also allow values like "2   # comment"
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


# Load .env first
_load_dotenv_inline(Path(".env"))

# Read env using robust helpers
MODEL = env_str("APP_WHISPER_MODEL_PATH", "").strip()
LANG = env_str("APP_WHISPER_LANGUAGE", "de").strip()
THR = env_int("APP_WHISPER_THREADS", 4)
SR = env_int("APP_SAMPLE_RATE", 48000)
DUR = env_float("APP_CHECK_DURATION", 6.0)
WANT_JSON = env_str("MIC_CHECK_JSON", "0").lower() in {"1", "true", "yes"}
WAV_OUT = env_str("MIC_CHECK_SAVE_WAV", "").strip()

# Fallback to ./models/ggml-base.bin if not set
if not MODEL:
    candidate = Path("models/ggml-base.bin")
    MODEL = str(candidate) if candidate.is_file() else "ggml-base.bin"


def pick_input_device(prefer_name: Optional[str] = None) -> Tuple[int, str]:
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


def resample_to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if x.size == 0:
        return np.array([], dtype=np.float32)
    if sr == 16000:
        return x.astype(np.float32, copy=False)
    tgt = int(round(x.shape[0] * (16000.0 / float(sr))))
    if tgt <= 0:
        return np.array([], dtype=np.float32)
    t_in = np.linspace(0.0, 1.0, num=x.shape[0], endpoint=False, dtype=np.float64)
    t_out = np.linspace(0.0, 1.0, num=tgt, endpoint=False, dtype=np.float64)
    y = np.interp(t_out, t_in, x.astype(np.float64, copy=False))
    return y.astype(np.float32, copy=False)


def save_wav_int16_mono(path: Path, x: np.ndarray, sr: int) -> None:
    import wave
    x16 = np.clip(x, -1.0, 1.0)
    x16 = (x16 * 32767.0).astype(np.int16, copy=False)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(x16.tobytes())


def main() -> None:
    prefer = env_str("APP_AUDIO_DEVICE", "").strip() or None
    try:
        dev_index, dev_name = pick_input_device(prefer)
    except Exception as e:
        print(f"[ERR] Could not select an input device: {e}")
        if WANT_JSON:
            print(json.dumps({"ok": False, "stage": "pick_device", "error": str(e)}))
        sys.exit(2)

    print(f"[INFO] Using input device #{dev_index}: {dev_name}")
    if not Path(MODEL).is_file():
        msg = f"Whisper model not found: {MODEL}"
        print(f"[ERR] {msg}")
        if WANT_JSON:
            print(json.dumps({"ok": False, "stage": "model_check", "error": msg}))
        sys.exit(3)

    sd.default.device = (dev_index, None)
    print(f"[AUDIO] Recording @ {SR} Hz for {DUR:.1f}s … please speak clearly")
    try:
        audio = sd.rec(int(DUR * SR), samplerate=SR, channels=1, dtype="float32")
        sd.wait()
    except Exception as e:
        print(f"[ERR] Recording failed: {e}")
        if WANT_JSON:
            print(json.dumps({"ok": False, "stage": "record", "error": str(e)}))
        sys.exit(4)

    x = audio[:, 0] if audio.size else np.array([], dtype=np.float32)
    if x.size == 0:
        print("[ERR] Empty buffer recorded.")
        if WANT_JSON:
            print(json.dumps({"ok": False, "stage": "record", "error": "empty_buffer"}))
        sys.exit(5)

    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x * x)))
    print(f"[AUDIO] Peak={peak:.3f}, RMS={rms:.3f}, samples={x.shape[0]}")

    x16 = resample_to_16k(x, SR)
    if x16.size == 0 or float(np.max(np.abs(x16))) < 0.01:
        print("[ERR] Audio empty or too quiet after resampling. Increase mic gain / speak louder.")
        if WANT_JSON:
            print(json.dumps({"ok": False, "stage": "resample", "error": "too_quiet_or_empty"}))
        sys.exit(6)

    try:
        from pywhispercpp.model import Model as WhisperModel
    except Exception as e:
        print(f"[ERR] pywhispercpp import failed: {e}")
        if WANT_JSON:
            print(json.dumps({"ok": False, "stage": "import", "error": str(e)}))
        sys.exit(7)

    print("[WHISPER] Loading model …")
    try:
        model = WhisperModel(
            MODEL,
            n_threads=THR,
            print_realtime=False,
            print_progress=False,
            language=LANG,
            translate=False,
        )
    except Exception as e:
        print(f"[ERR] Model load failed: {e}")
        if WANT_JSON:
            print(json.dumps({"ok": False, "stage": "load_model", "error": str(e)}))
        sys.exit(8)

    print("[WHISPER] Transcribing …")
    try:
        if hasattr(model, "transcribe_float32"):
            out = model.transcribe_float32(x16)
        elif hasattr(model, "transcribe"):
            out = model.transcribe(x16)
        else:
            x16i = (np.clip(x16, -1.0, 1.0) * 32767.0).astype(np.int16, copy=False)
            out = model.transcribe_pcm16(x16i)
    except Exception as e:
        print(f"[ERR] Transcription failed: {e}")
        if WANT_JSON:
            print(json.dumps({"ok": False, "stage": "transcribe", "error": str(e)}))
        sys.exit(9)

    text = out if isinstance(out, str) else (out.get("text", "") if isinstance(out, dict) else str(out))
    text = (text or "").strip()
    print(f"[WHISPER] text: {text}")

    if WANT_JSON:
        print(json.dumps({
            "ok": True,
            "device": {"index": dev_index, "name": dev_name},
            "sr": SR,
            "duration": DUR,
            "peak": peak,
            "rms": rms,
            "text": text
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
