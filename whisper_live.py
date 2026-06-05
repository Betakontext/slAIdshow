#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional, List

import numpy as np
import sounddevice as sd

# ---- Config aus ENV ----
DEVICE_RAW = os.getenv("APP_AUDIO_DEVICE", None)
DEVICE = None if (DEVICE_RAW is None or DEVICE_RAW.strip() == "") else (int(DEVICE_RAW) if DEVICE_RAW.isdigit() else DEVICE_RAW)
SR_ENV = int(os.getenv("APP_SAMPLE_RATE", "48000"))
FRAME_MS = int(os.getenv("APP_FRAME_DURATION_MS", "20"))
SNAPSHOT_SEC = float(os.getenv("APP_SNAPSHOT_SEC", "5.0"))

WHISPER_MODEL_PATH = os.getenv("APP_WHISPER_MODEL_PATH", "").strip()
WHISPER_LANGUAGE = os.getenv("APP_WHISPER_LANGUAGE", "de").strip()
WHISPER_THREADS = int(os.getenv("APP_WHISPER_THREADS", "4"))
WHISPER_TEMPERATURE = float(os.getenv("APP_WHISPER_TEMPERATURE", "0.0"))

# ---- Whisper laden ----
try:
    from pywhispercpp.model import Model as WhisperModel
except Exception as e:
    print(f"[ERR] pywhispercpp import fehlgeschlagen: {e}")
    sys.exit(1)

def resample_to_16k(samples: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return samples.astype(np.float32, copy=False)
    target_len = int(samples.shape[0] * (16000.0 / float(sr)))
    if target_len <= 0:
        return np.array([], dtype=np.float32)
    return np.interp(
        np.linspace(0.0, 1.0, num=target_len, endpoint=False, dtype=np.float64),
        np.linspace(0.0, 1.0, num=samples.shape[0], endpoint=False, dtype=np.float64),
        samples.astype(np.float64, copy=False),
    ).astype(np.float32, copy=False)

def load_whisper(model_path: str) -> Optional[WhisperModel]:
    p = Path(model_path)
    if not p.is_file():
        print(f"[ERR] Whisper-Modell nicht gefunden: {p}")
        return None
    try:
        wm = WhisperModel(
            str(p),
            n_threads=WHISPER_THREADS,
            print_progress=False,
            print_realtime=False,
            language=WHISPER_LANGUAGE if WHISPER_LANGUAGE else None,
            translate=False,
            temperature=WHISPER_TEMPERATURE,
        )
        print(f"[OK] Whisper geladen: {p.name}, threads={WHISPER_THREADS}, lang={WHISPER_LANGUAGE}")
        return wm
    except Exception as e:
        print(f"[ERR] Whisper Init fehlgeschlagen: {e}")
        return None

def transcribe_float32(model: WhisperModel, audio_f32_16k: np.ndarray) -> str:
    if audio_f32_16k.size == 0:
        return ""
    try:
        if hasattr(model, "transcribe_float32"):
            txt = model.transcribe_float32(audio_f32_16k)
        elif hasattr(model, "transcribe"):
            txt = model.transcribe(audio_f32_16k)
        else:
            # Fallback auf PCM16
            x = np.clip(audio_f32_16k, -1.0, 1.0)
            x16 = (x * 32767.0).astype(np.int16, copy=False)
            txt = model.transcribe_pcm16(x16)
        if isinstance(txt, str):
            return txt.strip()
        if isinstance(txt, dict):
            return (txt.get("text") or "").strip()
        return str(txt).strip()
    except Exception as e:
        print(f"[ERR] Transkription fehlgeschlagen: {e}")
        return ""

def open_stream_with_fallback(device, frame_ms: int):
    candidates = [SR_ENV]
    for sr in (48000, 44100, 32000, 16000):
        if sr not in candidates:
            candidates.append(sr)
    last_err = None
    for sr in candidates:
        frame_samples = int(sr * frame_ms / 1000)
        try:
            stream = sd.InputStream(
                samplerate=sr,
                channels=1,
                dtype="float32",
                device=device,
                blocksize=frame_samples,
            )
            print(f"[AUDIO] device={device!r} accepted sr={sr}")
            return stream, sr, frame_samples
        except Exception as e:
            print(f"[AUDIO] device={device!r} rejected sr={sr}: {e}")
            last_err = e
    raise RuntimeError(f"Kein unterstütztes sr für device={device!r}: {last_err}")

def main() -> int:
    if not WHISPER_MODEL_PATH:
        print("[ERR] APP_WHISPER_MODEL_PATH ist leer")
        return 1
    model = load_whisper(WHISPER_MODEL_PATH)
    if model is None:
        return 1

    try:
        stream, sr, frame_samples = open_stream_with_fallback(DEVICE, FRAME_MS)
    except Exception as e:
        print(f"[ERR] Audio-Stream öffnen fehlgeschlagen: {e}")
        return 1

    print("[INFO] Starte Aufnahme. Sprich einen Satz. Drücke STRG+C zum Beenden.")
    buf: List[np.ndarray] = []
    last_snap = time.time()
    total_frames = 0

    def cb(indata, frames, timeinfo, status):
        mono = indata[:, 0].copy()
        buf.append(mono)

    stream.callback = cb
    with stream:
        try:
            while True:
                time.sleep(FRAME_MS / 1000.0)
                if not buf:
                    continue
                frame = buf.pop(0)
                total_frames += 1

                # Alle SNAPSHOT_SEC ein Fenster transkribieren
                if (time.time() - last_snap) >= SNAPSHOT_SEC:
                    last_snap = time.time()
                    audio = np.frombuffer(b"".join([x.tobytes() for x in buf]), dtype=np.float32)
                    buf.clear()
                    print(f"[SNAP] frames={total_frames}, samples={audio.shape[0]}, sr={sr}")
                    audio16 = resample_to_16k(audio, sr)
                    if audio16.size == 0 or float(np.max(np.abs(audio16))) < 0.01:
                        print("[WHISPER] leer/zu leise")
                        continue
                    txt = transcribe_float32(model, audio16)
                    if txt:
                        print(f"[WHISPER] text: {txt}")
                    else:
                        print("[WHISPER] (kein Text)")
        except KeyboardInterrupt:
            print("\n[INFO] Beendet.")
            return 0

if __name__ == "__main__":
    raise SystemExit(main())
