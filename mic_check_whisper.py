#!/usr/bin/env python3
import os, sys, json, time
from pathlib import Path
import numpy as np
import sounddevice as sd

MODEL = os.getenv("APP_WHISPER_MODEL_PATH", "").strip()
LANG  = os.getenv("APP_WHISPER_LANGUAGE", "de").strip()
THR   = int(os.getenv("APP_WHISPER_THREADS","4"))
SR    = int(os.getenv("APP_SAMPLE_RATE","48000"))
DUR   = float(os.getenv("APP_CHECK_DURATION","6.0"))

def pick_input_device(prefer: str | None = None) -> int:
    devs = sd.query_devices()
    hostapis = sd.query_hostapis()
    # 1) Name-Match (z. B. "pulse" oder konkreter Name)
    if prefer:
        for i, d in enumerate(devs):
            if d.get("name") == prefer and d.get("max_input_channels",0) > 0:
                return i
        # Substring-Match
        for i, d in enumerate(devs):
            if prefer.lower() in d.get("name","").lower() and d.get("max_input_channels",0) > 0:
                return i
    # 2) Pulse bevorzugen
    for i, d in enumerate(devs):
        if "pulse" in d.get("name","").lower() and d.get("max_input_channels",0) > 0:
            return i
    # 3) Erstes echtes Input-Device
    for i, d in enumerate(devs):
        if d.get("max_input_channels",0) > 0:
            return i
    raise SystemExit("[ERR] Kein Eingabegerät mit max_input_channels>0 gefunden.")

def resample_to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return x.astype(np.float32, copy=False)
    tgt = int(x.shape[0] * (16000.0/float(sr)))
    if tgt <= 0:
        return np.array([], dtype=np.float32)
    t_in  = np.linspace(0.0, 1.0, num=x.shape[0], endpoint=False, dtype=np.float64)
    t_out = np.linspace(0.0, 1.0, num=tgt,        endpoint=False, dtype=np.float64)
    return np.interp(t_out, t_in, x.astype(np.float64, copy=False)).astype(np.float32, copy=False)

def main():
    prefer = os.getenv("APP_AUDIO_DEVICE", "").strip() or None
    dev_index = pick_input_device(prefer)
    dev = sd.query_devices()[dev_index]
    print(f"[INFO] Verwende Input-Device #{dev_index}: {dev['name']} (max_in_ch={dev['max_input_channels']})")

    if not Path(MODEL).is_file():
        raise SystemExit(f"[ERR] Modell nicht gefunden: {MODEL}")

    # WICHTIG: Für Aufnahme muss das Eingabegerät im ERSTEN Tupelteil stehen
    sd.default.device = (dev_index, None)

    print(f"[AUDIO] Aufnahme @ {SR} Hz, device_index={dev_index}. Bitte {DUR:.0f}s klar sprechen …")
    audio = sd.rec(int(DUR*SR), samplerate=SR, channels=1, dtype='float32')
    sd.wait()
    x = audio[:,0]
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x*x)))
    print(f"[AUDIO] Peak={peak:.3f}, RMS={rms:.3f}, samples={x.shape[0]}")

    x16 = resample_to_16k(x, SR)
    if x16.size == 0 or float(np.max(np.abs(x16))) < 0.01:
        raise SystemExit("[ERR] Audio leer oder zu leise. Bitte lauter sprechen / Mikro-Pegel prüfen.")

    try:
        from pywhispercpp.model import Model as WhisperModel
    except Exception as e:
        raise SystemExit(f"[ERR] pywhispercpp import fehlgeschlagen: {e}")

    print("[WHISPER] Lade Modell …")
    model = WhisperModel(MODEL, n_threads=THR, print_realtime=False, print_progress=False, language=LANG, translate=False)
    print("[WHISPER] Transkribiere …")
    if hasattr(model, "transcribe_float32"):
        out = model.transcribe_float32(x16)
    elif hasattr(model, "transcribe"):
        out = model.transcribe(x16)
    else:
        x16i = (np.clip(x16, -1.0, 1.0) * 32767.0).astype(np.int16, copy=False)
        out = model.transcribe_pcm16(x16i)

    text = out if isinstance(out, str) else (out.get("text","") if isinstance(out, dict) else str(out))
    print(f"[WHISPER] text: {text.strip()}")

if __name__ == "__main__":
    main()
