#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
verify_runtime.py
Vergleicht die kritischen Audio/Whisper-Parameter zwischen deiner UI (FastAPI-App)
und deinem Testskript-Setup, führt einen 3s-Probe-Record durch und gibt
empfohlene .env-Werte aus, um die UI auf das Testverhalten auszurichten.
"""

import os
from pathlib import Path
import json
import numpy as np
import sounddevice as sd

def env_str(k: str, d: str) -> str:
    v = os.getenv(k)
    return v.strip() if v is not None else d

def env_int(k: str, d: int) -> int:
    v = os.getenv(k)
    try: return int(v) if v is not None else d
    except: return d

def env_float(k: str, d: float) -> float:
    v = os.getenv(k)
    try: return float(v) if v is not None else d
    except: return d

def resample_to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return x.astype(np.float32, copy=False)
    tgt = int(x.shape[0] * (16000.0/float(sr)))
    if tgt <= 0:
        return np.array([], dtype=np.float32)
    t_in  = np.linspace(0.0, 1.0, num=x.shape[0], endpoint=False, dtype=np.float64)
    t_out = np.linspace(0.0, 1.0, num=tgt,        endpoint=False, dtype=np.float64)
    return np.interp(t_out, t_in, x.astype(np.float64, copy=False)).astype(np.float32, copy=False)

def main() -> None:
    # Lade .env ad-hoc, falls noch nicht exportiert
    envp = Path(".env")
    if envp.is_file():
        for line in envp.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k,v = s.split("=",1)
            os.environ.setdefault(k.strip(), v.strip().strip("'").strip('"'))

    # Sammle kritische Parameter
    cfg = {
        "APP_AUDIO_DEVICE": env_str("APP_AUDIO_DEVICE",""),
        "APP_SAMPLE_RATE": env_int("APP_SAMPLE_RATE", 48000),
        "APP_FRAME_DURATION_MS": env_int("APP_FRAME_DURATION_MS", 20),
        "APP_DISABLE_VAD": env_int("APP_DISABLE_VAD", 0),
        "APP_RMS_VAD_THRESHOLD": env_float("APP_RMS_VAD_THRESHOLD", 0.035),
        "APP_SNAPSHOT_SEC": env_float("APP_SNAPSHOT_SEC", 3.5),
        "APP_MAX_SILENCE_MS": env_int("APP_MAX_SILENCE_MS", 300),
        "APP_WHISPER_MODEL_PATH": env_str("APP_WHISPER_MODEL_PATH",""),
        "APP_WHISPER_LANGUAGE": env_str("APP_WHISPER_LANGUAGE","de"),
        "APP_WHISPER_THREADS": env_int("APP_WHISPER_THREADS",4),
        "APP_WHISPER_TEMPERATURE": env_float("APP_WHISPER_TEMPERATURE",0.0),
    }

    print("[RUNTIME] Aktuelle .env/ENV:")
    for k,v in cfg.items():
        print(f"  - {k} = {v}")

    # Sanity: Modell vorhanden?
    mp = Path(cfg["APP_WHISPER_MODEL_PATH"])
    if not mp.is_file():
        print(f"[ERR] Modell nicht gefunden: {mp}")
        return

    # 3s Probeaufnahme mit exakt dem konfigurierten Device/SR
    prefer = cfg["APP_AUDIO_DEVICE"] or None

    def pick_input_device(prefer):
        devs = sd.query_devices()
        if prefer:
            for i,d in enumerate(devs):
                if d.get("name")==prefer and d.get("max_input_channels",0)>0:
                    return i
            pl = prefer.lower()
            for i,d in enumerate(devs):
                if pl in (d.get("name","").lower()) and d.get("max_input_channels",0)>0:
                    return i
        for i,d in enumerate(devs):
            nm = (d.get("name") or "").lower()
            if ("pulse" in nm or "pipewire" in nm) and d.get("max_input_channels",0)>0:
                return i
        for i,d in enumerate(devs):
            if d.get("max_input_channels",0)>0:
                return i
        raise RuntimeError("Kein Input-Device gefunden")

    idx = pick_input_device(prefer)
    name = sd.query_devices()[idx]["name"]
    sd.default.device = (idx, None)
    sr = cfg["APP_SAMPLE_RATE"]

    print(f"[AUDIO] Probeaufnahme 3s @ {sr} Hz, Device {idx}:{name!r}")
    rec = sd.rec(int(3.0*sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    x = rec[:,0]
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    rms  = float(np.sqrt(np.mean(x*x))) if x.size else 0.0
    print(f"[AUDIO] Peak={peak:.3f} RMS={rms:.3f} samples={x.shape[0]}")

    if peak < 0.01:
        print("[HINT] Signal sehr leise. Erhöhe Mikrofon-Pegel oder senke APP_RMS_VAD_THRESHOLD (z.B. 0.02–0.03).")

    x16 = resample_to_16k(x, sr)
    if x16.size == 0:
        print("[ERR] Resampling ergab leeren Puffer.")
        return

    print("\n[RECOMMEND] Setze diese Werte, um UI ≈ Testskript zu erhalten:")
    recs = {
        "APP_DISABLE_VAD": 0,
        "APP_RMS_VAD_THRESHOLD": 0.035 if peak >= 0.1 else 0.02,
        "APP_SNAPSHOT_SEC": 3.0,
        "APP_MAX_SILENCE_MS": 300,
        "APP_FRAME_DURATION_MS": 20,
        "APP_WHISPER_TEMPERATURE": 0.0,
    }
    for k,v in recs.items():
        print(f"  {k}={v}")

    print("\n[CHECKLISTE]")
    print("  1) UI-Log muss zeigen: '[AUDIO] open OK' und KEIN 'switching to polling read() loop'")
    print("  2) VAD aktiv (disable_vad=False/webrtc_vad=True falls installiert)")
    print("  3) '[WHISPER] text:' nur für Segmente mit Peak>=0.01; leise Frames filtern")
    print("  4) SNAPSHOT_SEC 3.0–3.5; erstes Snapshot-Deadline ~1.0–1.2s")

if __name__ == "__main__":
    main()
