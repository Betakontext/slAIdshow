#!/usr/bin/env python3
from __future__ import annotations
import os
import numpy as np
import sounddevice as sd

DEVICE = os.getenv("APP_AUDIO_DEVICE", "pipewire")
SR = int(os.getenv("APP_SAMPLE_RATE", "16000"))
DUR = float(os.getenv("APP_TEST_DURATION", "3.0"))

def main() -> None:
    print(f"Teste Mikrofon: device='{DEVICE}', sr={SR}, duration={DUR}s")
    print("Bitte sprechen …")
    audio = sd.rec(int(DUR * SR), samplerate=SR, channels=1, dtype='float32', device=DEVICE)
    sd.wait()
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    print(f"Peak={peak:.3f}, RMS={rms:.3f}")
    if peak < 0.05:
        print("Warnung: Sehr niedriger Pegel. Prüfe pavucontrol / Mic-Gain.")
    else:
        print("OK: Pegel ausreichend.")

if __name__ == "__main__":
    main()
