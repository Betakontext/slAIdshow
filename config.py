from __future__ import annotations
import os
from pathlib import Path
from typing import Final

def env_str(key: str, default: str) -> str:
    return os.getenv(key, default)

def env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default

def env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default

# Audio
APP_AUDIO_DEVICE: Final[str] = env_str("APP_AUDIO_DEVICE", "pipewire")
APP_SAMPLE_RATE: Final[int] = env_int("APP_SAMPLE_RATE", 16000)
APP_FRAME_DURATION_MS: Final[int] = env_int("APP_FRAME_DURATION_MS", 20)
APP_VAD_AGGRESSIVENESS: Final[int] = env_int("APP_VAD_AGGRESSIVENESS", 0)
APP_MAX_SILENCE_MS: Final[int] = env_int("APP_MAX_SILENCE_MS", 300)
APP_SNAPSHOT_SEC: Final[float] = env_float("APP_SNAPSHOT_SEC", 7.0)

# Ollama
APP_OLLAMA_HOST: Final[str] = env_str("APP_OLLAMA_HOST", "127.0.0.1")
APP_OLLAMA_PORT: Final[int] = env_int("APP_OLLAMA_PORT", 11434)
APP_OLLAMA_MODEL: Final[str] = env_str("APP_OLLAMA_MODEL", "phi3:mini")
APP_OLLAMA_TEMPERATURE: Final[float] = env_float("APP_OLLAMA_TEMPERATURE", 0.2)

# ComfyUI
APP_COMFY_HOST: Final[str] = env_str("APP_COMFY_HOST", "127.0.0.1")
APP_COMFY_PORT: Final[int] = env_int("APP_COMFY_PORT", 8188)

# Output
APP_OUTPUT_DIR: Final[Path] = Path(env_str("APP_OUTPUT_DIR", "./outputs/images")).resolve()
APP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def assert_local_url(url: str) -> None:
    assert url.startswith("http://127.0.0.1:"), f"Nur localhost erlaubt: {url}"
