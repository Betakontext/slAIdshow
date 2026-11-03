# hardware_detect.py


import json
import subprocess
import platform
from pathlib import Path
from dataclasses import dataclass, asdict

OUTPUT_PATH = Path("/app/config/hardware.json")


@dataclass
class HardwareProfile:
    vram_gb: float
    gpu_name: str
    gpu_count: int
    cpu_ram_gb: float
    os: str
    is_apple_silicon: bool


def detect_nvidia() -> list[HardwareProfile]:
    """Via nvidia-ml-py (import: pynvml) – die aktuell gepflegte Library."""
    import pynvml
    pynvml.nvmlInit()
    profiles = []
    try:
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle).decode()
            profiles.append(HardwareProfile(
                vram_gb=mem.total / (1024**3),
                gpu_name=name,
                gpu_count=count,
                cpu_ram_gb=0,  # separat ermitteln
                os=platform.system(),
                is_apple_silicon=False,
            ))
    finally:
        pynvml.nvmlShutdown()
    return profiles


def detect_cpu_ram() -> float:
    """System-RAM in GB (via /proc/meminfo oder psutil)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024**2)
    except FileNotFoundError:
        pass
    # Fallback: psutil
    import psutil
    return psutil.virtual_memory().total / (1024**3)


def detect_apple_silicon() -> bool:
    """Macs erkennen (keine NVIDIA-GPU, aber unified memory)."""
    if platform.system() != "Darwin":
        return False
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5
        )
        return "Apple" in result.stdout
    except Exception:
        return False


def detect() -> dict:
    """Hauptfunktion – gibt das finale hardware.json zurück."""
    is_macs = detect_apple_silicon()

    if is_macs:
        # Apple Silicon: unified memory = "VRAM"
        ram = detect_cpu_ram()
        # Bei Macs ist ~75% des RAMs für GPU verfügbar
        vram = ram * 0.75
        profile = {
            "vram_gb": round(vram, 1),
            "gpu_name": "Apple Silicon (unified)",
            "gpu_count": 1,
            "cpu_ram_gb": round(ram, 1),
            "os": "Darwin",
            "is_apple_silicon": True,
        }
    else:
        profiles = detect_nvidia()
        if not profiles:
            # Kein GPU – CPU-only Fallback
            profile = {
                "vram_gb": 0,
                "gpu_name": "None (CPU-only)",
                "gpu_count": 0,
                "cpu_ram_gb": round(detect_cpu_ram(), 1),
                "os": platform.system(),
                "is_apple_silicon": False,
            }
        else:
            # Beste GPU auswählen (meist nur eine)
            best = max(profiles, key=lambda p: p.vram_gb)
            profile = asdict(best)
            profile["cpu_ram_gb"] = round(detect_cpu_ram(), 1)

    # Persistieren
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(profile, indent=2))
    print(f"✅ Hardware-Profil geschrieben: {OUTPUT_PATH}")
    print(f"   VRAM: {profile['vram_gb']} GB | GPU: {profile['gpu_name']}")
    return profile


if __name__ == "__main__":
    detect()
Tests: tests/test_hardware_detect.py
"""Tests mit Mocks – keine GPU im CI nötig."""
from unittest.mock import patch, MagicMock
import json
from pathlib import Path


def test_detect_no_gpu():
    with patch("slaidshow.hardware_detect.detect_nvidia", return_value=[]), \
         patch("slaidshow.hardware_detect.detect_cpu_ram", return_value=16.0), \
         patch("slaidshow.hardware_detect.detect_apple_silicon", return_value=False):
        result = detect()
    assert result["vram_gb"] == 0
    assert result["gpu_count"] == 0


def test_detect_6gb_gpu():
    mock_profile = HardwareProfile(
        vram_gb=6.0, gpu_name="NVIDIA GeForce RTX 3050",
        gpu_count=1, cpu_ram_gb=16.0, os="Linux", is_apple_silicon=False
    )
    with patch("slaidshow.hardware_detect.detect_nvidia", return_value=[mock_profile]), \
         patch("slaidshow.hardware_detect.detect_cpu_ram", return_value=16.0), \
         patch("slaidshow.hardware_detect.detect_apple_silicon", return_value=False):
        result = detect()
    assert result["vram_gb"] == 6.0
    assert "RTX 3050" in result["gpu_name"]


def test_detect_apple_silicon():
    with patch("slaidshow.hardware_detect.detect_apple_silicon", return_value=True), \
         patch("slaidshow.hardware_detect.detect_cpu_ram", return_value=32.0):
        result = detect()
    assert result["is_apple_silicon"] is True
    assert result["vram_gb"] == 22.4  # 32 * 0.75

Phase 0: config_generator.py (Tag 2)
Datei: slaidshow/config_generator.py
"""
Generiert die optimale Konfiguration basierend auf VRAM.
Input:  /app/config/hardware.json
Output: /app/config/generated.yaml
"""

import json
import yaml
from pathlib import Path

HARDWARE_PATH = Path("/app/config/hardware.json")
OUTPUT_PATH = Path("/app/config/generated.yaml")

# ─── VRAM-Tiers ────────────────────────────────────────────────────────

TIERS = {
    "6gb": {
        "min_vram": 6.0,
        "max_vram": 7.9,
        "stt": {
            "model": "parakeet-tdt-0.6b-v3",
            "device": "cpu",
            "backend": "faster-whisper",
        },
        "llm": {
            "model": "qwen3:1.7b-q4_K_M",
            "device": "cuda",
            "context": 4096,
            "vram_budget_mb": 2560,
        },
        "image": {
            "base_model": "sd_15.safetensors",
            "lora": "lcm-lora-sdv1-5.safetensors",
            "steps": 4,
            "cfg": 1.0,
            "resolution": [512, 512],
            "device": "cuda",
            "vram_budget_mb": 3584,
            "unload_after_generate": True,
        },
        "features": {
            "in_image_text": False,      # HTML-Overlay statt
            "multi_model": False,        # nur 1 Bildmodell
            "parallel_inference": False, # sequenziell
            "styles": ["standard", "sketch"],
        },
        "expected_latency_s": 2.5,
    },
    "8gb": {
        "min_vram": 8.0,
        "max_vram": 11.9,
        "stt": {
            "model": "parakeet-tdt-0.6b-v3",
            "device": "cuda",
            "backend": "faster-whisper",
        },
        "llm": {
            "model": "qwen3:4b-q4_K_M",
            "device": "cuda",
            "context": 8192,
            "vram_budget_mb": 3072,
        },
        "image": {
            "base_model": "sd_15.safetensors",
            "lora": "lcm-lora-sdv1-5.safetensors",
            "steps": 4,
            "cfg": 1.0,
            "resolution": [768, 768],
            "device": "cuda",
            "vram_budget_mb": 4096,
            "unload_after_generate": False,
        },
        "features": {
            "in_image_text": False,
            "multi_model": False,
            "parallel_inference": False,
            "styles": ["standard", "sketch", "watercolor"],
        },
        "expected_latency_s": 1.8,
    },
    "16gb": {
        "min_vram": 12.0,
        "max_vram": 15.9,
        "stt": {
            "model": "parakeet-tdt-0.6b-v3",
            "device": "cuda",
            "backend": "faster-whisper",
        },
        "llm": {
            "model": "gemma4:26b-a4b-q4_K_M",
            "device": "cuda",
            "context": 16384,
            "vram_budget_mb": 14336,
        },
        "image": {
            "base_model": "flux2_klein_4b_fp8.safetensors",
            "lora": None,
            "steps": 4,
            "cfg": 1.0,
            "resolution": [1024, 1024],
            "device": "cuda",
            "vram_budget_mb": 8192,
            "unload_after_generate": True,
        },
        "features": {
            "in_image_text": False,
            "multi_model": True,
            "parallel_inference": True,
            "styles": ["standard", "photoreal", "sketch", "watercolor", "3d"],
        },
        "expected_latency_s": 1.2,
    },
    "24gb": {
        "min_vram": 16.0,
        "max_vram": 999.0,
        "stt": {
            "model": "parakeet-tdt-0.6b-v3",
            "device": "cuda",
            "backend": "faster-whisper",
        },
        "llm": {
            "model": "qwen3.8:27b-q4_K_M",
            "device": "cuda",
            "context": 32768,
            "vram_budget_mb": 18432,
        },
        "image": {
            "base_model": "flux2_klein_4b_fp8.safetensors",
            "lora": None,
            "steps": 4,
            "cfg": 1.0,
            "resolution": [1024, 1024],
            "device": "cuda",
            "vram_budget_mb": 12288,
            "unload_after_generate": False,
        },
        "features": {
            "in_image_text": True,       # Qwen-Image 2.1 als 2. Modell
            "multi_model": True,
            "parallel_inference": True,
            "styles": ["standard", "photoreal", "sketch", "watercolor", "3d", "text"],
        },
        "expected_latency_s": 0.8,
    },
}


def select_tier(vram_gb: float) -> str:
    """Wähle das passende Tier basierend auf verfügbarem VRAM."""
    if vram_gb < 6.0:
        raise ValueError(
            f"VRAM {vram_gb} GB ist zu wenig (Minimum: 6 GB). "
            f"Bitte GPU mit mind. 6 GB VRAM verwenden."
        )
    for tier_name, tier in TIERS.items():
        if tier["min_vram"] <= vram_gb <= tier["max_vram"]:
            return tier_name
    # Fallback: höchstes erreichbares Tier
    return "24gb"


def generate(hardware: dict) -> dict:
    """Generiere die finale Konfiguration."""
    vram = hardware["vram_gb"]
    tier_name = select_tier(vram)
    tier = TIERS[tier_name]

    config = {
        "profile": tier_name,
        "hardware": hardware,
        "stt": tier["stt"],
        "llm": tier["llm"],
        "image": tier["image"],
        "features": tier["features"],
        "orchestration": {
            "parallel": tier["features"]["parallel_inference"],
            "vram_scheduler": not tier["features"]["parallel_inference"],
            "max_concurrent_generations": 1 if vram < 12 else 2,
        },
        "expected_latency_s": tier["expected_latency_s"],
    }

    # Persistieren
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(yaml.dump(config, default_flow_style=False, allow_unicode=True))
    print(f"✅ Config generiert: {OUTPUT_PATH} (Tier: {tier_name})")
    print(f"   STT: {tier['stt']['model']} on {tier['stt']['device']}")
    print(f"   LLM: {tier['llm']['model']} on {tier['llm']['device']}")
    print(f"   Bild: {tier['image']['base_model']} ({tier['image']['steps']} steps)")
    return config
