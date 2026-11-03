# model_downloader.py
import yaml
import subprocess
from pathlib import Path

CONFIG_PATH = Path("/app/config/generated.yaml")
MODELS_DIR = Path("/app/models")


def download_models(config: dict) -> None:
    """Lädt basierend auf Config die benötigten Modelle."""

    # 1. Ollama-Modell
    llm_model = config["llm"]["model"]
    print(f"📥 Lade Ollama-Modell: {llm_model}")
    subprocess.run(
        ["ollama", "pull", llm_model],
        check=True, capture_output=True
    )

    # 2. STT-Modell (faster-whisper / parakeet)
    stt_model = config["stt"]["model"]
    stt_dir = MODELS_DIR / "stt" / stt_model
    if not stt_dir.exists():
        print(f"📥 Lade STT-Modell: {stt_model}")
        subprocess.run(
            ["huggingface-cli", "download", stt_model, "--local-dir", str(stt_dir)],
            check=True, capture_output=True
        )

    # 3. ComfyUI-Modelle
    image_cfg = config["image"]
    comfy_models = MODELS_DIR / "comfyui"

    # Base model
    base = image_cfg["base_model"]
    base_path = comfy_models / "checkpoints" / base
    if not base_path.exists():
        print(f"📥 Lade ComfyUI Base: {base}")
        _download_comfyui_model(base, "checkpoints")

    # LoRA (falls vorhanden)
    lora = image_cfg.get("lora")
    if lora:
        lora_path = comfy_models / "loras" / lora
        if not lora_path.exists():
            print(f"📥 Lade ComfyUI LoRA: {lora}")
            _download_comfyui_model(lora, "loras")

    # 4. Optional: 2. Bildmodell für in_image_text (nur bei 24GB+)
    if config["features"].get("in_image_text"):
        text_model = "qwen_image_2_1_fp8.safetensors"
        text_path = comfy_models / "checkpoints" / text_model
        if not text_path.exists():
            print(f"📥 Lade Text-Modell: {text_model}")
            _download_comfyui_model(text_model, "checkpoints")

    print("✅ Alle Modelle geladen.")


def _download_comfyui_model(filename: str, subfolder: str) -> None:
    """Download via huggingface-cli (URL je nach Modell)."""
    url_map = {
        "sd_15.safetensors": "stabilityai/sd-base-v1-5",
        "lcm-lora-sdv1-5.safetensors": "latent-consistency/lcm-lora-sdv1-5",
        "flux2_klein_4b_fp8.safetensors": "black-forest-labs/flux2-klein-4b",
        "qwen_image_2_1_fp8.safetensors": "Qwen/Qwen-Image-2.1",
    }
    repo = url_map.get(filename, filename)
    target = MODELS_DIR / "comfyui" / subfolder
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["huggingface-cli", "download", repo, filename,
         "--local-dir", str(target)],
        check=True, capture_output=True
    )
