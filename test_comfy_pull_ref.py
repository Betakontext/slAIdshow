# test_comfy_pullref.py
# -*- coding: utf-8 -*-
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import httpx

from comfyui_bridge import generate_from_prompt_dict

def env_str(k: str, d: str = "") -> str:
    return (os.getenv(k, d) or "").strip()

def env_int(k: str, d: int) -> int:
    try:
        return int(env_str(k, str(d)))
    except Exception:
        return d

async def pull_ref_to_comfy_input(comfy_host: str, pull_port: int, ref_url: str) -> str:
    """
    Ruft den Pull-Service auf dem ComfyUI-Host auf und liefert den Basename zurück.
    """
    endpoint = f"http://{comfy_host}:{pull_port}/pull_from_url"
    params = {"url": ref_url}
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(endpoint, params=params)
        r.raise_for_status()
        data = r.json()
        if "basename" not in data:
            raise RuntimeError(f"pull_ref failed: {data}")
        return data["basename"]

def patch_loadimage_basename(prompt: Dict[str, Any], node_id: str, image_key: str, basename: str) -> Dict[str, Any]:
    """
    Setzt bei Node `node_id` das Input-Feld `image_key` auf `basename`.
    Unterstützt String oder Objekt-Formate für den LoadImage-Node.
    """
    node = prompt.get(node_id)
    if not node or "inputs" not in node:
        raise KeyError(f"Node {node_id} not found in prompt")
    # Einige Workflows erwarten {"image": "file.png"}, andere "image": "file.png"
    val = node["inputs"].get(image_key)
    if isinstance(val, dict):
        node["inputs"][image_key]["image"] = basename
    else:
        node["inputs"][image_key] = basename
    return prompt

async def main() -> None:
    comfy_host = env_str("APP_COMFY_HOST", "192.168.188.24")
    comfy_port = env_int("APP_COMFY_PORT", 8188)
    pull_port = env_int("APP_COMFY_PULL_PORT", 8190)  # Port des pull_ref_server.py

    # URL, unter der der ComfyUI-Host die Referenz vom App-Server abrufen kann:
    app_server_host = env_str("APP_SERVER_HOST", "192.168.188.10")
    app_server_port = env_int("APP_SERVER_PORT", 8080)
    ref_basename = env_str("APP_REF_BASENAME", "PLACEHOLDER.png")  # z. B. "donald-duck.webp"
    ref_url = f"http://{app_server_host}:{app_server_port}/ref/{ref_basename}"

    workflow_path = Path(env_str("APP_COMFY_WORKFLOW", "./workflows/text2img_SD15-FP16_fileRef.json")).resolve()
    out_dir = Path(env_str("APP_OUTPUT_DIR", "./outputs/images")).resolve()
    node_id = env_str("APP_COMFY_NODE_REF_IMAGE", "8")
    image_key = env_str("APP_COMFY_KEY_REF_IMAGE_PATH", "image")

    if not workflow_path.is_file():
        raise FileNotFoundError(f"Workflow fehlt: {workflow_path}")

    # 1) Referenz vom App-Server auf dem ComfyUI-Host ablegen
    basename = await pull_ref_to_comfy_input(comfy_host, pull_port, ref_url)
    print(f"[PULL] saved on ComfyUI host input/: {basename}")

    # 2) Workflow laden und Node 8 patchen
    body: Dict[str, Any] = json.loads(workflow_path.read_text(encoding="utf-8"))
    body = patch_loadimage_basename(body, node_id=node_id, image_key=image_key, basename=basename)

    # 3) Generieren
    images: List[Path] = await generate_from_prompt_dict(
        body,
        out_dir=out_dir,
        positive_text=env_str("APP_TEST_POSITIVE", "Kinder spielen im Schwimmbad an der Küste; golden hour, detailreich"),
        negative_text=env_str("APP_TEST_NEGATIVE", "text, watermark, logo, low quality, blurry, bad anatomy"),
        width=env_int("APP_TEST_WIDTH", 800),
        height=env_int("APP_TEST_HEIGHT", 600),
        steps=env_int("APP_TEST_STEPS", 20),
        cfg=float(env_str("APP_TEST_CFG", "8.0")),
        sampler_name=env_str("APP_TEST_SAMPLER", "euler"),
        denoise=float(env_str("APP_TEST_DENOISE", "0.52")),
        host=comfy_host,
        port=comfy_port,
        max_wait_sec=float(env_str("APP_TEST_MAX_WAIT", "180")),
    )
    print("Saved:", [str(p) for p in images])

if __name__ == "__main__":
    asyncio.run(main())
