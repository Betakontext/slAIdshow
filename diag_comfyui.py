# diag_comfyui.py – Robuste Diagnose mit dynamischer Checkpoint-Erkennung
from __future__ import annotations
import os
import sys
import json
import time
import asyncio
from typing import Any, Dict, List, Optional

import httpx

COMFY_HOST = os.getenv("APP_COMFY_HOST", "127.0.0.1").strip() or "127.0.0.1"
COMFY_PORT = int(os.getenv("APP_COMFY_PORT", "8188"))
COMFY_BASE = f"http://{COMFY_HOST}:{COMFY_PORT}"

def timeout(total: float = 30.0) -> httpx.Timeout:
    total = max(5.0, min(total, 240.0))
    return httpx.Timeout(connect=5.0, read=total, write=5.0, pool=5.0)

async def get_models(client: httpx.AsyncClient) -> Dict[str, List[str]]:
    # Liefert ein Dict { "checkpoints": [..], "vae": [..], ... } soweit verfügbar
    try:
        r = await client.get(f"{COMFY_BASE}/models", timeout=timeout(30.0))
        r.raise_for_status()
        j = r.json()
        # Manche Builds erwarten /models?type=checkpoints, andere liefern ein dict mit Kategorien
        if isinstance(j, dict) and j:
            return {k: [it.get("name") if isinstance(it, dict) else str(it) for it in v] if isinstance(v, list) else [] for k, v in j.items()}
    except Exception:
        pass
    # Fallback: versuche checkpoints einzeln
    try:
        r = await client.get(f"{COMFY_BASE}/models?type=checkpoints", timeout=timeout(30.0))
        r.raise_for_status()
        j = r.json()
        if isinstance(j, dict) and "checkpoints" in j:
            ckpts = [m.get("name") for m in j.get("checkpoints", []) if isinstance(m, dict) and m.get("name")]
            return {"checkpoints": ckpts}
        if isinstance(j, list):
            return {"checkpoints": [str(x) for x in j]}
    except Exception:
        pass
    return {}

def choose_checkpoint(models_index: Dict[str, List[str]]) -> Optional[str]:
    ckpts = models_index.get("checkpoints", []) or []
    # Bevorzuge "anything-v4.5-pruned.safetensors", wenn vorhanden (aus deinem Dump)
    preferred = "anything-v4.5-pruned.safetensors"
    for name in ckpts:
        if name == preferred:
            return name
    # Sonst nimm den ersten vernünftigen .safetensors
    for name in ckpts:
        if isinstance(name, str) and (name.endswith(".safetensors") or name.endswith(".ckpt")):
            return name
    return None

async def main() -> int:
    print(f"[DIAG] Target: {COMFY_BASE}")
    async with httpx.AsyncClient(timeout=timeout(30.0)) as client:
        # 1) Health
        try:
            r = await client.get(f"{COMFY_BASE}/history")
            r.raise_for_status()
            print("[DIAG] /history OK")
        except Exception as e:
            print("[DIAG][ERR] /history failed:", e)
            return 1

        # 2) Modelle ermitteln
        models_index = await get_models(client)
        print("[DIAG] Models index keys:", list(models_index.keys()))
        ckpt_name = choose_checkpoint(models_index)
        if not ckpt_name:
            print("[DIAG][ERR] Kein Checkpoint gefunden. Bitte sorge dafür, dass unter ComfyUI/models/checkpoints ein Modell liegt.")
            return 2
        print("[DIAG] Using checkpoint:", ckpt_name)

        # 3) Minimal-Graph mit dynamischem ckpt_name
        prompt_text = "Test prompt from diag, small 128x128"
        api_graph: Dict[str, Any] = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt_name}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt_text, "clip": [1, 1]}},
            "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "low quality, blurry, bad anatomy", "clip": [1, 1]}},
            "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 128, "height": 128, "batch_size": 1}},
            "5": {"class_type": "KSampler", "inputs": {
                "seed": 1, "steps": 5, "cfg": 5.0, "sampler_name": "euler a",
                "model": [1, 0], "positive": [2, 0], "negative": [3, 0], "latent_image": [4, 0]
            }},
            "6": {"class_type": "VAEDecode", "inputs": {"samples": [5, 0], "vae": [1, 2]}},
            "7": {"class_type": "SaveImage", "inputs": {
                "filename_prefix": f"diag_{int(time.time())}",
                "images": [6, 0],
                "return_previews": True,
                "save_metadata": True,
                "embed_workflow": True
            }},
        }
        body = {"prompt": api_graph}
        try:
            r = await client.post(f"{COMFY_BASE}/prompt", json=body, timeout=timeout(120.0))
            r.raise_for_status()
            jr = r.json()
            pid = jr.get("prompt_id") or jr.get("id")
            print("[DIAG] /prompt OK, prompt_id:", pid)
        except Exception as e:
            print("[DIAG][ERR] /prompt failed:", e)
            print("[DIAG] Sent body preview:", json.dumps(body)[:600])
            return 3

        # 4) Poll
        deadline = time.time() + 180
        got_filename: Optional[str] = None
        while time.time() < deadline:
            await asyncio.sleep(1.0)
            try:
                s = await client.get(f"{COMFY_BASE}/prompt/{pid}", timeout=timeout(15.0))
                status = s.json()
                msg = (status.get("status") or {}).get("exec_info") or (status.get("status") or {}).get("status")
                if msg:
                    print("[DIAG] status:", msg)
            except Exception:
                pass
            try:
                h = await client.get(f"{COMFY_BASE}/history/{pid}", timeout=timeout(15.0))
                hist = h.json()
                entry = (hist.get(pid) if isinstance(hist, dict) else None) or hist
                # Rekursiv nach Filenames suchen
                def find_files(obj: Any, acc: List[str]):
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            if k == "filename" and isinstance(v, str):
                                acc.append(v)
                            else:
                                find_files(v, acc)
                    elif isinstance(obj, list):
                        for it in obj:
                            find_files(it, acc)
                files: List[str] = []
                find_files(entry, files)
                if files:
                    got_filename = files[0]
                    print("[DIAG] Found filename in history:", got_filename)
                    break
            except Exception:
                pass

        if not got_filename:
            print("[DIAG][WARN] No filename reported in history. Prüfe ComfyUI/output auf diag_*-Dateien.")
            return 4
        else:
            # 5) Download testen
            try:
                u = f"{COMFY_BASE}/view"
                q = {"filename": got_filename, "type": "output"}
                rr = await client.get(u, params=q, timeout=timeout(30.0))
                rr.raise_for_status()
                print(f"[DIAG] Download OK, bytes={len(rr.content)}")
                return 0
            except Exception as e:
                print("[DIAG][ERR] Download failed:", e)
                return 5

if __name__ == "__main__":
    ec = asyncio.run(main())
    sys.exit(ec)
