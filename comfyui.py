#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# ---- ENV helpers (leichtgewichtig, analog zu app.py) ----
def _env_str(k: str, d: str) -> str:
    return (os.getenv(k, d) or "").strip()

def _env_int(k: str, d: int) -> int:
    try:
        return int(os.getenv(k, str(d)))
    except Exception:
        return d

def _env_bool01(k: str, d: int = 0) -> bool:
    v = (os.getenv(k, str(d)) or "").strip().lower()
    return v in {"1", "true", "yes", "on"}

def _env_float(k: str, d: float) -> float:
    try:
        return float(os.getenv(k, str(d)))
    except Exception:
        return d

# ---- Config ----
COMFY_HOST = _env_str("APP_COMFY_HOST", "127.0.0.1")
COMFY_PORT = _env_int("APP_COMFY_PORT", 8188)
COMFY_TIMEOUT_SEC = _env_float("APP_COMFY_TIMEOUT_SEC", 120.0)
COMFY_MAX_RETRIES = _env_int("APP_COMFY_MAX_RETRIES", 5)
COMFY_RETRY_BASE_DELAY = _env_float("APP_COMFY_RETRY_BASE_DELAY", 1.0)
COMFY_DISABLE = _env_bool01("APP_DISABLE_COMFYUI", 1)

# Ausgabe-Verzeichnis ggf. aus app.py-ENV übernehmen
OUTPUT_DIR = Path(_env_str("APP_OUTPUT_DIR", "./outputs/images")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Sicherheits-Check: nur localhost zulassen ----
def assert_local(host: str) -> None:
    if host != "127.0.0.1":
        raise AssertionError(f"Only localhost allowed for ComfyUI, got {host}")

assert_local(COMFY_HOST)

# ---- HTTP Utils ----
def _limits() -> httpx.Limits:
    return httpx.Limits(max_keepalive_connections=6, max_connections=12, keepalive_expiry=20.0)

def _timeout() -> httpx.Timeout:
    t = max(30.0, min(float(COMFY_TIMEOUT_SEC), 300.0))
    return httpx.Timeout(connect=5.0, read=t, write=5.0, pool=5.0)

def _is_retryable(e: Exception) -> bool:
    if isinstance(e, (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError)):
        return True
    if isinstance(e, httpx.HTTPStatusError):
        code = e.response.status_code if e.response else None
        return code in (408, 409, 429, 500, 502, 503, 504)
    return False

def _comfy_url(path: str) -> str:
    return f"http://{COMFY_HOST}:{COMFY_PORT}{path}"

# ---- Pydantic Models (vereinfachte Sichten auf ComfyUI-API) ----
class ComfyUIPromptResponse(BaseModel):
    prompt_id: str = Field(alias="prompt_id")

class ComfyUIFileInfo(BaseModel):
    filename: str
    subfolder: Optional[str] = None
    type: Optional[str] = None  # "output" etc.

class ComfyUIOutput(BaseModel):
    # ComfyUI /history gibt nodes → { outputs: { images: [ {filename, subfolder, type}, ... ] } }
    images: List[ComfyUIFileInfo] = Field(default_factory=list)

class ComfyUIHistoryNode(BaseModel):
    status: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, ComfyUIOutput] = Field(default_factory=dict)

class ComfyUIHistoryResponse(BaseModel):
    # Struktur: { "prompt_id": { "workflow": {...}, "outputs": { node_id: { "images": [...] } }, "status": {...} } }
    # Je nach Version variiert leicht; wir akzeptieren generisch:
    model_config = ConfigDict(extra="allow")
    # Wir speichern die rohe JSON-Antwort als dict
    raw: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "ComfyUIHistoryResponse":
        # keine strikte Normalisierung; wir behalten raw für maximale Kompatibilität
        return cls(raw=data)

    def extract_image_paths(self) -> List[Tuple[str, Optional[str], Optional[str]]]:
        # Versucht, alle (filename, subfolder, type) Tripel aus raw zu extrahieren
        out: List[Tuple[str, Optional[str], Optional[str]]] = []
        try:
            # Häufige Form: { "prompt_id": { "outputs": { node_id: { "images": [ {filename,...}, ...] } } } }
            # oder Top-Level direkt "outputs"
            container = self.raw
            # Falls der Key genau die Prompt-ID ist, nimm dessen Inhalt
            if len(container) == 1 and isinstance(next(iter(container.values())), dict):
                container = next(iter(container.values()))
            outputs = container.get("outputs", {})
            if isinstance(outputs, dict):
                for _node_id, node_out in outputs.items():
                    images = node_out.get("images") or []
                    if isinstance(images, list):
                        for img in images:
                            if isinstance(img, dict) and "filename" in img:
                                out.append(
                                    (
                                        str(img.get("filename")),
                                        (img.get("subfolder") or None),
                                        (img.get("type") or None),
                                    )
                                )
        except Exception:
            pass
        return out

# ---- Kernfunktionen ----
async def _post_with_retries(client: httpx.AsyncClient, url: str, body: dict, max_retries: int, base_delay: float) -> httpx.Response:
    delay = base_delay
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_exc = e
            retryable = _is_retryable(e)
            print(f"[COMFYUI] POST attempt {attempt} failed: {e}")
            if attempt >= max_retries or not retryable:
                break
            await asyncio.sleep(delay)
            delay *= 1.7
    raise RuntimeError(f"ComfyUI POST failed after {max_retries} attempts: {last_exc}")

async def _get_with_retries(client: httpx.AsyncClient, url: str, max_retries: int, base_delay: float) -> httpx.Response:
    delay = base_delay
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_exc = e
            retryable = _is_retryable(e)
            print(f"[COMFYUI] GET attempt {attempt} failed: {e}")
            if attempt >= max_retries or not retryable:
                break
            await asyncio.sleep(delay)
            delay *= 1.7
    raise RuntimeError(f"ComfyUI GET failed after {max_retries} attempts: {last_exc}")

async def comfy_available() -> bool:
    try:
        async with httpx.AsyncClient(limits=_limits(), timeout=_timeout()) as c:
            r = await c.get(_comfy_url("/queue"))
            r.raise_for_status()
            return True
    except Exception:
        return False

def build_default_text2img_workflow(prompt: str, seed: Optional[int] = None, width: int = 1024, height: int = 1024, steps: int = 28, cfg: float = 6.5, sampler_name: str = "dpmpp_2m") -> Dict[str, Any]:
    """
    Minimaler ComfyUI-Workflow (als Beispiel).
    Wichtig: Passe Node-IDs und konkreten Knoten-Typen an deinen lokalen Workflow an.
    Dieser Platzhalter nutzt die Standard-Text-Encoder/UNet/VAEDecode-Knoten-Bezeichnungen, wie sie in vielen Beispielen vorkommen.
    """
    # Hinweis: In der Praxis importierst du dein aus ComfyUI exportiertes JSON und überschreibst nur die Eingabe-Text/Seed/Size Felder.
    # Hier geben wir eine sehr einfache Vorlage zurück.
    return {
        "prompt": {
            "7": {  # Knoten-ID beispielhaft
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed if seed is not None else 1234,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": sampler_name,
                    "scheduler": "karras",
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["5", 0],
                    "negative": ["6", 0],
                    "latent_image": ["3", 0],
                },
            },
            "3": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": width, "height": height, "batch_size": 1},
            },
            "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "SDXL.safetensors"}},
            "5": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": ["4", 1]},
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "text, watermark, logo, low quality, bad anatomy, blurry", "clip": ["4", 1]},
            },
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["4", 2]}},
            "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0]}},
        }
    }

async def post_prompt(workflow: Dict[str, Any]) -> str:
    """
    Sendet den Workflow an ComfyUI (/prompt) und gibt die prompt_id zurück.
    """
    if COMFY_DISABLE:
        raise RuntimeError("ComfyUI disabled by APP_DISABLE_COMFYUI=1")
    if not isinstance(workflow, dict):
        raise ValueError("workflow must be a dict")
    async with httpx.AsyncClient(limits=_limits(), timeout=_timeout()) as client:
        r = await _post_with_retries(
            client,
            _comfy_url("/prompt"),
            workflow,
            max_retries=COMFY_MAX_RETRIES,
            base_delay=COMFY_RETRY_BASE_DELAY,
        )
        data = r.json()
        try:
            parsed = ComfyUIPromptResponse.model_validate(data)
            return parsed.prompt_id
        except ValidationError:
            # Manche Builds geben {"prompt_id":"...","number":...} direkt zurück
            pid = (data or {}).get("prompt_id")
            if isinstance(pid, str) and pid:
                return pid
            raise

async def poll_history_until_done(prompt_id: str, poll_interval: float = 1.0, max_wait_sec: float = 180.0) -> ComfyUIHistoryResponse:
    """
    Pollt /history/{id}, bis fertig oder Timeout. Gibt die rohe History-Antwort zurück.
    """
    if COMFY_DISABLE:
        raise RuntimeError("ComfyUI disabled by APP_DISABLE_COMFYUI=1")
    deadline = asyncio.get_event_loop().time() + max_wait_sec
    async with httpx.AsyncClient(limits=_limits(), timeout=_timeout()) as client:
        while True:
            if asyncio.get_event_loop().time() >= deadline:
                raise TimeoutError("ComfyUI history polling timed out")
            r = await _get_with_retries(
                client,
                _comfy_url(f"/history/{prompt_id}"),
                max_retries=COMFY_MAX_RETRIES,
                base_delay=COMFY_RETRY_BASE_DELAY,
            )
            data = r.json()
            hist = ComfyUIHistoryResponse.from_json(data)
            # Heuristik: Prüfe auf Status "completed" o. ä., oder ob Outputs Bilder enthalten
            images = hist.extract_image_paths()
            status_text = ""
            try:
                container = data
                if len(container) == 1 and isinstance(next(iter(container.values())), dict):
                    container = next(iter(container.values()))
                status_text = str(container.get("status", {}).get("status", ""))  # "completed" / "running"
            except Exception:
                pass

            if images and (status_text.lower() in {"completed", "success", ""}):
                return hist
            await asyncio.sleep(poll_interval)

async def download_outputs(history: ComfyUIHistoryResponse, out_dir: Path) -> List[Path]:
    """
    Lädt alle in der History referenzierten Output-Bilder via /view?filename=...&subfolder=...&type=output
    und speichert sie in out_dir. Gibt die lokalen Pfade zurück.
    """
    if COMFY_DISABLE:
        raise RuntimeError("ComfyUI disabled by APP_DISABLE_COMFYUI=1")
    out_dir.mkdir(parents=True, exist_ok=True)

    items = history.extract_image_paths()
    if not items:
        raise RuntimeError("No images found in ComfyUI history")

    saved: List[Path] = []
    async with httpx.AsyncClient(limits=_limits(), timeout=_timeout()) as client:
        for (filename, subfolder, ftype) in items:
            params = {"filename": filename}
            if subfolder:
                params["subfolder"] = subfolder
            if ftype:
                params["type"] = ftype
            url = _comfy_url("/view")
            delay = COMFY_RETRY_BASE_DELAY
            content: Optional[bytes] = None
            last_exc: Optional[Exception] = None
            for attempt in range(1, COMFY_MAX_RETRIES + 1):
                try:
                    r = await client.get(url, params=params)
                    r.raise_for_status()
                    content = r.content
                    break
                except Exception as e:
                    last_exc = e
                    print(f"[COMFYUI] download attempt {attempt} failed: {e}")
                    if attempt < COMFY_MAX_RETRIES and _is_retryable(e):
                        await asyncio.sleep(delay)
                        delay *= 1.7
                        continue
                    break
            if content is None:
                raise RuntimeError(f"failed_to_download_output: {filename} ({last_exc})")
            # Sicherer Dateiname
            safe_name = filename.replace("\\", "_").replace("/", "_")
            target = out_dir / f"comfy_{safe_name}"
            target.write_bytes(content)
            if target.stat().st_size < 1024:
                raise RuntimeError(f"downloaded_file_too_small: {target}")
            print(f"[COMFYUI] saved {target} ({target.stat().st_size} bytes)")
            saved.append(target)
    return saved

# ---- High-level Convenience ----
async def generate_with_comfyui(
    prompt_text: str,
    out_dir: Optional[Path] = None,
    workflow: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    width: int = 1024,
    height: int = 1024,
    steps: int = 28,
    cfg: float = 6.5,
    sampler_name: str = "dpmpp_2m",
    poll_interval: float = 1.0,
    max_wait_sec: float = 180.0,
) -> List[Path]:
    """
    Komplettablauf: (1) Workflow bauen/übernehmen, (2) /prompt posten, (3) /history poll, (4) Outputs herunterladen.
    Gibt eine Liste lokaler Bildpfade zurück.
    """
    if COMFY_DISABLE:
        raise RuntimeError("ComfyUI disabled by APP_DISABLE_COMFYUI=1")
    out = out_dir or OUTPUT_DIR
    if workflow is None:
        workflow = build_default_text2img_workflow(
            prompt=prompt_text,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
        )
    # Falls du ein aus der UI exportiertes Workflow-JSON hast, kannst du hier Felder überschreiben:
    # z. B. workflow["prompt"]["5"]["inputs"]["text"] = prompt_text

    pid = await post_prompt(workflow)
    hist = await poll_history_until_done(pid, poll_interval=poll_interval, max_wait_sec=max_wait_sec)
    paths = await download_outputs(hist, out)
    return paths

# ---- Einfacher Selbsttest via CLI ----
if __name__ == "__main__":
    import asyncio as _asyncio

    async def _main():
        if COMFY_DISABLE:
            print("ComfyUI disabled (APP_DISABLE_COMFYUI=1); enable it to run the test.")
            return
        ok = await comfy_available()
        if not ok:
            print("ComfyUI not available on localhost; start it first.")
            return
        try:
            images = await generate_with_comfyui(
                prompt_text="A serene landscape, lake at sunset, soft golden light, high detail, photorealistic",
                seed=1234,
                width=1024,
                height=576,
                steps=28,
                cfg=6.5,
            )
            print("Generated:", [str(p) for p in images])
        except Exception as e:
            print("Error:", e)

    _asyncio.run(_main())
