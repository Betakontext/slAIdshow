#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
image_backend.py

Unified interface for image generation backends:
- LocalComfyBackend (primary; strict localhost)
- PollinationsBackend (optional cloud fallback; disabled by default)

Usage:
    backend = make_image_backend_from_env()
    img_path = await backend.generate_image(prompt="A cat in a hat", out_dir="outputs/images")

Env:
    IMAGE_BACKEND = local|pollinations (default: local)
    # Local (ComfyUI)
    APP_COMFY_HOST (default 127.0.0.1)
    APP_COMFY_PORT (default 8188)
    # Pollinations (cloud)
    POLLINATIONS_BASE (default https://image.pollinations.ai)
    POLLINATIONS_WIDTH (default 1024)
    POLLINATIONS_HEIGHT (default 576)
    POLLINATIONS_NOLOGO (default true)
    POLLINATIONS_SEED (optional int)
"""

from __future__ import annotations

import asyncio
import os
import uuid
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import httpx
from pydantic import BaseModel, Field, validator
from urllib.parse import quote as url_quote


# ---------- Helpers: env parsing ----------

def _strip_inline_comment(s: str) -> str:
    s = s.strip()
    out = []
    in_single = False
    in_double = False
    for ch in s:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        if ch == "#" and not in_single and not in_double:
            break
        out.append(ch)
    return "".join(out).strip()

def env_str(k: str, default: str) -> str:
    v = os.getenv(k)
    return _strip_inline_comment(v).strip() if v is not None else default

def env_int(k: str, default: int) -> int:
    v = os.getenv(k)
    if v is None:
        return default
    try:
        return int(_strip_inline_comment(v))
    except Exception:
        return default

def env_bool(k: str, default: bool = False) -> bool:
    v = os.getenv(k)
    if v is None:
        return default
    vv = _strip_inline_comment(v).lower()
    return vv in {"1", "true", "yes", "on"}


# ---------- Retry wrapper ----------

async def _with_retries(coro_factory, attempts: int = 3, base_delay: float = 0.4):
    last_exc = None
    for i in range(attempts):
        try:
            return await coro_factory()
        except Exception as e:
            last_exc = e
            await asyncio.sleep(base_delay * (2 ** i))
    raise last_exc if last_exc else RuntimeError("Retry failed without exception")


# ---------- Pydantic models ----------

class GenRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1200)
    out_dir: Path = Field(default=Path("outputs/images"))
    filename: Optional[str] = None

    @validator("out_dir")
    def _ensure_out_dir(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

    def target_path(self) -> Path:
        name = self.filename or f"img_{uuid.uuid4().hex}.jpg"
        return self.out_dir / name


# ---------- Backend interface ----------

class ImageBackend(ABC):
    @abstractmethod
    async def generate_image(self, prompt: str, out_dir: str | Path, filename: Optional[str] = None) -> Path:
        ...


# ---------- Local Comfy Backend (primary) ----------

class LocalComfyConfig(BaseModel):
    host: str = Field(default=env_str("APP_COMFY_HOST", "127.0.0.1"))
    port: int = Field(default=env_int("APP_COMFY_PORT", 8188))

    @validator("host")
    def _enforce_local(cls, v: str) -> str:
        if v != "127.0.0.1":
            raise ValueError(f"Localhost only for safety (got {v})")
        return v

class LocalComfyBackend(ImageBackend):
    def __init__(self, cfg: Optional[LocalComfyConfig] = None):
        self.cfg = cfg or LocalComfyConfig()
        self.base = f"http://{self.cfg.host}:{self.cfg.port}"

    async def _post_prompt(self, client: httpx.AsyncClient, workflow: dict) -> str:
        async def _do():
            r = await client.post(f"{self.base}/prompt", json=workflow)
            r.raise_for_status()
            data = r.json()
            return data.get("prompt_id") or data.get("prompt") or data.get("id")
        return await _with_retries(_do)

    async def _wait_result(self, client: httpx.AsyncClient, prompt_id: str, timeout_s: float = 60.0) -> dict:
        # Poll history for completion
        deadline = asyncio.get_event_loop().time() + timeout_s
        while True:
            async def _do():
                r = await client.get(f"{self.base}/history/{prompt_id}")
                if r.status_code == 404:
                    # not yet ready
                    return None
                r.raise_for_status()
                return r.json()
            data = await _with_retries(_do, attempts=2)
            if data:
                return data
            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError("ComfyUI generation timed out")
            await asyncio.sleep(0.5)

    async def _download_first_image(self, client: httpx.AsyncClient, history: dict, target: Path) -> Path:
        # Minimal extraction: grab first image path from history and download via /view
        # Many ComfyUI workflows save to output; some require /view endpoint usage
        node_data = next(iter(history.values()), {})
        imgs = []
        for v in node_data.get("outputs", {}).values():
            if isinstance(v, dict) and "images" in v:
                imgs.extend(v["images"])
        if not imgs:
            raise RuntimeError("No images found in ComfyUI history")

        img0 = imgs[0]
        # When using /view, fields: "filename", "subfolder", "type" (e.g., "output")
        filename = img0.get("filename")
        subfolder = img0.get("subfolder", "")
        img_type = img0.get("type", "output")

        async def _do():
            params = {"filename": filename, "subfolder": subfolder, "type": img_type}
            r = await client.get(f"{self.base}/view", params=params)
            r.raise_for_status()
            target.write_bytes(r.content)
            return target

        return await _with_retries(_do)

    async def generate_image(self, prompt: str, out_dir: str | Path, filename: Optional[str] = None) -> Path:
        req = GenRequest(prompt=prompt, out_dir=Path(out_dir), filename=filename)
        # Minimal workflow: a text-to-image node graph must be preconfigured or injected here.
        # For simplicity, assume a saved workflow on ComfyUI that reads 'prompt' via a 'CLIPTextEncode' input.
        # Here we send a trivial payload; adjust to your workflow JSON.
        workflow = {
            "prompt": {
                # Replace with your actual node definitions; this is a placeholder contract
                "1": {
                    "class_type": "KSampler",
                    "inputs": {"seed": random.randint(1, 2**31-1), "cfg": 5.0, "steps": 12}
                },
                "2": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {"text": req.prompt}
                }
            }
        }

        timeout = httpx.Timeout(10.0, connect=3.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            prompt_id = await self._post_prompt(client, workflow)
            history = await self._wait_result(client, prompt_id, timeout_s=120.0)
            path = req.target_path()
            return await self._download_first_image(client, history, path)


# ---------- Pollinations Backend (optional cloud fallback) ----------

class PollinationsConfig(BaseModel):
    base: str = Field(default=env_str("POLLINATIONS_BASE", "https://image.pollinations.ai"))
    width: int = Field(default=env_int("POLLINATIONS_WIDTH", 1024))
    height: int = Field(default=env_int("POLLINATIONS_HEIGHT", 576))
    nologo: bool = Field(default=env_bool("POLLINATIONS_NOLOGO", True))
    seed: Optional[int] = Field(default=None)

class PollinationsBackend(ImageBackend):
    def __init__(self, cfg: Optional[PollinationsConfig] = None):
        self.cfg = cfg or PollinationsConfig()

    def _build_url(self, prompt: str) -> str:
        p = url_quote(prompt)
        url = f"{self.cfg.base}/prompt/{p}?width={self.cfg.width}&height={self.cfg.height}"
        if self.cfg.nologo:
            url += "&nologo=true"
        if self.cfg.seed is not None:
            url += f"&seed={int(self.cfg.seed)}"
        return url

    async def generate_image(self, prompt: str, out_dir: str | Path, filename: Optional[str] = None) -> Path:
        req = GenRequest(prompt=prompt, out_dir=Path(out_dir), filename=filename)
        url = self._build_url(req.prompt)
        timeout = httpx.Timeout(20.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async def _do():
                r = await client.get(url)
                r.raise_for_status()
                path = req.target_path()
                path.write_bytes(r.content)
                return path
            return await _with_retries(_do, attempts=3, base_delay=0.6)


# ---------- Factory ----------

def make_image_backend_from_env() -> ImageBackend:
    backend = env_str("IMAGE_BACKEND", "local").lower()
    if backend == "local":
        return LocalComfyBackend()
    elif backend == "pollinations":
        # Warning: external network; ensure you explicitly opt in
        return PollinationsBackend()
    else:
        raise ValueError(f"Unsupported IMAGE_BACKEND: {backend}")
