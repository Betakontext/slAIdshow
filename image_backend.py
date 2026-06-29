#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Image backends for slAIdshow (relaxed, no guards) – Pollinations + ComfyUI local/remote.
English instructions and comments; short German notes on complex logic.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import httpx
from pydantic import BaseModel, Field, ValidationError

# ---------- Minimal logging helper ----------
def _dbg_enabled() -> bool:
    v = (os.getenv("APP_IMAGE_BACKEND_DEBUG", "0") or "").strip().lower()
    return v in {"1", "true", "yes", "on"}

def _log(msg: str) -> None:
    if _dbg_enabled():
        print(f"[IMG] {msg}")

# ---------- ENV helpers ----------
def _env_str(k: str, d: str) -> str:
    return (os.getenv(k, d) or "").strip()

def _env_int(k: str, d: int) -> int:
    try:
        return int(os.getenv(k, str(d)))
    except Exception:
        return d

def _env_float(k: str, d: float) -> float:
    try:
        return float(os.getenv(k, str(d)))
    except Exception:
        return d

def _env_bool01(k: str, d: int = 0) -> bool:
    v = (os.getenv(k, str(d)) or "").strip().lower()
    return v in {"1", "true", "yes", "on"}

# ---------- httpx tuning ----------
def _httpx_limits() -> httpx.Limits:
    return httpx.Limits(max_keepalive_connections=10, max_connections=20, keepalive_expiry=30.0)

# ---------- Style engine shims (soft dep) ----------
try:
    from style_engine import resolve_style_descriptors_for_reference  # type: ignore
except Exception:
    def resolve_style_descriptors_for_reference(*args, **kwargs) -> List[str]:
        return []

# ---------- Comfy bridge imports (Signature: prompt_dict, out_dir, host, port, max_wait_sec) ----------
try:
    from comfyui_bridge import generate_from_prompt_dict  # type: ignore
except Exception as e:
    generate_from_prompt_dict = None  # type: ignore

# ---------- Utilities ----------
def _clamp8(v: int) -> int:
    v = max(64, min(4096, int(v)))
    return v - (v % 8)

def _resolve_size_for_backend(backend_name: str, req_w: Optional[int], req_h: Optional[int]) -> Tuple[Optional[int], Optional[int]]:
    if isinstance(req_w, int) and req_w > 0 and isinstance(req_h, int) and req_h > 0:
        return _clamp8(req_w), _clamp8(req_h)
    b = (backend_name or "").strip().lower()
    if b in {"comfyui", "comfyui_local", "comfyui_remote"}:
        cw = _env_int("APP_COMFY_WIDTH", _env_int("APP_IMAGE_WIDTH", 512))
        ch = _env_int("APP_COMFY_HEIGHT", _env_int("APP_IMAGE_HEIGHT", 512))
        return _clamp8(cw), _clamp8(ch)
    elif b == "pollinations":
        pw = _env_int("POLLINATIONS_WIDTH", _env_int("APP_IMAGE_WIDTH", 1024))
        ph = _env_int("POLLINATIONS_HEIGHT", _env_int("APP_IMAGE_HEIGHT", 1024))
        return _clamp8(pw), _clamp8(ph)
    gw = _env_int("APP_IMAGE_WIDTH", 512)
    gh = _env_int("APP_IMAGE_HEIGHT", 512)
    return _clamp8(gw), _clamp8(gh)

def _mime_for(name: str) -> str:
    n = name.lower()
    if n.endswith(".jpg") or n.endswith(".jpeg"):
        return "image/jpeg"
    if n.endswith(".png"):
        return "image/png"
    if n.endswith(".webp"):
        return "image/webp"
    if n.endswith(".bmp"):
        return "image/bmp"
    return "application/octet-stream"

def _size_from_wh(width: int, height: int) -> str:
    if width > 0 and height > 0:
        return f"{width}x{height}"
    return "1024x1024"

# ---------- Backend interface ----------
class ImageBackend:
    async def generate(self, prompt: str, width: int | None = None, height: int | None = None, negative_prompt: str | None = None, **kwargs: Any) -> Path:
        raise NotImplementedError

# ---------- Pollinations backend (kept functional; split api_base/gen_base) ----------
class _PollinationsV1Datum(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None

class _PollinationsV1Response(BaseModel):
    created: Optional[int] = None
    data: List[_PollinationsV1Datum] = Field(default_factory=list)

class PollinationsConfig(BaseModel):
    # Note: keep api_base for GET and gen_base for v1 POST separated
    api_base: str = Field(default_factory=lambda: _env_str("POLLINATIONS_API_BASE", "https://image.pollinations.ai").rstrip("/"))
    gen_base: str = Field(default_factory=lambda: _env_str("POLLINATIONS_GEN_BASE", "https://gen.pollinations.ai").rstrip("/"))
    secret: str = Field(default_factory=lambda: _env_str("POLLINATIONS_SECRET", ""))
    model: Optional[str] = Field(default_factory=lambda: (_env_str("POLLINATIONS_MODEL", "") or None))
    width: int = Field(default_factory=lambda: _env_int("POLLINATIONS_WIDTH", 1024))
    height: int = Field(default_factory=lambda: _env_int("POLLINATIONS_HEIGHT", 1024))
    nologo: bool = Field(default_factory=lambda: _env_bool01("POLLINATIONS_NOLOGO", 1))
    seed_raw: Optional[str] = Field(default_factory=lambda: os.getenv("POLLINATIONS_SEED"))
    use_v1: bool = Field(default_factory=lambda: _env_bool01("POLLINATIONS_USE_V1", 1))
    size_override: str = Field(default_factory=lambda: _env_str("POLLINATIONS_SIZE", ""))
    v1_edits_path: str = Field(default_factory=lambda: _env_str("POLLINATIONS_V1_IMAGES_EDITS_ENDPOINT", "/v1/images/edits"))
    v1_generations_path: str = Field(default_factory=lambda: _env_str("POLLINATIONS_V1_IMAGES_GENERATIONS_ENDPOINT", "/v1/images/generations"))
    prompt_suffix_style_only: str = Field(default_factory=lambda: _env_str("POLLINATIONS_STYLE_SUFFIX", "adopt the exact visual style, colors, and textures from the reference image; only transfer style, not content."))

    @property
    def seed(self) -> Optional[int]:
        if self.seed_raw is None:
            return None
        try:
            return int(self.seed_raw)
        except Exception:
            return None

async def _retrying_post(
    client: httpx.AsyncClient,
    url: str,
    *,
    json_payload: Dict[str, Any] | None = None,
    files: Dict[str, Any] | None = None,
    headers: Dict[str, str] | None = None,
    max_attempts: int = 4,
    base_delay: float = 0.8,
) -> httpx.Response:
    # Deutsch: Exponentielles Backoff für Pollinations POST-Calls
    last_exc: Optional[Exception] = None
    delay = float(base_delay)
    for attempt in range(1, max_attempts + 1):
        try:
            if files is not None:
                r = await client.post(url, headers=headers, files=files)
            else:
                r = await client.post(url, headers=headers, json=json_payload)
            if r.status_code in (400, 401, 403, 404, 405):
                r.raise_for_status()
            if r.status_code in (429, 500, 502, 503):
                raise httpx.HTTPStatusError(f"transient {r.status_code}", request=r.request, response=r)
            r.raise_for_status()
            return r
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError, httpx.HTTPStatusError) as e:
            last_exc = e
            status = getattr(e, "response", None).status_code if getattr(e, "response", None) else None
            retryable = (status in (429, 500, 502, 503)) or isinstance(e, (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError))
            if attempt >= max_attempts or not retryable:
                break
            await asyncio.sleep(delay)
            delay *= 1.8
    raise RuntimeError(f"pollinations_post_failed after {max_attempts} attempts: {last_exc}")

async def _retrying_get(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: Dict[str, str] | None = None,
    max_attempts: int = 4,
    base_delay: float = 0.8,
) -> httpx.Response:
    last_exc: Optional[Exception] = None
    delay = float(base_delay)
    for attempt in range(1, max_attempts + 1):
        try:
            r = await client.get(url, headers=headers)
            if r.status_code in (429, 500, 502, 503):
                raise httpx.HTTPStatusError(f"transient {r.status_code}", request=r.request, response=r)
            r.raise_for_status()
            return r
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError, httpx.HTTPStatusError) as e:
            last_exc = e
            status = getattr(e, "response", None).status_code if getattr(e, "response", None) else None
            retryable = (status in (429, 500, 502, 503)) or isinstance(e, (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError))
            if attempt >= max_attempts or not retryable:
                break
            await asyncio.sleep(delay)
            delay *= 1.8
    raise RuntimeError(f"pollinations_get_failed after {max_attempts} attempts: {last_exc}")

class PollinationsBackend(ImageBackend):
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = PollinationsConfig()

    def _merge_descriptors(self, base_prompt: str, descriptors: List[str]) -> str:
        ds = [d for d in (descriptors or []) if isinstance(d, str) and d.strip()]
        if not ds:
            return base_prompt
        return (base_prompt.rstrip(",") + ", " + ", ".join(ds)).strip().strip(",")

    def _resolve_descriptors_if_any(self, style_reference_path: Optional[Path]) -> List[str]:
        if not style_reference_path or not style_reference_path.exists():
            return []
        try:
            return resolve_style_descriptors_for_reference(ref_path=style_reference_path, prefer_cloud=True)  # type: ignore[arg-type]
        except Exception:
            return []

    def _build_pollinations_image_url(self, api_base: str, prompt: str,
                                      model: Optional[str], width: Optional[int], height: Optional[int],
                                      nologo: bool, seed: Optional[int]) -> str:
        from urllib.parse import quote, urlencode
        base = (api_base or "").rstrip("/")
        encoded_prompt = quote(prompt, safe="")
        url = f"{base}/image/{encoded_prompt}"
        params: dict[str, str] = {}
        if model:
            params["model"] = model
        if width and width > 0:
            params["width"] = str(width)
        if height and height > 0:
            params["height"] = str(height)
        if nologo:
            params["nologo"] = "true"
        if seed is not None:
            params["seed"] = str(seed)
        if params:
            url = f"{url}?{urlencode(params)}"
        return url

    async def _fetch_v1(self, prompt: str, width: int | None, height: int | None) -> Path:
        # v1 POST with JSON; prefer b64_json, fallback to returned URL
        url = f"{self.cfg.gen_base}{self.cfg.v1_generations_path}"
        headers = {"Authorization": f"Bearer {self.cfg.secret}", "Content-Type": "application/json"}
        w = width if (width and width > 0) else self.cfg.width
        h = height if (height and height > 0) else self.cfg.height
        payload = {"model": self.cfg.model or "flux", "prompt": prompt, "size": (self.cfg.size_override or _size_from_wh(w, h))}
        delay = 1.0
        last_exc: Optional[Exception] = None
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0), limits=_httpx_limits()) as client:
            for attempt in range(1, 6):
                try:
                    r = await client.post(url, headers=headers, json=payload)
                    r.raise_for_status()
                    parsed = _PollinationsV1Response.model_validate(r.json())
                    if not parsed.data:
                        raise RuntimeError("pollinations_v1_empty_data")
                    first = parsed.data[0]
                    if first.b64_json:
                        from base64 import b64decode
                        raw = b64decode(first.b64_json, validate=True)
                        target = self.out_dir / f"img_{uuid.uuid4().hex}.jpg"
                        target.write_bytes(raw)
                        if target.stat().st_size < 1024:
                            raise RuntimeError("pollinations_v1_too_small")
                        return target
                    if first.url and first.url.startswith("http"):
                        ir = await _retrying_get(client, first.url)
                        content = ir.content
                        target = self.out_dir / f"img_{uuid.uuid4().hex}.jpg"
                        target.write_bytes(content)
                        if target.stat().st_size < 1024:
                            raise RuntimeError("pollinations_v1_too_small")
                        return target
                    raise RuntimeError("pollinations_v1_missing_data")
                except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError, httpx.HTTPStatusError, ValidationError) as e:
                    last_exc = e
                    if attempt < 5:
                        await asyncio.sleep(delay)
                        delay *= 1.7
                    continue
        raise RuntimeError(f"pollinations_v1_all_attempts_failed: {last_exc}")

    async def _fetch_get(self, prompt: str, width: int | None, height: int | None) -> Path:
        # Legacy GET against api_base (separate from gen_base)
        w = width if (width and width > 0) else self.cfg.width
        h = height if (height and height > 0) else self.cfg.height
        url = self._build_pollinations_image_url(self.cfg.api_base, prompt, self.cfg.model, w, h, self.cfg.nologo, self.cfg.seed)
        params: dict[str, str] = {}
        if self.cfg.secret:
            params["key"] = self.cfg.secret
        delay = 1.0
        last_exc: Optional[Exception] = None
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0), limits=_httpx_limits(), follow_redirects=True) as client:
            for attempt in range(1, 5):
                try:
                    r = await client.get(url, params=params)
                    r.raise_for_status()
                    content = r.content
                    if not content or len(content) < 1024:
                        raise RuntimeError("pollinations_get_too_small")
                    target = self.out_dir / f"img_{uuid.uuid4().hex}.jpg"
                    target.write_bytes(content)
                    return target
                except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError, httpx.HTTPStatusError) as e:
                    last_exc = e
                    if attempt < 4:
                        await asyncio.sleep(delay)
                        delay *= 1.7
                    continue
        raise RuntimeError(f"pollinations_get_all_attempts_failed: {last_exc}")

    async def _post_v1_edits(self, prompt: str, image_url: str, width: int | None, height: int | None, *, negative_prompt: str | None = None, seed: Optional[int] = None) -> Path:
        url = f"{self.cfg.gen_base}{self.cfg.v1_edits_path}"
        headers = {"Authorization": f"Bearer {self.cfg.secret}", "Content-Type": "application/json"}
        w = width if (width and width > 0) else self.cfg.width
        h = height if (height and height > 0) else self.cfg.height
        payload: dict[str, Any] = {
            "model": self.cfg.model or "flux",
            "prompt": prompt,
            "image": image_url,
            "size": (self.cfg.size_override or _size_from_wh(w, h)),
            "response_format": "url",
        }
        if negative_prompt and negative_prompt.strip():
            payload["negative_prompt"] = negative_prompt.strip()
        if seed is not None:
            payload["seed"] = int(seed)
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0), limits=_httpx_limits(), follow_redirects=True) as client:
            r = await _retrying_post(client, url, json_payload=payload, headers=headers)
            try:
                j = r.json()
            except Exception:
                raise RuntimeError("pollinations_v1_edits_invalid_json")
            data = j.get("data")
            if isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, dict):
                    if "url" in first and isinstance(first["url"], str) and first["url"].startswith("http"):
                        ir = await _retrying_get(client, first["url"])
                        content = ir.content
                        target = self.out_dir / f"img_{uuid.uuid4().hex}.jpg"
                        target.write_bytes(content)
                        if target.stat().st_size < 1024:
                            raise RuntimeError("pollinations_v1_edits_too_small")
                        return target
                    if "b64_json" in first and isinstance(first["b64_json"], str):
                        from base64 import b64decode
                        raw = b64decode(first["b64_json"], validate=True)
                        target = self.out_dir / f"img_{uuid.uuid4().hex}.jpg"
                        target.write_bytes(raw)
                        if target.stat().st_size < 1024:
                            raise RuntimeError("pollinations_v1_edits_too_small")
                        return target
            raise RuntimeError("pollinations_v1_edits_missing_data")

    async def _post_v1_edits_multipart(self, prompt: str, image_path: Path, width: int | None, height: int | None, *, negative_prompt: str | None = None, seed: Optional[int] = None) -> Path:
        if not image_path.exists() or not image_path.is_file():
            raise FileNotFoundError(image_path)
        url = f"{self.cfg.gen_base}{self.cfg.v1_edits_path}"
        headers = {"Authorization": f"Bearer {self.cfg.secret}"}
        w = width if (width and width > 0) else self.cfg.width
        h = height if (height and height > 0) else self.cfg.height
        files: Dict[str, Any] = {
            "prompt": (None, (prompt or "").strip()),
            "response_format": (None, "url"),
            "n": (None, "1"),
            "model": (None, (self.cfg.model or "flux")),
            "size": (None, (self.cfg.size_override or _size_from_wh(w, h))),
            "image": (image_path.name, image_path.read_bytes(), _mime_for(image_path.name)),
        }
        if negative_prompt and negative_prompt.strip():
            files["negative_prompt"] = (None, negative_prompt.strip())
        if seed is not None:
            files["seed"] = (None, str(int(seed)))
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0), limits=_httpx_limits(), follow_redirects=True, http2=True) as client:
            r = await _retrying_post(client, url, files=files, headers=headers)
            try:
                j = r.json()
            except Exception:
                raise RuntimeError("pollinations_v1_edits_multipart_invalid_json")
            data = j.get("data")
            if isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, dict):
                    if "url" in first and isinstance(first["url"], str) and first["url"].startswith("http"):
                        ir = await _retrying_get(client, first["url"])
                        content = ir.content
                        target = self.out_dir / f"img_{uuid.uuid4().hex}.jpg"
                        target.write_bytes(content)
                        if target.stat().st_size < 1024:
                            raise RuntimeError("pollinations_v1_edits_mp_too_small")
                        return target
                    if "b64_json" in first and isinstance(first["b64_json"], str):
                        from base64 import b64decode
                        raw = b64decode(first["b64_json"], validate=True)
                        target = self.out_dir / f"img_{uuid.uuid4().hex}.jpg"
                        target.write_bytes(raw)
                        if target.stat().st_size < 1024:
                            raise RuntimeError("pollinations_v1_edits_mp_too_small")
                        return target
            raise RuntimeError("pollinations_v1_edits_multipart_missing_data")

    async def generate(self, prompt: str, width: int | None = None, height: int | None = None, negative_prompt: str | None = None, **kwargs: Any) -> Path:
        full_prompt = (prompt or "").strip()
        style_reference_path: Optional[Path] = kwargs.get("style_reference_path")
        if isinstance(style_reference_path, str):
            style_reference_path = Path(style_reference_path)
        if isinstance(style_reference_path, Path) and style_reference_path.exists():
            desc = self._resolve_descriptors_if_any(style_reference_path)
            if desc:
                full_prompt = self._merge_descriptors(full_prompt, desc)
            suffix = (self.cfg.prompt_suffix_style_only or "").strip()
            if suffix:
                full_prompt = f"{full_prompt}\n{suffix}"
        n_prompt = (negative_prompt or "").strip()
        if n_prompt:
            full_prompt = f"{full_prompt}\n-- negative: {n_prompt}"
        rw, rh = _resolve_size_for_backend("pollinations", width, height)
        eff_w = rw if (rw and rw > 0) else (width if (width and width > 0) else self.cfg.width)
        eff_h = rh if (rh and rh > 0) else (height if (height and height > 0) else self.cfg.height)
        seed_override: Optional[int] = kwargs.get("seed") if isinstance(kwargs.get("seed"), int) else self.cfg.seed

        # Edits via local multipart
        if isinstance(style_reference_path, Path) and style_reference_path.exists():
            return await self._post_v1_edits_multipart(full_prompt, style_reference_path, eff_w, eff_h, negative_prompt=n_prompt or None, seed=seed_override)

        # Edits via URL
        style_reference_url: Optional[str] = kwargs.get("style_reference_url")
        if isinstance(style_reference_url, str) and style_reference_url.strip():
            suffix = (self.cfg.prompt_suffix_style_only or "").strip()
            if suffix:
                full_prompt = f"{full_prompt}\n{suffix}"
            return await self._post_v1_edits(full_prompt, style_reference_url.strip(), eff_w, eff_h, negative_prompt=n_prompt or None, seed=seed_override)

        # Plain generation
        if self.cfg.use_v1 and self.cfg.secret:
            try:
                return await self._fetch_v1(full_prompt, eff_w, eff_h)
            except Exception as e:
                _log(f"v1_failed: {e}; falling back to GET flavor")
                return await self._fetch_get(full_prompt, eff_w, eff_h)
        else:
            return await self._fetch_get(full_prompt, eff_w, eff_h)

# ---------- ComfyUI backend (local/remote via Bridge) ----------
class ComfyConfig(BaseModel):
    host: str = Field(default_factory=lambda: _env_str("APP_COMFY_HOST", "127.0.0.1"))
    port: int = Field(default_factory=lambda: _env_int("APP_COMFY_PORT", 8188))
    workflow_path: Path = Field(default_factory=lambda: Path(_env_str("APP_COMFY_WORKFLOW", "./workflows/text2img_SD15-FP16.json")).resolve())
    width: int = Field(default_factory=lambda: _env_int("APP_COMFY_WIDTH", _env_int("APP_IMAGE_WIDTH", 512)))
    height: int = Field(default_factory=lambda: _env_int("APP_COMFY_HEIGHT", _env_int("APP_IMAGE_HEIGHT", 512)))
    steps: int = Field(default_factory=lambda: _env_int("APP_COMFY_STEPS", 20))
    cfg: float = Field(default_factory=lambda: _env_float("APP_COMFY_CFG", 6.5))
    sampler: str = Field(default_factory=lambda: _env_str("APP_COMFY_SAMPLER", "euler"))
    timeout_sec: float = Field(default_factory=lambda: _env_float("APP_COMFY_TIMEOUT_SEC", 300.0))
    disabled: bool = Field(default_factory=lambda: _env_bool01("APP_DISABLE_COMFYUI", 0))
    negative: str = Field(default_factory=lambda: _env_str("APP_COMFY_NEGATIVE", "text, watermark, logo, low quality, blurry, bad anatomy"))
    output_dir: Path = Field(default_factory=lambda: Path(_env_str("APP_OUTPUT_DIR", str(Path.cwd() / "outputs" / "images"))).resolve())

class LocalComfyBackend(ImageBackend):
    def __init__(self, out_dir: Path) -> None:
        if generate_from_prompt_dict is None:
            raise RuntimeError("comfyui_bridge.generate_from_prompt_dict not available")
        self.cfg = ComfyConfig()
        self.out_dir = Path(out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _load_prompt_file_only_prompt_dict(self) -> dict:
        """Read workflow JSON and return the 'prompt' dict (Comfy /prompt format) if present."""
        data = json.loads(self.cfg.workflow_path.read_text(encoding="utf-8"))
        if "prompt" in data and isinstance(data["prompt"], dict):
            return data["prompt"]
        return data

    def _override_text_nodes(self, prompt_dict: dict, positive: str, negative: str) -> None:
        """Set positive/negative on first two CLIPTextEncode nodes (heuristic)."""
        clip_nodes: List[dict] = []
        for node in prompt_dict.values():
            if isinstance(node, dict) and str(node.get("class_type", "")).strip() == "CLIPTextEncode":
                clip_nodes.append(node)
        if clip_nodes:
            inputs = clip_nodes[0].get("inputs", {})
            if isinstance(inputs, dict):
                if "text" in inputs:
                    inputs["text"] = positive
        if len(clip_nodes) > 1:
            inputs = clip_nodes[1].get("inputs", {})
            if isinstance(inputs, dict):
                if "text" in inputs:
                    inputs["text"] = negative

    def _override_dimensions(self, prompt_dict: dict, width: int, height: int) -> None:
        """Inject resolution across typical latent creators and samplers."""
        for node in prompt_dict.values():
            if not isinstance(node, dict):
                continue
            cls = str(node.get("class_type") or node.get("class", "")).strip()
            inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
            if cls in {"EmptyLatentImage", "EmptyLatentImageBatch", "LatentImage", "CreateLatentImage"}:
                if "width" in inputs:
                    inputs["width"] = width
                if "height" in inputs:
                    inputs["height"] = height
            if cls.startswith("KSampler"):
                if "width" in inputs:
                    inputs["width"] = width
                if "height" in inputs:
                    inputs["height"] = height

    async def generate(self, prompt: str, width: int | None = None, height: int | None = None, negative_prompt: str | None = None, **kwargs: Any) -> Path:
        # Load workflow and transform to prompt_dict
        workflow_path = Path(os.getenv("APP_COMFY_WORKFLOW", str(self.cfg.workflow_path)) or str(self.cfg.workflow_path)).resolve()
        if not workflow_path.exists():
            raise FileNotFoundError(f"workflow_missing: {workflow_path}")
        prompt_dict = self._load_prompt_file_only_prompt_dict()

        # Inject positive/negative and dimensions
        pos = (prompt or "").strip()
        neg = (negative_prompt or self.cfg.negative or "").strip()
        self._override_text_nodes(prompt_dict, positive=pos, negative=neg)
        rw, rh = _resolve_size_for_backend("comfyui", width, height)
        eff_w = int(rw if (rw and rw > 0) else (width if (width and width > 0) else self.cfg.width))
        eff_h = int(rh if (rh and rh > 0) else (height if (height and height > 0) else self.cfg.height))
        self._override_dimensions(prompt_dict, eff_w, eff_h)

        # Resolve host/port (local or remote as configured in ENV)
        host = (os.getenv("APP_COMFY_HOST", self.cfg.host) or self.cfg.host)
        port = int(os.getenv("APP_COMFY_PORT", str(self.cfg.port)) or self.cfg.port)
        if _dbg_enabled():
            _log(f"Comfy target: {host}:{port} | steps={self.cfg.steps} cfg={self.cfg.cfg} sampler={self.cfg.sampler}")

        # Dispatch to bridge
        images: List[str] = await generate_from_prompt_dict(  # type: ignore[call-arg]
            prompt_dict=prompt_dict,
            out_dir=str(self.out_dir),
            host=str(host),
            port=int(port),
            max_wait_sec=float(self.cfg.timeout_sec),
        )

        if images and isinstance(images[0], str):
            p = Path(images[0]).resolve()
            if p.exists() and p.is_file() and p.stat().st_size >= 1024:
                return p
        raise RuntimeError("comfyui_no_output_file")

# ---------- Factory ----------
def build_image_backend() -> ImageBackend:
    backend = (_env_str("IMAGE_BACKEND", "comfyui") or "comfyui").lower()
    out_dir = Path(_env_str("APP_OUTPUT_DIR", str(Path.cwd() / "outputs" / "images"))).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if backend in {"comfyui", "comfyui_remote"}:
        return LocalComfyBackend(out_dir)
    if backend == "pollinations":
        return PollinationsBackend(out_dir)
    # Default
    return LocalComfyBackend(out_dir)
