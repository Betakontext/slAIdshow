# pollinations_backend.py
# English instructions and comments. German comments annotate complex logic succinctly.
# Asynchronous Pollinations backend (v1 + legacy GET) for slAIdshow.
from __future__ import annotations

import asyncio
import os
import uuid
from base64 import b64decode
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import httpx
from pydantic import BaseModel, Field, ValidationError


# =========================
# Env + helpers
# =========================

def _env_str(k: str, d: str = "") -> str:
    return (os.getenv(k, d) or "").strip()

def _env_int(k: str, d: int) -> int:
    try:
        return int(os.getenv(k, str(d)))
    except Exception:
        return d

def _env_bool01(k: str, d: int = 0) -> bool:
    v = (os.getenv(k, str(d)) or "").strip().lower()
    return v in {"1","true","yes","on"}

def _debug() -> bool:
    return (_env_str("APP_IMAGE_BACKEND_DEBUG", "0").lower() in {"1","true","yes","on"})

def _app_root_dir() -> Path:
    return Path(_env_str("APP_OUTPUT_DIR", ".")).resolve()

def _outputs_images_dir() -> Path:
    return (_app_root_dir() / "outputs" / "images").resolve()

def _ensure_outputs_dir() -> Path:
    p = _outputs_images_dir()
    p.mkdir(parents=True, exist_ok=True)
    return p

def _httpx_limits() -> httpx.Limits:
    return httpx.Limits(max_connections=20, max_keepalive_connections=10, keepalive_expiry=30.0)

def _timeout_default() -> httpx.Timeout:
    return httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=5.0)

def _retryable_status(status: Optional[int]) -> bool:
    return status in (429, 500, 502, 503, 504)

async def _retrying_request(
    method: str,
    url: str,
    *,
    client: Optional[httpx.AsyncClient] = None,
    json_payload: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    files: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    max_attempts: int = 4,
    base_delay: float = 0.8,
    timeout: Optional[httpx.Timeout] = None,
    follow_redirects: bool = False,
) -> httpx.Response:
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(limits=_httpx_limits(), timeout=timeout or _timeout_default(), follow_redirects=follow_redirects)
    try:
        delay = float(base_delay)
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                r = await client.request(
                    method.upper(),
                    url,
                    json=json_payload,
                    data=data,
                    files=files,
                    headers=headers,
                )
                if _retryable_status(r.status_code):
                    raise httpx.HTTPStatusError(f"transient {r.status_code}", request=r.request, response=r)
                r.raise_for_status()
                return r
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError, httpx.HTTPStatusError) as e:
                last_exc = e
                status = getattr(e, "response", None).status_code if getattr(e, "response", None) else None
                retryable = _retryable_status(status) or isinstance(e, (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError))
                if attempt >= max_attempts or not retryable:
                    break
                if _debug():
                    print(f"[POLLINATIONS][retry] attempt={attempt} url={url} err={type(e).__name__} next_in={delay:.2f}s")
                await asyncio.sleep(delay)
                delay *= 1.7
        raise RuntimeError(f"request_failed after {max_attempts} attempts: {last_exc}")
    finally:
        if owns_client:
            await client.aclose()

def _pick_ext_from_content_type(ct: Optional[str]) -> str:
    if not ct:
        return ".jpg"
    ct = ct.lower().split(";")[0].strip()
    return {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }.get(ct, ".jpg")

async def _download_and_store_image_bytes(content: bytes, *, suggested_ext: str) -> Path:
    out_dir = _ensure_outputs_dir()
    target = out_dir / f"img_{uuid.uuid4().hex}{suggested_ext if suggested_ext.startswith('.') else '.' + suggested_ext}"
    target.write_bytes(content)
    if _debug():
        print(f"[POLLINATIONS] saved {target} ({len(content)} bytes)")
    return target

def _size_from_wh(width: int, height: int) -> str:
    if width > 0 and height > 0:
        return f"{width}x{height}"
    return "1024x1024"

def _merge_negative_into_prompt(prompt: str, negative: Optional[str]) -> str:
    if negative and negative.strip():
        return f"{prompt.strip()}\n-- negative: {negative.strip()}"
    return prompt.strip()


# =========================
# Public protocol
# =========================

class ImageBackend(Protocol):
    async def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 768,
        height: int = 512,
        style: Optional[Dict[str, Any]] = None,
        reference: Optional[Path] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path: ...
    async def close(self) -> None: ...


# =========================
# Config + models
# =========================

class PollinationsConfig(BaseModel):
    api_base: str = Field(default_factory=lambda: _env_str("POLLINATIONS_API_BASE", "https://image.pollinations.ai").rstrip("/"))
    gen_base: str = Field(default_factory=lambda: _env_str("POLLINATIONS_GEN_BASE", "https://gen.pollinations.ai").rstrip("/"))
    secret: str = Field(default_factory=lambda: _env_str("POLLINATIONS_SECRET", ""))
    model: Optional[str] = Field(default_factory=lambda: (_env_str("POLLINATIONS_MODEL", "") or None))
    width: int = Field(default_factory=lambda: _env_int("POLLINATIONS_WIDTH", 1024))
    height: int = Field(default_factory=lambda: _env_int("POLLINATIONS_HEIGHT", 1024))
    use_v1: bool = Field(default_factory=lambda: _env_bool01("POLLINATIONS_USE_V1", 1))
    size_override: str = Field(default_factory=lambda: _env_str("POLLINATIONS_SIZE", ""))
    v1_edits_path: str = Field(default_factory=lambda: _env_str("POLLINATIONS_V1_IMAGES_EDITS_ENDPOINT", "/v1/images/edits"))
    v1_generations_path: str = Field(default_factory=lambda: _env_str("POLLINATIONS_V1_IMAGES_GENERATIONS_ENDPOINT", "/v1/images/generations"))
    prompt_suffix_style_only: str = Field(default_factory=lambda: _env_str("POLLINATIONS_STYLE_SUFFIX", "adopt the exact visual style, colors, and textures from the reference image; only transfer style, not content."))
    allow_cloud: bool = Field(default_factory=lambda: _env_bool01("ALLOW_CLOUD_IMAGE_BACKEND", 1))
    nologo: bool = Field(default_factory=lambda: _env_bool01("POLLINATIONS_NOLOGO", 1))
    seed_raw: Optional[str] = Field(default_factory=lambda: os.getenv("POLLINATIONS_SEED"))

    @property
    def seed(self) -> Optional[int]:
        if self.seed_raw is None:
            return None
        try:
            return int(self.seed_raw)
        except Exception:
            return None

    def require_cloud_enabled(self) -> None:
        if not self.allow_cloud:
            raise RuntimeError("Cloud image backend not allowed (set ALLOW_CLOUD_IMAGE_BACKEND=1)")
        if not self.secret and self.use_v1:
            raise RuntimeError("POLLINATIONS_SECRET missing for v1 usage (set POLLINATIONS_USE_V1=0 for legacy GET)")

class _V1Datum(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None

class _V1Response(BaseModel):
    created: Optional[int] = None
    data: List[_V1Datum] = Field(default_factory=list)


# =========================
# Backend
# =========================

class PollinationsBackend(ImageBackend):
    """
    Pollinations Cloud backend.
    - v1 Generations/Edits via Bearer Secret with robust retries
    - Legacy GET fallback to ensure availability
    """

    def __init__(self) -> None:
        self.cfg = PollinationsConfig()

    async def close(self) -> None:
        return None

    async def _v1_generations(self, prompt: str, width: int, height: int) -> Path:
        self.cfg.require_cloud_enabled()
        url = f"{self.cfg.gen_base}{self.cfg.v1_generations_path}"
        headers = {"Authorization": f"Bearer {self.cfg.secret}", "Content-Type": "application/json"}
        payload = {
            "model": self.cfg.model or "flux",
            "prompt": prompt,
            "size": (self.cfg.size_override or _size_from_wh(width, height)),
        }
        async with httpx.AsyncClient(timeout=_timeout_default(), limits=_httpx_limits()) as client:
            delay = 1.0
            last_exc: Optional[Exception] = None
            for attempt in range(1, 6):
                try:
                    r = await client.post(url, headers=headers, json=payload)
                    r.raise_for_status()
                    parsed = _V1Response.model_validate(r.json())
                    if not parsed.data:
                        raise RuntimeError("pollinations_v1_empty_data")
                    first = parsed.data[0]
                    if first.b64_json:
                        raw = b64decode(first.b64_json, validate=True)
                        return await _download_and_store_image_bytes(raw, suggested_ext=".jpg")
                    if first.url and first.url.startswith("http"):
                        rr = await _retrying_request("GET", first.url, client=client, timeout=_timeout_default(), follow_redirects=True)
                        return await _download_and_store_image_bytes(rr.content, suggested_ext=_pick_ext_from_content_type(rr.headers.get("Content-Type", "")))
                    raise RuntimeError("pollinations_v1_missing_data")
                except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError, httpx.HTTPStatusError, ValidationError) as e:
                    last_exc = e
                    if attempt < 5:
                        if _debug():
                            print(f"[POLLINATIONS][v1-gen] attempt {attempt} failed: {e}; retry in {delay:.2f}s")
                        await asyncio.sleep(delay)
                        delay *= 1.7
                    continue
        raise RuntimeError(f"pollinations_v1_generations_failed: {last_exc}")

    async def _v1_edits_url(self, prompt: str, image_url: str, width: int, height: int, *, negative_prompt: Optional[str] = None, seed: Optional[int] = None) -> Path:
        self.cfg.require_cloud_enabled()
        url = f"{self.cfg.gen_base}{self.cfg.v1_edits_path}"
        headers = {"Authorization": f"Bearer {self.cfg.secret}", "Content-Type": "application/json"}
        payload: Dict[str, Any] = {
            "model": self.cfg.model or "flux",
            "prompt": prompt,
            "image": image_url,
            "size": (self.cfg.size_override or _size_from_wh(width, height)),
            "response_format": "url",
        }
        if negative_prompt and negative_prompt.strip():
            payload["negative_prompt"] = negative_prompt.strip()
        if seed is not None:
            payload["seed"] = int(seed)
        async with httpx.AsyncClient(timeout=_timeout_default(), limits=_httpx_limits(), follow_redirects=True) as client:
            r = await _retrying_request("POST", url, client=client, json_payload=payload, headers=headers, timeout=_timeout_default())
            j = r.json()
            data = j.get("data")
            if isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, dict):
                    if "url" in first and isinstance(first["url"], str) and first["url"].startswith("http"):
                        rr = await _retrying_request("GET", first["url"], client=client, timeout=_timeout_default(), follow_redirects=True)
                        return await _download_and_store_image_bytes(rr.content, suggested_ext=_pick_ext_from_content_type(rr.headers.get("Content-Type", "")))
                    if "b64_json" in first and isinstance(first["b64_json"], str):
                        raw = b64decode(first["b64_json"], validate=True)
                        return await _download_and_store_image_bytes(raw, suggested_ext=".jpg")
        raise RuntimeError("pollinations_v1_edits_url_failed")

    async def _v1_edits_multipart(self, prompt: str, image_path: Path, width: int, height: int, *, negative_prompt: Optional[str] = None, seed: Optional[int] = None) -> Path:
        self.cfg.require_cloud_enabled()
        if not image_path.exists() or not image_path.is_file():
            raise FileNotFoundError(image_path)
        url = f"{self.cfg.gen_base}{self.cfg.v1_edits_path}"
        headers = {"Authorization": f"Bearer {self.cfg.secret}"}
        files: Dict[str, Any] = {
            "prompt": (None, (prompt or "").strip()),
            "response_format": (None, "url"),
            "n": (None, "1"),
            "model": (None, (self.cfg.model or "flux")),
            "size": (None, (self.cfg.size_override or _size_from_wh(width, height))),
            "image": (image_path.name, image_path.read_bytes(), "application/octet-stream"),
        }
        if negative_prompt and negative_prompt.strip():
            files["negative_prompt"] = (None, negative_prompt.strip())
        if seed is not None:
            files["seed"] = (None, str(int(seed)))
        async with httpx.AsyncClient(timeout=_timeout_default(), limits=_httpx_limits(), follow_redirects=True) as client:
            r = await _retrying_request("POST", url, client=client, files=files, headers=headers, timeout=_timeout_default())
            j = r.json()
            data = j.get("data")
            if isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, dict):
                    if "url" in first and isinstance(first["url"], str) and first["url"].startswith("http"):
                        rr = await _retrying_request("GET", first["url"], client=client, timeout=_timeout_default(), follow_redirects=True)
                        return await _download_and_store_image_bytes(rr.content, suggested_ext=_pick_ext_from_content_type(rr.headers.get("Content-Type", "")))
                    if "b64_json" in first and isinstance(first["b64_json"], str):
                        raw = b64decode(first["b64_json"], validate=True)
                        return await _download_and_store_image_bytes(raw, suggested_ext=".jpg")
        raise RuntimeError("pollinations_v1_edits_multipart_failed")

    def _legacy_url(self, prompt: str, width: int, height: int) -> str:
        # GET {api_base}/image/{encoded}?model=&width=&height=&nologo=&seed=
        from urllib.parse import quote, urlencode
        base = self.cfg.api_base.rstrip("/")
        encoded_prompt = quote(prompt, safe="")
        url = f"{base}/image/{encoded_prompt}"
        params: Dict[str, str] = {}
        if self.cfg.model:
            params["model"] = self.cfg.model
        if width and width > 0:
            params["width"] = str(width)
        if height and height > 0:
            params["height"] = str(height)
        if self.cfg.nologo:
            params["nologo"] = "true"
        if self.cfg.seed is not None:
            params["seed"] = str(self.cfg.seed)
        if self.cfg.secret and not self.cfg.use_v1:
            params["key"] = self.cfg.secret
        if params:
            url = f"{url}?{urlencode(params)}"
        return url

    async def _legacy_get(self, prompt: str, width: int, height: int) -> Path:
        url = self._legacy_url(prompt, width, height)
        async with httpx.AsyncClient(timeout=_timeout_default(), limits=_httpx_limits(), follow_redirects=True) as client:
            r = await _retrying_request("GET", url, client=client, timeout=_timeout_default(), follow_redirects=True)
            ext = _pick_ext_from_content_type(r.headers.get("Content-Type", ""))
            content = r.content
            if not content or len(content) < 1024:
                raise RuntimeError("pollinations_legacy_content_too_small")
            return await _download_and_store_image_bytes(content, suggested_ext=ext)

    def _merge_style_descriptors(self, prompt: str, extra: Optional[Dict[str, Any]]) -> str:
        if not extra:
            return prompt
        ds = extra.get("style_descriptors")
        if not isinstance(ds, list):
            return prompt
        vals = [s for s in ds if isinstance(s, str) and s.strip()]
        if not vals:
            return prompt
        merged = (prompt.rstrip(",") + ", " + ", ".join(vals)).strip().strip(",")
        return merged

    async def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 768,
        height: int = 512,
        style: Optional[Dict[str, Any]] = None,
        reference: Optional[Path] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        # 1) Prompt merging (style descriptors and negative suffix)
        full_prompt = self._merge_style_descriptors((prompt or "").strip(), extra)
        full_prompt = _merge_negative_into_prompt(full_prompt, negative_prompt)

        # 2) Prefer style-transfer (edits) when reference exists
        eff_w = width if width and width > 0 else self.cfg.width
        eff_h = height if height and height > 0 else self.cfg.height

        style_reference_url: Optional[str] = None
        if extra and isinstance(extra.get("style_reference_url"), str):
            style_reference_url = extra["style_reference_url"].strip() or None

        # For style-transfer, append the style-only suffix
        if (reference and reference.exists()) or style_reference_url:
            suffix = (self.cfg.prompt_suffix_style_only or "").strip()
            if suffix:
                full_prompt = f"{full_prompt}\n{suffix}"

        # 2a) Local multipart edits
        if reference and reference.exists() and self.cfg.use_v1 and self.cfg.secret:
            seed_override: Optional[int] = extra.get("seed") if isinstance(extra, dict) else None
            try:
                return await self._v1_edits_multipart(full_prompt, reference, eff_w, eff_h, negative_prompt=negative_prompt, seed=seed_override or self.cfg.seed)
            except Exception as e:
                if _debug():
                    print(f"[POLLINATIONS][multipart-edits] failed: {e}; fallback to url-edits/gen")

        # 2b) Remote-URL edits
        if style_reference_url and self.cfg.use_v1 and self.cfg.secret:
            seed_override: Optional[int] = extra.get("seed") if isinstance(extra, dict) else None
            try:
                return await self._v1_edits_url(full_prompt, style_reference_url, eff_w, eff_h, negative_prompt=negative_prompt, seed=seed_override or self.cfg.seed)
            except Exception as e:
                if _debug():
                    print(f"[POLLINATIONS][url-edits] failed: {e}; fallback to gen")

        # 3) Plain generations (v1 preferred, else legacy)
        if self.cfg.use_v1 and self.cfg.secret:
            try:
                return await self._v1_generations(full_prompt, eff_w, eff_h)
            except Exception as e:
                if _debug():
                    print(f"[POLLINATIONS][v1-gen] failed: {e}; fallback to legacy GET")
                return await self._legacy_get(full_prompt, eff_w, eff_h)
        else:
            return await self._legacy_get(full_prompt, eff_w, eff_h)
