# slAIdshow : image_backend.py
from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Any, Tuple, Dict

import httpx
from pydantic import BaseModel, Field, ValidationError

from comfyui_bridge import generate_from_prompt_dict


# --- Debug flag (optional minimal logging) ---
def _debug() -> bool:
    return (os.getenv("APP_IMAGE_BACKEND_DEBUG", "0") or "").strip().lower() in {"1", "true", "yes", "on"}


# --- Environment helpers ---
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


# --- HTTP client tuning ---
def _httpx_limits() -> httpx.Limits:
    return httpx.Limits(max_keepalive_connections=10, max_connections=20, keepalive_expiry=30.0)

def _timeout_short() -> httpx.Timeout:
    return httpx.Timeout(connect=3.0, read=6.0, write=4.0, pool=4.0)

def _timeout_long(total: float) -> httpx.Timeout:
    total = max(10.0, min(total, 240.0))
    return httpx.Timeout(connect=8.0, read=total, write=8.0, pool=8.0)


# --- Utility ---
def _clamp_dim(v: Optional[int]) -> Optional[int]:
    if v is None:
        return None
    x = max(64, min(2048, int(v)))
    return x - (x % 8)

def _now() -> float:
    return time.time()


def _is_in_allowed_subnets(ip: str, subnets_str: str) -> bool:
    try:
        ip_addr = ipaddress.ip_address(ip)
    except Exception:
        return False
    parts = [p.strip() for p in (subnets_str or "").replace(",", " ").split() if p.strip()]
    for cidr in parts:
        try:
            net = ipaddress.ip_network(cidr, strict=False)
            if ip_addr in net:
                return True
        except Exception:
            continue
    return False


def _assert_image_backend_host_policy(host: str) -> None:
    if host in {"127.0.0.1", "localhost"}:
        return
    allow_remote = _env_bool01("APP_ALLOW_REMOTE_BACKENDS", 0)
    if not allow_remote:
        raise AssertionError(f"Only localhost allowed, got {host}")
    subnets = _env_str("APP_ALLOWED_SUBNETS", "")
    if not subnets:
        return
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return
    if not _is_in_allowed_subnets(host, subnets):
        raise AssertionError(f"Remote host {host} not in allowed subnets ({subnets})")


def _clamp8(v: int) -> int:
    v = max(64, min(4096, int(v)))
    return v - (v % 8)


def _env_opt_int(name: str) -> Optional[int]:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _resolve_size_for_backend(backend_name: str, req_w: Optional[int], req_h: Optional[int]) -> Tuple[Optional[int], Optional[int]]:
    if isinstance(req_w, int) and req_w > 0 and isinstance(req_h, int) and req_h > 0:
        return _clamp8(req_w), _clamp8(req_h)
    gw = _env_opt_int("APP_IMAGE_WIDTH")
    gh = _env_opt_int("APP_IMAGE_HEIGHT")
    if gw and gh:
        return _clamp8(gw), _clamp8(gh)
    b = (backend_name or "").strip().lower()
    if b == "comfyui":
        cw = _env_opt_int("APP_COMFY_WIDTH")
        ch = _env_opt_int("APP_COMFY_HEIGHT")
        if cw and ch:
            return _clamp8(cw), _clamp8(ch)
    elif b == "pollinations":
        pw = _env_opt_int("POLLINATIONS_WIDTH")
        ph = _env_opt_int("POLLINATIONS_HEIGHT")
        if pw and ph:
            return _clamp8(pw), _clamp8(ph)
    return None, None


# --- Style runtime ---
@dataclass
class StyleRuntime:
    reference_path: Optional[Path] = None
    reference_strength: float = 0.6

    @property
    def has_reference(self) -> bool:
        return self.reference_path is not None and self.reference_path.exists() and self.reference_path.is_file()


# --- Backend interface ---
class ImageBackend:
    async def generate(self, prompt: str, width: int | None = None, height: int | None = None, negative_prompt: str | None = None, **kwargs: Any) -> Path:
        raise NotImplementedError


# --- Retry helpers (Pollinations) ---
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
    """Retry POST with backoff on 429/5xx and network timeouts."""
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
    """Retry GET with backoff for image download."""
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


# --- Small helpers ---
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


# --- Pollinations models ---
class _PollinationsV1Datum(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None

class _PollinationsV1Response(BaseModel):
    created: Optional[int] = None
    data: list[_PollinationsV1Datum] = Field(default_factory=list)

class PollinationsConfig(BaseModel):
    """
    Configuration for Pollinations image API access (V1).
    Requires ALLOW_CLOUD_IMAGE_BACKEND=1 and POLLINATIONS_SECRET.
    """
    api_base: str = Field(default_factory=lambda: _env_str("POLLINATIONS_API_BASE", "https://gen.pollinations.ai").rstrip("/"))
    gen_base: str = Field(default_factory=lambda: _env_str("POLLINATIONS_GEN_BASE", "https://gen.pollinations.ai").rstrip("/"))
    secret: str = Field(default_factory=lambda: _env_str("POLLINATIONS_SECRET", ""))
    model: Optional[str] = Field(default_factory=lambda: _env_str("POLLINATIONS_MODEL", "") or None)
    width: int = Field(default_factory=lambda: _env_int("POLLINATIONS_WIDTH", 1024))
    height: int = Field(default_factory=lambda: _env_int("POLLINATIONS_HEIGHT", 1024))
    nologo: bool = Field(default_factory=lambda: _env_bool01("POLLINATIONS_NOLOGO", 1))
    seed_raw: Optional[str] = Field(default_factory=lambda: os.getenv("POLLINATIONS_SEED"))
    use_v1: bool = Field(default_factory=lambda: _env_bool01("POLLINATIONS_USE_V1", 1))
    size_override: str = Field(default_factory=lambda: _env_str("POLLINATIONS_SIZE", ""))
    allow_cloud: bool = Field(default_factory=lambda: _env_bool01("ALLOW_CLOUD_IMAGE_BACKEND", 0))
    # Endpoints (align with style_engine)
    v1_edits_path: str = Field(default_factory=lambda: _env_str("POLLINATIONS_V1_IMAGES_EDITS_ENDPOINT", "/v1/images/edits"))
    v1_generations_path: str = Field(default_factory=lambda: _env_str("POLLINATIONS_V1_IMAGES_GENERATIONS_ENDPOINT", "/v1/images/generations"))
    # Prompt suffix for style-only transfer (URL mode or multipart)
    prompt_suffix_style_only: str = Field(default_factory=lambda: _env_str("POLLINATIONS_STYLE_SUFFIX", "adopt the exact visual style, colors, and textures from the reference image; only transfer style, not content."))

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
            raise RuntimeError("Cloud image backend not allowed (set ALLOW_CLOUD_IMAGE_BACKEND=1 to enable)")
        if not self.secret:
            raise RuntimeError("POLLINATIONS_SECRET missing in environment")


def _size_from_wh(width: int, height: int) -> str:
    if width > 0 and height > 0:
        return f"{width}x{height}"
    return "1024x1024"


class PollinationsBackend(ImageBackend):
    """
    Cloud backend for Pollinations (V1). Supports:
    - V1 images edits with reference image URL or multipart file
    - V1 images generations as text-only fallback
    - GET /image/... as last resort
    - Storage hook for style_engine direct V1 responses
    """
    def __init__(self, out_dir: Path, cfg: Optional[PollinationsConfig] = None, style: Optional[StyleRuntime] = None) -> None:
        self.out_dir = Path(out_dir).resolve()
        self.cfg = cfg or PollinationsConfig()
        self.style = style or StyleRuntime()
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def set_style_runtime(self, style: Optional[StyleRuntime]) -> None:
        self.style = style or StyleRuntime()

    async def store_generated_result(self, image_url: Optional[str], b64: Optional[str]) -> Path:
        """
        Storage hook used by style_engine:
        - If image_url is provided, download and save.
        - Else if b64 provided, decode and save.
        - Returns the local Path of the saved image.
        """
        if not (image_url or b64):
            raise RuntimeError("no result to store")
        target = self.out_dir / f"img_{uuid.uuid4().hex}.jpg"
        async with httpx.AsyncClient(timeout=_timeout_long(120.0), limits=_httpx_limits(), follow_redirects=True) as client:
            if image_url:
                r = await _retrying_get(client, image_url)
                content = r.content
                if not content or len(content) < 1024:
                    raise RuntimeError("downloaded image too small")
                target.write_bytes(content)
                return target
            else:
                from base64 import b64decode
                raw = b64decode(b64 or "", validate=True)
                if not raw or len(raw) < 1024:
                    raise RuntimeError("decoded image too small")
                target.write_bytes(raw)
                return target

    async def _post_v1_edits(self, prompt: str, image_url: str, width: int | None, height: int | None, *, negative_prompt: str | None = None, seed: Optional[int] = None) -> Path:
        """
        V1 Edits endpoint: JSON payload with image URL. Returns URL or base64; we handle both.
        """
        self.cfg.require_cloud_enabled()
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

        timeout = _timeout_long(180.0)
        async with httpx.AsyncClient(timeout=timeout, limits=_httpx_limits(), follow_redirects=True) as client:
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
                        img_url = first["url"]
                        ir = await _retrying_get(client, img_url)
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
        """
        V1 Edits endpoint: multipart upload with local image file.
        Aligns with style_engine multipart path (privacy-first).
        """
        self.cfg.require_cloud_enabled()
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

        timeout = _timeout_long(180.0)
        async with httpx.AsyncClient(timeout=timeout, limits=_httpx_limits(), follow_redirects=True, http2=True) as client:
            r = await _retrying_post(client, url, files=files, headers=headers)
            try:
                j = r.json()
            except Exception:
                raise RuntimeError("pollinations_v1_edits_multipart_invalid_json")

            # Parse both URL and b64_json cases
            data = j.get("data")
            if isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, dict):
                    if "url" in first and isinstance(first["url"], str) and first["url"].startswith("http"):
                        img_url = first["url"]
                        ir = await _retrying_get(client, img_url)
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

    async def _fetch_v1(self, prompt: str, width: int | None, height: int | None) -> Path:
        self.cfg.require_cloud_enabled()
        url = f"{self.cfg.gen_base}{self.cfg.v1_generations_path}"
        headers = {"Authorization": f"Bearer {self.cfg.secret}", "Content-Type": "application/json"}
        w = width if (width and width > 0) else self.cfg.width
        h = height if (height and height > 0) else self.cfg.height
        payload = {"model": self.cfg.model or "flux", "prompt": prompt, "size": (self.cfg.size_override or _size_from_wh(w, h))}
        timeout = _timeout_long(120.0)
        delay = 1.0
        last_exc: Optional[Exception] = None
        async with httpx.AsyncClient(timeout=timeout, limits=_httpx_limits()) as client:
            for attempt in range(1, 5 + 1):
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

    async def _fetch_get(self, prompt: str, width: int | None, height: int | None) -> Path:
        self.cfg.require_cloud_enabled()
        w = width if (width and width > 0) else self.cfg.width
        h = height if (height and height > 0) else self.cfg.height
        url = self._build_pollinations_image_url(self.cfg.gen_base, prompt, self.cfg.model, w, h, self.cfg.nologo, self.cfg.seed)
        params: dict[str, str] = {}
        if self.cfg.secret:
            params["key"] = self.cfg.secret
        timeout = _timeout_long(120.0)
        delay = 1.0
        last_exc: Optional[Exception] = None
        async with httpx.AsyncClient(timeout=timeout, limits=_httpx_limits(), follow_redirects=True) as client:
            for attempt in range(1, 4 + 1):
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

    async def generate(self, prompt: str, width: int | None = None, height: int | None = None, negative_prompt: str | None = None, **kwargs: Any) -> Path:
        """
        Generate via Pollinations V1:
        - If style_reference_path provided: use V1 edits (multipart, local file conditioning)
        - Else if style_reference_url provided: use V1 edits (image URL conditioning)
        - Else: V1 generations (text-only) or GET fallback
        """
        if not self.cfg.allow_cloud:
            raise RuntimeError("Cloud image backend disabled (ALLOW_CLOUD_IMAGE_BACKEND=0)")
        if not self.cfg.secret:
            raise RuntimeError("POLLINATIONS_SECRET missing")

        full_prompt = (prompt or "").strip()
        n_prompt = (negative_prompt or "").strip()
        if n_prompt:
            full_prompt = f"{full_prompt}\n-- negative: {n_prompt}"

        rw, rh = _resolve_size_for_backend("pollinations", width, height)
        eff_w = rw if (rw and rw > 0) else (width if (width and width > 0) else self.cfg.width)
        eff_h = rh if (rh and rh > 0) else (height if (height and height > 0) else self.cfg.height)

        # Multipart preference for privacy-first local reference
        style_reference_path: Optional[Path] = kwargs.get("style_reference_path")
        if isinstance(style_reference_path, Path):
            suffix = (self.cfg.prompt_suffix_style_only or "").strip()
            if suffix:
                full_prompt = f"{full_prompt}\n{suffix}"
            return await self._post_v1_edits_multipart(full_prompt, style_reference_path, eff_w, eff_h, negative_prompt=n_prompt or None, seed=self.cfg.seed)

        # URL-mode fallback/alternative
        style_reference_url: Optional[str] = kwargs.get("style_reference_url")
        if isinstance(style_reference_url, str) and style_reference_url.strip():
            suffix = (self.cfg.prompt_suffix_style_only or "").strip()
            if suffix:
                full_prompt = f"{full_prompt}\n{suffix}"
            return await self._post_v1_edits(full_prompt, style_reference_url.strip(), eff_w, eff_h, negative_prompt=n_prompt or None, seed=self.cfg.seed)

        # Text-only
        if self.cfg.use_v1:
            try:
                return await self._fetch_v1(full_prompt, eff_w, eff_h)
            except Exception as e:
                if _debug():
                    print(f"[POLLINATIONS][v1_failed] {type(e).__name__}: {e}")
                return await self._fetch_get(full_prompt, eff_w, eff_h)
        else:
            return await self._fetch_get(full_prompt, eff_w, eff_h)

    def _normalize_image_urls(self, image: str) -> str:
        """
        Falls Call-Site mehrere URLs mit Komma geliefert hat, normiere auf '|' Trenner.
        """
        s = (image or "").strip()
        if "," in s and "|" not in s:
            parts = [p.strip() for p in s.split(",") if p.strip()]
            return "|".join(parts)
        return s

    async def _post_v1_edits(self, prompt: str, image_url: str, width: int | None, height: int | None, *, negative_prompt: str | None = None, seed: Optional[int] = None) -> Path:
        self.cfg.require_cloud_enabled()
        url = f"{self.cfg.gen_base}{self.cfg.v1_edits_path}"
        headers = {"Authorization": f"Bearer {self.cfg.secret}", "Content-Type": "application/json"}
        w = width if (width and width > 0) else self.cfg.width
        h = height if (height and height > 0) else self.cfg.height

        normalized = self._normalize_image_urls(image_url)
        payload: dict[str, Any] = {
            "model": self.cfg.model or "flux",
            "prompt": prompt,
            "image": normalized,  # erlaubt 'url1|url2'
            "size": (self.cfg.size_override or _size_from_wh(w, h)),
            "response_format": "url",
        }
        if negative_prompt and negative_prompt.strip():
            payload["negative_prompt"] = negative_prompt.strip()
        if seed is not None:
            payload["seed"] = int(seed)

        timeout = _timeout_long(180.0)
        async with httpx.AsyncClient(timeout=timeout, limits=_httpx_limits(), follow_redirects=True) as client:
            r = await _retrying_post(client, url, json_payload=payload, headers=headers)


# --- ComfyUI models ---
class ComfyConfig(BaseModel):
    host: str = Field(default_factory=lambda: _env_str("APP_COMFY_HOST", "127.0.0.1"))
    port: int = Field(default_factory=lambda: _env_int("APP_COMFY_PORT", 8188))
    workflow_path: Path = Field(default_factory=lambda: Path(_env_str("APP_COMFY_WORKFLOW", "./workflows/text2img_SD15-FP16.json")).resolve())
    width: int = Field(default_factory=lambda: _env_int("APP_COMFY_WIDTH", int(_env_str("APP_IMAGE_WIDTH", "512") or "512")))
    height: int = Field(default_factory=lambda: _env_int("APP_COMFY_HEIGHT", int(_env_str("APP_IMAGE_HEIGHT", "512") or "512")))
    steps: int = Field(default_factory=lambda: _env_int("APP_COMFY_STEPS", 20))
    cfg: float = Field(default_factory=lambda: _env_float("APP_COMFY_CFG", 6.5))
    sampler: str = Field(default_factory=lambda: _env_str("APP_COMFY_SAMPLER", "euler"))
    timeout_sec: float = Field(default_factory=lambda: _env_float("APP_COMFY_TIMEOUT_SEC", 180.0))
    disabled: bool = Field(default_factory=lambda: _env_bool01("APP_DISABLE_COMFYUI", 1))
    comfy_output_dir: Optional[Path] = Field(default_factory=lambda: (Path(_env_str("APP_COMFY_OUTPUT_DIR", "")).resolve() if _env_str("APP_COMFY_OUTPUT_DIR", "") else None))
    comfy_input_dir: Optional[Path] = Field(default_factory=lambda: (Path(_env_str("APP_COMFY_INPUT_DIR", "")).resolve() if _env_str("APP_COMFY_INPUT_DIR", "") else None))
    negative: str = Field(default_factory=lambda: _env_str("APP_COMFY_NEGATIVE", "text, watermark, logo, low quality, blurry, bad anatomy"))

    node_id_positive: str = Field(default_factory=lambda: _env_str("APP_COMFY_NODE_POS", "2"))
    node_id_negative: str = Field(default_factory=lambda: _env_str("APP_COMFY_NODE_NEG", "3"))
    node_id_latent: str = Field(default_factory=lambda: _env_str("APP_COMFY_NODE_LATENT", "4"))

    node_id_ref_image: Optional[str] = Field(default_factory=lambda: (_env_str("APP_COMFY_NODE_REF_IMAGE", "") or None))
    node_id_ipadapter: Optional[str] = Field(default_factory=lambda: (_env_str("APP_COMFY_NODE_IPADAPTER", "") or None))
    node_key_ref_image_path: str = Field(default_factory=lambda: _env_str("APP_COMFY_KEY_REF_IMAGE_PATH", "image"))
    node_key_ref_weight: str = Field(default_factory=lambda: _env_str("APP_COMFY_KEY_REF_WEIGHT", "weight"))

    ref_mode: str = Field(default_factory=lambda: (_env_str("APP_COMFY_REF_MODE", "file") or "file").lower())
    node_id_ref_url: Optional[str] = Field(default_factory=lambda: (_env_str("APP_COMFY_NODE_REF_URL", "") or None))
    node_key_ref_url: str = Field(default_factory=lambda: _env_str("APP_COMFY_KEY_REF_URL", "url"))

    def assert_local(self) -> None:
        _assert_image_backend_host_policy(self.host)


class LocalComfyBackend(ImageBackend):
    def __init__(self, out_dir: Path, cfg: Optional[ComfyConfig] = None, style: Optional[StyleRuntime] = None) -> None:
        self.out_dir = Path(out_dir).resolve()
        self.cfg = cfg or ComfyConfig()
        self.style = style or StyleRuntime()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.assert_local()
        self._samplers_cache: Optional[Set[str]] = None

    def set_style_runtime(self, style: Optional[StyleRuntime]) -> None:
        self.style = style or StyleRuntime()

    async def _available(self) -> bool:
        if self.cfg.disabled:
            return False
        try:
            async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_short()) as c:
                r = await c.get(f"http://{self.cfg.host}:{self.cfg.port}/history")
                r.raise_for_status()
                return True
        except Exception:
            return False

    async def _fetch_valid_samplers(self) -> Set[str]:
        if self._samplers_cache is not None:
            return self._samplers_cache
        url = f"http://{self.cfg.host}:{self.cfg.port}/object_info/KSampler"
        try:
            async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_short()) as c:
                r = await c.get(url)
                r.raise_for_status()
                j = r.json()
                choices = j.get("input", {}).get("sampler_name", {}).get("choices", [])
                if isinstance(choices, list):
                    self._samplers_cache = {str(x) for x in choices}
                else:
                    self._samplers_cache = set()
        except Exception:
            self._samplers_cache = {
                "euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_sde",
                "dpmpp_2m_karras", "heun", "dpm_fast", "uni_pc"
            }
        return self._samplers_cache

    def _normalize_sampler(self, name: str) -> str:
        n = (name or "").strip().lower()
        if n in {"euler a", "euler_a", "euler-ancestral"}:
            return "euler_ancestral"
        return n

    def _load_prompt_file(self) -> dict:
        data = json.loads(self.cfg.workflow_path.read_text(encoding="utf-8"))
        if "prompt" in data and isinstance(data["prompt"], dict):
            return data["prompt"]
        return data

    def _override_text_nodes(self, prompt_dict: dict, positive: str, negative: str) -> None:
        pos_set = False
        neg_set = False
        node_pos = prompt_dict.get(self.cfg.node_id_positive)
        if isinstance(node_pos, dict) and node_pos.get("class_type") == "CLIPTextEncode":
            inputs = node_pos.get("inputs")
            if isinstance(inputs, dict) and "text" in inputs:
                inputs["text"] = positive
                pos_set = True
        node_neg = prompt_dict.get(self.cfg.node_id_negative)
        if isinstance(node_neg, dict) and node_neg.get("class_type") == "CLIPTextEncode":
            inputs = node_neg.get("inputs")
            if isinstance(inputs, dict) and "text" in inputs:
                inputs["text"] = negative
                neg_set = True
        if not (pos_set and neg_set):
            clip_nodes = []
            for node in prompt_dict.values():
                if isinstance(node, dict) and node.get("class_type") == "CLIPTextEncode":
                    clip_nodes.append(node)
            if clip_nodes and not pos_set:
                inputs = clip_nodes[0].get("inputs", {})
                if isinstance(inputs, dict) and "text" in inputs:
                    inputs["text"] = positive
                    pos_set = True
            if len(clip_nodes) > 1 and not neg_set:
                inputs = clip_nodes[1].get("inputs", {})
                if isinstance(inputs, dict) and "text" in inputs:
                    inputs["text"] = negative
                    neg_set = True

    def _override_dimensions_in_prompt(self, prompt_dict: dict, width: int, height: int) -> None:
        node_latent = prompt_dict.get(self.cfg.node_id_latent)
        if isinstance(node_latent, dict):
            cls = str(node_latent.get("class_type") or node_latent.get("class", "")).strip()
            if cls in {"EmptyLatentImage", "EmptyLatentImageBatch", "LatentImage", "CreateLatentImage"}:
                inputs = node_latent.get("inputs")
                if isinstance(inputs, dict):
                    if "width" in inputs:
                        inputs["width"] = width
                    if "height" in inputs:
                        inputs["height"] = height
                    return
        for node in prompt_dict.values():
            if not isinstance(node, dict):
                continue
            cls = str(node.get("class_type") or node.get("class", "")).strip()
            if cls in {"EmptyLatentImage", "EmptyLatentImageBatch", "LatentImage", "CreateLatentImage"}:
                inputs = node.get("inputs")
                if isinstance(inputs, dict):
                    if "width" in inputs:
                        inputs["width"] = width
                    if "height" in inputs:
                        inputs["height"] = height
            if cls.startswith("KSampler"):
                inputs = node.get("inputs")
                if isinstance(inputs, dict):
                    if "width" in inputs:
                        inputs["width"] = width
                    if "height" in inputs:
                        inputs["height"] = height

    def _copy_to_comfy_input(self, src: Path) -> Path:
        try:
            if not self.cfg.comfy_input_dir:
                return src
            inp = self.cfg.comfy_input_dir
            inp.mkdir(parents=True, exist_ok=True)
            dst = inp / src.name
            if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
                shutil.copy2(src, dst)
            return dst
        except Exception as e:
            if _debug():
                print(f"[COMFY][style] copy to input failed: {e}")
            return src

    def _path_strategy_for_comfy_input(self, staged_path: Path) -> Tuple[str, str]:
        abs_posix = staged_path.resolve().as_posix()
        if self.cfg.comfy_input_dir:
            return abs_posix, staged_path.name
        return abs_posix, abs_posix

    def _build_signed_url(self, basename: str) -> str:
        try:
            from app import build_signed_url
        except Exception as e:
            raise RuntimeError(f"build_signed_url unavailable: {e}")
        return build_signed_url(basename)

    def _inject_style_reference_file(self, prompt_dict: dict) -> None:
        if not (self.style and self.style.has_reference):
            return
        ref_path_abs = self.style.reference_path.resolve()
        ref_staged = self._copy_to_comfy_input(ref_path_abs)
        _, node_value = self._path_strategy_for_comfy_input(ref_staged)
        weight = float(max(0.0, min(1.0, self.style.reference_strength)))
        if _debug():
            print(f"[COMFY][style:file] node_val={node_value} w={weight}")

        if self.cfg.node_id_ref_image:
            node = prompt_dict.get(self.cfg.node_id_ref_image)
            if isinstance(node, dict):
                inputs = node.get("inputs")
                if isinstance(inputs, dict):
                    key = self.cfg.node_key_ref_image_path or "image"
                    if key in inputs:
                        inputs[key] = node_value
        if self.cfg.node_id_ipadapter:
            node = prompt_dict.get(self.cfg.node_id_ipadapter)
            if isinstance(node, dict):
                inputs = node.get("inputs")
                if isinstance(inputs, dict):
                    k = self.cfg.node_key_ref_weight or "weight"
                    if k in inputs:
                        inputs[k] = weight

        for node in prompt_dict.values():
            if not isinstance(node, dict):
                continue
            cls = str(node.get("class_type") or node.get("class", "")).strip()
            inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
            if cls in {"LoadImage", "ImageFromPath", "LoadImageMask"}:
                if "image" in inputs and isinstance(inputs.get("image"), (str, type(None))):
                    inputs["image"] = node_value
            if "ipadapter" in cls.lower() or cls in {"IPAdapter", "IPAdapterModelApply", "IPAdapterAdvanced"}:
                if "weight" in inputs:
                    try:
                        inputs["weight"] = weight
                    except Exception:
                        pass
                if "image" in inputs and isinstance(inputs.get("image"), (str, type(None))):
                    inputs["image"] = node_value

    def _inject_style_reference_url(self, prompt_dict: dict) -> None:
        if not (self.style and self.style.has_reference):
            return
        basename = self.style.reference_path.resolve().name
        try:
            signed = self._build_signed_url(basename)
        except Exception as e:
            raise RuntimeError(f"ref_url_build_failed: {e}")
        if _debug():
            print(f"[COMFY][style:url] basename={basename} url={signed}")

        node_id = self.cfg.node_id_ref_url or self.cfg.node_id_ref_image
        if node_id:
            node = prompt_dict.get(node_id)
            if isinstance(node, dict):
                inputs = node.get("inputs")
                if isinstance(inputs, dict):
                    key = self.cfg.node_key_ref_url or "url"
                    inputs[key] = signed
                    return

        for node in prompt_dict.values():
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs")
            if not isinstance(inputs, dict):
                continue
            if "url" in inputs and isinstance(inputs.get("url"), (str, type(None))):
                inputs["url"] = signed
                return

        if _debug():
            print("[COMFY][style:url] no suitable node found for URL injection")

    def _copy_latest_from_comfy(self, since_ts: float) -> Optional[Path]:
        src_dir = self.cfg.comfy_output_dir
        if not src_dir or not src_dir.exists():
            return None
        candidates = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            candidates.extend(src_dir.rglob(ext))
        if not candidates:
            return None
        recent = [p for p in candidates if p.is_file() and p.stat().st_mtime >= since_ts - 0.8]
        if not recent:
            return None
        latest = max(recent, key=lambda p: p.stat().st_mtime)
        target = self.out_dir / f"img_{uuid.uuid4().hex}{latest.suffix.lower()}"
        try:
            shutil.copy2(latest, target)
            if target.stat().st_size < 1024:
                return None
            return target
        except Exception:
            return None

    async def _short_history_hint(self) -> str:
        try:
            async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_short()) as c:
                r = await c.get(f"http://{self.cfg.host}:{self.cfg.port}/history")
                if not r.is_success:
                    return f"history_http_{r.status_code}"
                j = r.json()
                keys = list(j.keys())
                if not keys:
                    return "history_empty"
                last = j.get(keys[-1], {})
                status = last.get("status", {})
                prompt_errors = status.get("prompt_errors") or []
                if prompt_errors:
                    txt = str(prompt_errors[0])
                    return f"prompt_error: {txt[:180]}"
                return "history_ok"
        except Exception as e:
            return f"history_exc:{type(e).__name__}"

    async def generate(self, prompt: str, width: int | None = None, height: int | None = None, negative_prompt: str | None = None, **kwargs: Any) -> Path:
        if self.cfg.allow_cloud is False:
            raise RuntimeError("Cloud image backend disabled (ALLOW_CLOUD_IMAGE_BACKEND=0)")
        if not self.cfg.secret:
            raise RuntimeError("POLLINATIONS_SECRET missing")

        full_prompt = (prompt or "").strip()
        n_prompt = (negative_prompt or "").strip()
        if n_prompt:
            full_prompt = f"{full_prompt}\n-- negative: {n_prompt}"

        rw, rh = _resolve_size_for_backend("pollinations", width, height)
        eff_w = rw if (rw and rw > 0) else (width if (width and width > 0) else self.cfg.width)
        eff_h = rh if (rh and rh > 0) else (height if (height and height > 0) else self.cfg.height)

        # Preferred: multipart with local reference (privacy-first)
        style_reference_path: Optional[Path] = kwargs.get("style_reference_path")
        if isinstance(style_reference_path, Path):
            suffix = (self.cfg.prompt_suffix_style_only or "").strip()
            if suffix:
                full_prompt = f"{full_prompt}\n{suffix}"
            return await self._post_v1_edits_multipart(full_prompt, style_reference_path, eff_w, eff_h, negative_prompt=n_prompt or None, seed=self.cfg.seed)

        # URL-mode if provided
        style_reference_url: Optional[str] = kwargs.get("style_reference_url")
        if isinstance(style_reference_url, str) and style_reference_url.strip():
            suffix = (self.cfg.prompt_suffix_style_only or "").strip()
            if suffix:
                full_prompt = f"{full_prompt}\n{suffix}"
            return await self._post_v1_edits(full_prompt, style_reference_url.strip(), eff_w, eff_h, negative_prompt=n_prompt or None, seed=self.cfg.seed)

        # Text-only
        if self.cfg.use_v1:
            try:
                return await self._fetch_v1(full_prompt, eff_w, eff_h)
            except Exception as e:
                if _debug():
                    print(f"[POLLINATIONS][v1_failed] {type(e).__name__}: {e}")
                return await self._fetch_get(full_prompt, eff_w, eff_h)
        else:
            return await self._fetch_get(full_prompt, eff_w, eff_h)


# --- Backend factory ---
class BackendEnv(BaseModel):
    image_backend: str = Field(default_factory=lambda: _env_str("IMAGE_BACKEND", "comfyui").lower())
    allow_cloud: bool = Field(default_factory=lambda: _env_bool01("ALLOW_CLOUD_IMAGE_BACKEND", 0))
    output_dir: Path = Field(default_factory=lambda: Path(_env_str("APP_OUTPUT_DIR", "./outputs/images")).resolve())

def build_image_backend(style: Optional[StyleRuntime] = None) -> ImageBackend:
    env = BackendEnv()
    out_dir = env.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if env.image_backend == "comfyui":
        cfg = ComfyConfig()
        return LocalComfyBackend(out_dir=out_dir, cfg=cfg, style=style)
    elif env.image_backend == "pollinations":
        cfg = PollinationsConfig()
        cfg.allow_cloud = env.allow_cloud
        return PollinationsBackend(out_dir=out_dir, cfg=cfg, style=style)
    else:
        raise RuntimeError(f"Unsupported IMAGE_BACKEND={env.image_backend}")
