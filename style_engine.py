#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

# Comments in English; short German notes where logic is complex.

import asyncio
import base64
import contextlib
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from pydantic import BaseModel, Field, ConfigDict, field_validator, HttpUrl, ValidationError

# =========================
# Environment helpers
# =========================

def _env_str(k: str, d: str = "") -> str:
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

def _debug() -> bool:
    return _env_bool01("STYLE_DEBUG", 0)

# =========================
# Data models
# =========================

class BuiltPrompt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    positive: str = Field(default="")
    negative: str = Field(default="")

class StyleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # High-level stylistic controls
    style_preset: str = Field(default="photo")
    style_details: str = Field(default="")
    negative_base: str = Field(default="")
    color_scheme: str = Field(default="")

    # Reference-based styling
    use_reference: bool = Field(default=False)
    reference_id: Optional[str] = Field(default=None)
    reference_strength: float = Field(default=0.6, ge=0.0, le=1.0)

    # Cloud vision toggle (UI-controlled): if True and model is set, try Vision descriptors
    reference_cloud: bool = Field(default=False)

    # Internal
    persisted_path: Optional[Path] = Field(default=None, exclude=True)

    @field_validator("style_preset")
    @classmethod
    def _normalize_preset(cls, v: str) -> str:
        return (v or "").strip() or "photo"

    @field_validator("style_details", "negative_base", "color_scheme")
    @classmethod
    def _normalize_optional_text(cls, v: str) -> str:
        return (v or "").strip()

# =========================
# Reference file store
# =========================

class ReferenceStore:
    """Local reference image store with strict filename handling."""

    def __init__(self, base_dir: Union[str, Path]) -> None:
        self.base = Path(base_dir).resolve()
        self.base.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_basename(name: str) -> str:
        """Allow only [A-Za-z0-9._-] and forbid path separators."""
        if not name:
            raise ValueError("empty name")
        if "/" in name or "\\" in name:
            raise ValueError("invalid path separator")
        if not re.fullmatch(r"[A-Za-z0-9._\-]+", name):
            raise ValueError("illegal characters")
        return name

    def put(self, filename: str, data: bytes) -> Tuple[str, Path]:
        """Store bytes under a safe unique basename. Returns (basename, full_path)."""
        if not isinstance(data, (bytes, bytearray)):
            raise ValueError("data must be bytes")
        try:
            base_raw = filename or "ref.png"
            base = self._safe_basename(Path(base_raw).name)
        except Exception:
            base = "ref.png"

        target = self.base / base
        if target.exists():
            stem = target.stem
            suf = target.suffix or ".png"
            for i in range(1, 1000):
                cand = self.base / f"{stem}_{i}{suf}"
                if not cand.exists():
                    target = cand
                    break

        Path(target).write_bytes(bytes(data))
        return (target.name, target)

    def get_path(self, reference_id: str) -> Path:
        """Resolve a stored basename to an absolute path within the store root."""
        base = self._safe_basename(reference_id)
        p = (self.base / base).resolve()
        if p.parent != self.base or not p.exists() or not p.is_file():
            raise FileNotFoundError(f"reference not found: {reference_id}")
        return p

# =========================
# Style prompt builder
# =========================

def _compose_positive(user_topic: str, cfg: StyleConfig) -> str:
    """Compose positive prompt text from user topic, preset, style details, and color scheme."""
    topic = (user_topic or "").strip()
    parts: List[str] = []
    if topic:
        parts.append(topic)
    preset = (cfg.style_preset or "photo").lower()
    if preset == "photo":
        parts.append("high-quality photograph, realistic lighting, crisp focus")
    elif preset == "illustration":
        parts.append("clean illustration, vector-like clarity, balanced composition")
    elif preset == "watercolor":
        parts.append("soft watercolor style, gentle gradients, textured paper look")
    elif preset == "sketch":
        parts.append("pencil sketch, fine lines, cross-hatching, monochrome")
    else:
        parts.append("detailed, balanced composition")

    if cfg.style_details:
        parts.append(cfg.style_details)
    if cfg.color_scheme:
        parts.append(f"color scheme: {cfg.color_scheme}")

    positive = ", ".join([p for p in parts if p])
    positive = re.sub(r"\s{2,}", " ", positive).strip().strip(",")
    return positive

def _compose_negative(cfg: StyleConfig) -> str:
    """Merge user-provided negative_base with a sensible default list, deduplicated case-insensitively."""
    base = (cfg.negative_base or "").strip()
    default_neg = "low quality, blurry, overexposed, underexposed, watermark, text artifacts"
    if base:
        items = [s.strip() for s in (base.split(",") + default_neg.split(","))]
        seen: set[str] = set()
        merged: List[str] = []
        for it in items:
            itn = it.lower()
            if not itn or itn in seen:
                continue
            seen.add(itn)
            merged.append(it)
        return ", ".join(merged)
    return default_neg

def build_prompt(user_text: str, cfg: StyleConfig) -> BuiltPrompt:
    pos = _compose_positive(user_text, cfg)
    neg = _compose_negative(cfg)
    return BuiltPrompt(positive=pos, negative=neg)

build_style_prompt = build_prompt

# =========================
# ComfyUI integration (local)
# =========================

def prepare_backend_style(backend: Any, cfg: StyleConfig, refs_dir: Union[str, Path]) -> None:
    """
    Stage the local reference image into the LocalComfyBackend and, if supported by the backend,
    pass resolved style descriptors. Vision path is NOT used here to avoid async in sync contexts.
    """
    try:
        # Early exit when no reference is configured
        if not cfg or not getattr(cfg, "use_reference", False) or not getattr(cfg, "reference_id", None):
            return

        # Resolve reference path securely through the store
        store = ReferenceStore(refs_dir)
        ref_path = store.get_path(cfg.reference_id)
        if not ref_path.exists():
            print(f"[STYLE] reference file missing on disk: {ref_path}")
            return

        # Stage the image into a compatible backend (if method is available)
        if hasattr(backend, "stage_reference_image"):
            try:
                backend.stage_reference_image(ref_path, float(cfg.reference_strength))  # type: ignore[attr-defined]
                print("[STYLE] staged reference into LocalComfyBackend.")
            except Exception as e:
                print(f"[STYLE] backend staging failed: {e}")

        # For ComfyUI in a sync context, prefer local descriptors only (no cloud here).
        try:
            desc = _local_style_descriptors(ref_path)
            if hasattr(backend, "stage_style_descriptors") and desc:
                backend.stage_style_descriptors(desc)  # type: ignore[attr-defined]
                print(f"[STYLE] staged {len(desc)} style descriptors into LocalComfyBackend.")
        except Exception as e:
            print(f"[STYLE] descriptor staging skipped: {e}")

    except Exception as e:
        print(f"[STYLE] prepare_backend_style failed: {e}")

# =========================
# Pollinations integration (V1)
# =========================

POLLINATIONS_API_BASE: str = _env_str("POLLINATIONS_API_BASE", "https://gen.pollinations.ai")
POLLINATIONS_V1_IMAGES_EDITS_PATH: str = _env_str("POLLINATIONS_V1_IMAGES_EDITS_PATH", "/v1/images/edits")
POLLINATIONS_SECRET: str = _env_str("POLLINATIONS_SECRET", "")
POLLINATIONS_IMAGE_MODEL: str = _env_str("POLLINATIONS_IMAGE_MODEL", _env_str("POLLINATIONS_MODEL", "flux"))

FEATURE_POLLINATIONS_V1_EDITS: bool = _env_bool01("FEATURE_POLLINATIONS_V1_EDITS", 1)
FEATURE_POLLINATIONS_USE_URL_MODE: bool = _env_bool01("FEATURE_POLLINATIONS_USE_URL_MODE", 0)

ALLOW_CLOUD_IMAGE_BACKEND: bool = _env_bool01("ALLOW_CLOUD_IMAGE_BACKEND", 0)
POLLINATIONS_ENABLE_UPLOAD: bool = _env_bool01("POLLINATIONS_ENABLE_UPLOAD", 1)

APP_TIMEOUT_SEC: float = float(_env_float("APP_OLLAMA_TIMEOUT_SEC", 90.0))
APP_MAX_RETRIES: int = _env_int("APP_OLLAMA_MAX_RETRIES", 4)
APP_RETRY_BASE_DELAY: float = float(_env_float("APP_OLLAMA_RETRY_BASE_DELAY", 0.8))

APP_IMAGE_WIDTH: int = _env_int("APP_IMAGE_WIDTH", 1024)
APP_IMAGE_HEIGHT: int = _env_int("APP_IMAGE_HEIGHT", 1024)

HTTP_USER_AGENT: str = _env_str("APP_HTTP_USER_AGENT", "slAIDshow/2026 (pollinations-style)")

class PollinationsRefUploadResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    ok: bool = False
    url: Optional[str] = None
    size: Optional[int] = None
    error: Optional[str] = None

class PollinationsPromptPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    positive: str
    negative: str
    refs: List[str] = Field(default_factory=list)
    merged_prompt: str

class V1ImagesItem(BaseModel):
    url: Optional[HttpUrl] = None
    b64_json: Optional[str] = None

class V1ImagesResp(BaseModel):
    data: List[V1ImagesItem] = Field(default_factory=list)

def _httpx_limits() -> httpx.Limits:
    """Tuned connection pooling for bursty generation calls."""
    return httpx.Limits(max_keepalive_connections=6, max_connections=8, keepalive_expiry=20.0)

def _timeout_generic() -> httpx.Timeout:
    """Balanced timeouts for generation and uploads."""
    return httpx.Timeout(connect=8.0, read=APP_TIMEOUT_SEC, write=30.0, pool=8.0)

def _default_headers() -> Dict[str, str]:
    """Uniform User-Agent; Authorization is added separately when needed."""
    return {"User-Agent": HTTP_USER_AGENT}

def _v1_edits_endpoint() -> str:
    """Build the full V1 edits endpoint URL."""
    path = POLLINATIONS_V1_IMAGES_EDITS_PATH
    if not path.startswith("/"):
        path = "/" + path
    return f"{POLLINATIONS_API_BASE.rstrip('/')}{path}"

def _auth_headers() -> Dict[str, str]:
    """Add Bearer token from environment; raises when missing."""
    if not POLLINATIONS_SECRET:
        raise RuntimeError("POLLINATIONS_SECRET not set in environment")
    h = _default_headers()
    h["Authorization"] = f"Bearer {POLLINATIONS_SECRET}"
    return h

def _mime_for(name: str) -> str:
    """Map extension to mime; default to octet-stream for unknown types."""
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

def _parse_v1_images_response(resp: httpx.Response) -> Tuple[Optional[str], Optional[str]]:
    """Parse flexible Pollinations V1 images response."""
    try:
        js = resp.json()
    except Exception:
        return (None, None)
    # First try strict pydantic schema
    try:
        parsed = V1ImagesResp(**js)
        if parsed.data:
            item = parsed.data[0]
            return (str(item.url) if item.url else None, item.b64_json)
    except ValidationError:
        # Generic extraction for alternative response shapes
        try:
            url = js.get("url") or ((js.get("data") or [{}])[0] or {}).get("url")
            b64 = ((js.get("data") or [{}])[0] or {}).get("b64_json") or js.get("b64_json")
            if isinstance(url, str) and not url.startswith("http"):
                url = None
            if isinstance(b64, str) and len(b64) < 32:
                b64 = None
            return (url, b64)
        except Exception:
            return (None, None)
    return (None, None)

async def _post_with_retries(
    client: httpx.AsyncClient,
    url: str,
    *,
    files: Dict[str, Any],
    headers: Dict[str, str],
    max_retries: int = APP_MAX_RETRIES,
    base_delay: float = APP_RETRY_BASE_DELAY,
) -> httpx.Response:
    """POST with simple exponential backoff on transient network errors."""
    last_exc: Optional[Exception] = None
    delay = float(base_delay)
    for attempt in range(1, int(max_retries) + 1):
        try:
            return await client.post(url, files=files, headers=headers)
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            last_exc = e
            if attempt >= max_retries:
                break
            await asyncio.sleep(delay)
            delay *= 1.8
    raise RuntimeError(f"pollinations_post_failed after {max_retries} attempts: {last_exc}")

async def _v1_edits_multipart(
    prompt: str,
    image_file: Path,
    *,
    width: int,
    height: int,
    negative_prompt: str = "",
    seed: Optional[int] = None,
    model: str = POLLINATIONS_IMAGE_MODEL,
) -> Tuple[Optional[str], Optional[str]]:
    """Perform V1 image edit with a local file as multipart upload."""
    if not image_file.exists():
        raise FileNotFoundError(image_file)
    mime = _mime_for(image_file.name)
    files: Dict[str, Any] = {
        "prompt": (None, prompt),
        "response_format": (None, "url"),
        "n": (None, "1"),
        "model": (None, model),
        "size": (None, f"{width}x{height}"),
        "image": (image_file.name, image_file.read_bytes(), mime),
    }
    if negative_prompt:
        files["negative_prompt"] = (None, negative_prompt)
    if seed is not None:
        files["seed"] = (None, str(seed))

    async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_generic(), follow_redirects=True, http2=True, headers=_default_headers()) as client:
        print(f"[STYLE] pollinations: POST {_v1_edits_endpoint()} (multipart), model={model}")
        resp = await _post_with_retries(client, _v1_edits_endpoint(), files=files, headers=_auth_headers())
        if resp.status_code >= 400:
            raise RuntimeError(f"pollinations multipart error {resp.status_code}: {resp.text[:512]}")
        return _parse_v1_images_response(resp)

async def _v1_edits_url_mode(
    prompt: str,
    image_url: str,
    *,
    width: int,
    height: int,
    negative_prompt: str = "",
    seed: Optional[int] = None,
    model: str = POLLINATIONS_IMAGE_MODEL,
) -> Tuple[Optional[str], Optional[str]]:
    """Perform V1 image edit by providing an image URL (server-side fetch)."""
    # NOTE: Field name is 'image' to mirror working request shape.
    files: Dict[str, Any] = {
        "prompt": (None, prompt),
        "response_format": (None, "url"),
        "n": (None, "1"),
        "model": (None, model),
        "size": (None, f"{width}x{height}"),
        "image": (None, image_url),
    }
    if negative_prompt:
        files["negative_prompt"] = (None, negative_prompt)
    if seed is not None:
        files["seed"] = (None, str(seed))

    async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_generic(), follow_redirects=True, http2=True, headers=_default_headers()) as client:
        print(f"[STYLE] pollinations: POST {_v1_edits_endpoint()} (url-mode), model={model}")
        resp = await _post_with_retries(client, _v1_edits_endpoint(), files=files, headers=_auth_headers())
        if resp.status_code >= 400:
            raise RuntimeError(f"pollinations url-mode error {resp.status_code}: {resp.text[:512]}")
        return _parse_v1_images_response(resp)

# Legacy helper for media uploads (optional in some setups)
POLLINATIONS_UPLOAD_ENDPOINT: str = _env_str("POLLINATIONS_UPLOAD_ENDPOINT", "/upload")

def validate_image_bytes_minimal(data: bytes) -> None:
    """Minimal content and magic check to reject obvious non-images."""
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError("invalid bytes")
    if len(data) < 128:
        raise ValueError("image too small")
    head = data[:16]
    if not any((
        head.startswith(b"\x89PNG"),
        head.startswith(b"\xFF\xD8\xFF"),
        head.startswith(b"RIFF") and b"WEBP" in data[:64],
        head[:2] == b"BM",
    )):
        print("[STYLE] unknown image magic header; continuing")

async def pollinations_upload_ref(path: Path) -> PollinationsRefUploadResult:
    """Upload a local reference image to Pollinations media store, gated by ALLOW_CLOUD_IMAGE_BACKEND."""
    if not ALLOW_CLOUD_IMAGE_BACKEND:
        return PollinationsRefUploadResult(ok=False, error="cloud_disabled")
    if not POLLINATIONS_SECRET:
        return PollinationsRefUploadResult(ok=False, error="missing_secret")
    if not POLLINATIONS_ENABLE_UPLOAD:
        return PollinationsRefUploadResult(ok=False, error="upload_disabled")
    if not path.exists() or not path.is_file():
        return PollinationsRefUploadResult(ok=False, error="file_missing")

    data = path.read_bytes()
    try:
        validate_image_bytes_minimal(data)
    except Exception as e:
        return PollinationsRefUploadResult(ok=False, error=f"invalid_image:{e}", size=len(data))

    mime = _mime_for(path.name)
    neutral_name = "reference" + (Path(path.name).suffix.lower() or ".jpg")

    url = f"{POLLINATIONS_API_BASE.rstrip('/')}{POLLINATIONS_UPLOAD_ENDPOINT}"
    headers = _auth_headers()
    files = {"file": (neutral_name, data, mime)}

    async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_generic(), follow_redirects=True, http2=True, headers=_default_headers()) as client:
        try:
            resp = await _post_with_retries(client, url, files=files, headers=headers)
        except Exception as e:
            return PollinationsRefUploadResult(ok=False, error=f"network:{e}", size=len(data))

    try:
        js = resp.json()
    except Exception:
        return PollinationsRefUploadResult(ok=False, error="invalid_json", size=len(data))

    ok = bool(js.get("ok", True))
    media_url = js.get("url") or js.get("media_url") or js.get("href")
    if not isinstance(media_url, str) or not media_url.startswith("http"):
        return PollinationsRefUploadResult(ok=False, error="missing_media_url", size=len(data))

    return PollinationsRefUploadResult(ok=ok, url=media_url, size=len(data))

async def resolve_reference_urls_for_pollinations(
    cfg: StyleConfig,
    refs_dir: Union[str, Path],
) -> List[str]:
    """Return a list of remote reference URLs (at most one) for Pollinations, or empty on failure."""
    try:
        if not cfg or not getattr(cfg, "use_reference", False) or not getattr(cfg, "reference_id", None):
            return []
        store = ReferenceStore(refs_dir)
        ref_path = store.get_path(cfg.reference_id)
        up = await pollinations_upload_ref(ref_path)
        if up.ok and up.url:
            return [up.url]
        return []
    except Exception as e:
        print(f"[-STYLE] pollinations resolve failed: {e}")
        return []

def build_pollinations_prompt(
    user_text: str,
    cfg: StyleConfig,
    refs: List[str],
    negative_override: Optional[str] = None,
) -> PollinationsPromptPayload:
    """Build a merged prompt for Pollinations with optional ref-url prefixes."""
    base = build_prompt(user_text, cfg)
    negative = (negative_override or "").strip() or base.negative

    ref_prefixes: List[str] = []
    for u in refs:
        if not isinstance(u, str) or not u.startswith("http"):
            continue
        ref_prefixes.append(f"style guided by: {u}")

    parts: List[str] = []
    if ref_prefixes:
        parts.extend(ref_prefixes)
    if base.positive:
        parts.append(base.positive)

    merged_pos = "; ".join(parts).strip().strip(";")
    merged_full = merged_pos
    if negative:
        merged_full = f"{merged_pos}\n-- negative: {negative}".strip()

    return PollinationsPromptPayload(
        positive=merged_pos,
        negative=negative,
        refs=list(ref_prefixes),
        merged_prompt=merged_full,
    )

# =========================
# Cloud Vision + Local Style pipeline
# =========================

# Feature flag and endpoints for Ollama Vision.
FEATURE_STYLE_REF_VISION_CLOUD: bool = _env_bool01("FEATURE_STYLE_REF_VISION_CLOUD", 1)
APP_OLLAMA_HOST: str = _env_str("APP_OLLAMA_HOST", "http://localhost:11434")
APP_OLLAMA_VISION_MODEL: str = _env_str("APP_OLLAMA_VISION_MODEL", "")

# Best-effort import of local style features for fallback if cloud is off/unavailable.
with contextlib.suppress(Exception):
    from style_features import extract_style_descriptors  # type: ignore[attr-defined]

class _OllamaMsgContent(BaseModel):
    """Minimal multimodal content item (text or base64 image)."""
    type: str
    text: Optional[str] = None
    image: Optional[str] = None  # base64-encoded image data

class _OllamaMessage(BaseModel):
    role: str
    content: List[_OllamaMsgContent]

class _OllamaChatReq(BaseModel):
    model: str
    messages: List[_OllamaMessage]
    stream: bool = False
    options: Dict[str, Any] = Field(default_factory=dict)

class _OllamaChatRespMsg(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class _OllamaChatResp(BaseModel):
    message: Optional[_OllamaChatRespMsg] = None
    done: Optional[bool] = None

def _vision_instruction() -> str:
    """Deterministic instruction to request short, style-focused descriptors only."""
    return (
        "Analyze only the STYLE of this image. "
        "Return 4-6 short, comma-separated descriptors (no sentences, no extra words). "
        "Focus on: line clarity/thickness, color palette/saturation, halftone dots, straight lines, brush texture, contrast, micro-detail/bokeh. "
        "Output MUST be a single comma-separated line."
    )

def _normalize_descriptors_line(s: str, max_items: int = 6) -> List[str]:
    """Normalize and sanitize a descriptor CSV line into a compact unique list."""
    s = (s or "").strip().replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"[;:|]", ",", s)
    raw = [p.strip().strip(".") for p in s.split(",")]
    out: List[str] = []
    for p in raw:
        p = re.sub(r"[^A-Za-z0-9 _/\-]", "", p)
        if not p:
            continue
        if len(p) > 64:
            p = p[:64].rstrip()
        lo = p.lower()
        if lo in {"style", "descriptor", "descriptors", "none", "n/a"}:
            continue
        if p not in out:
            out.append(p)
        if len(out) >= max_items:
            break
    return out

def _encode_image_b64(path: Path, warn_threshold: int = 2_500_000, hard_limit: int = 8_000_000) -> Optional[str]:
    """Read image and return base64 ascii string; warn or skip when very large."""
    try:
        data = path.read_bytes()
        if len(data) > warn_threshold:
            print(f"[STYLE] warning: large image size {len(data)} bytes")
        # Hard limit guard to avoid huge JSON (optional)
        if len(data) > hard_limit and _env_bool01("STYLE_VISION_HARD_LIMIT_ON", 1):
            print("[STYLE] image too large for inlining to Ollama; skipping vision descriptors")
            return None
        return base64.b64encode(data).decode("ascii")
    except Exception as e:
        print(f"[STYLE] base64 encode failed: {e}")
        return None

async def _ollama_chat_descriptors(
    *,
    model: str,
    host: str,
    image_path: Path,
    retries: int,
    base_delay: float,
    timeout: httpx.Timeout,
) -> List[str]:
    """Call Ollama /api/chat with multimodal message to extract style descriptors."""
    if not image_path.exists():
        return []
    b64 = _encode_image_b64(image_path)
    if not b64:
        return []
    url = f"{host.rstrip('/')}/api/chat"
    payload = _OllamaChatReq(
        model=model,
        messages=[
            _OllamaMessage(
                role="user",
                content=[
                    _OllamaMsgContent(type="image", image=b64),
                    _OllamaMsgContent(type="text", text=_vision_instruction()),
                ],
            )
        ],
        stream=False,
        options={"temperature": 0.1},
    )
    delay = float(base_delay)
    last_exc: Optional[Exception] = None
    async with httpx.AsyncClient(limits=_httpx_limits(), timeout=timeout, follow_redirects=True, headers=_default_headers()) as client:
        for attempt in range(1, int(retries) + 1):
            try:
                resp = await client.post(url, json=payload.model_dump())
                if resp.status_code >= 400:
                    # Handle transient HTTP codes with backoff; fail fast on others.
                    if resp.status_code in (408, 429, 500, 502, 503, 504):
                        await asyncio.sleep(delay)
                        delay *= 1.8
                        continue
                    print(f"[STYLE] ollama http {resp.status_code}: {resp.text[:256]}")
                    return []
                js = resp.json()
                try:
                    parsed = _OllamaChatResp(**js)
                    content = parsed.message.content if (parsed.message and parsed.message.content) else None
                except ValidationError:
                    msg = js.get("message") or {}
                    content = msg.get("content") if isinstance(msg, dict) else None
                if not content:
                    return []
                return _normalize_descriptors_line(content, max_items=6)
            except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
                last_exc = e
                if attempt >= retries:
                    break
                await asyncio.sleep(delay)
                delay *= 1.8
            except Exception as e:
                last_exc = e
                break
    if last_exc:
        print(f"[STYLE] ollama call failed: {last_exc}")
    return []

def _local_style_descriptors(image_path: Path, limit: int = 6) -> List[str]:
    """Compute style descriptors locally via the optimized style_features pipeline."""
    try:
        if 'extract_style_descriptors' not in globals():
            return []
        ds = extract_style_descriptors(image_path, debug=False)  # type: ignore[name-defined]
        out: List[str] = []
        for d in ds:
            d = (d or "").strip()
            if not d:
                continue
            d = re.sub(r"[^A-Za-z0-9 _/\-]", "", d)
            if d and d not in out:
                out.append(d)
            if len(out) >= limit:
                break
        return out
    except Exception as e:
        print(f"[STYLE] local style_features failed: {e}")
        return []

async def resolve_style_descriptors_for_reference_async(
    *,
    ref_path: Path,
    prefer_cloud: bool,
) -> List[str]:
    """
    Async resolver for style descriptors. Prefers cloud Vision when enabled and configured,
    otherwise falls back to the local style pipeline.
    """
    if not ref_path or not ref_path.exists():
        return []
    if prefer_cloud and FEATURE_STYLE_REF_VISION_CLOUD and APP_OLLAMA_VISION_MODEL:
        cloud = await _ollama_chat_descriptors(
            model=APP_OLLAMA_VISION_MODEL,
            host=APP_OLLAMA_HOST,
            image_path=ref_path,
            retries=APP_MAX_RETRIES,
            base_delay=APP_RETRY_BASE_DELAY,
            timeout=_timeout_generic(),
        )
        if cloud:
            return cloud
    # Fallback to local descriptors
    return _local_style_descriptors(ref_path, limit=6)

def resolve_style_descriptors_for_reference_sync(
    *,
    ref_path: Path,
    prefer_cloud: bool,
) -> List[str]:
    """
    Sync wrapper intended for CLI/offline contexts. It MUST NOT be used inside a running event loop.
    In running loops (e.g., FastAPI handlers) the async version must be awaited directly.
    """
    if not ref_path or not ref_path.exists():
        return []
    # If an event loop is already running, refuse to block and instruct callers to use async.
    try:
        loop = asyncio.get_running_loop()
        # If this does not raise, we are in an active loop.
        raise RuntimeError("resolve_style_descriptors_for_reference_sync called inside running event loop")
    except RuntimeError as e:
        # In a running loop -> explicit signal
        if "running event loop" in str(e):
            print(f"[STYLE] descriptor staging skipped (sync in loop not allowed): {e}")
            return []
        # No running loop; proceed to run the async resolver
        return asyncio.run(resolve_style_descriptors_for_reference_async(ref_path=ref_path, prefer_cloud=prefer_cloud))
    except Exception:
        # No running loop; execute async resolver
        return asyncio.run(resolve_style_descriptors_for_reference_async(ref_path=ref_path, prefer_cloud=prefer_cloud))

def _merge_descriptors_into_prompt(user_text: str, base_positive: str, descriptors: List[str], lean: bool) -> str:
    """Append concise descriptors to either the lean or composed positive prompt."""
    descr = [d for d in descriptors if isinstance(d, str) and d]
    if not descr:
        return base_positive if not lean else (user_text.strip() or base_positive or "high-quality image")
    suffix = ", " + ", ".join(descr)
    if lean:
        base = user_text.strip() or base_positive.strip() or "high-quality image"
        return (base + suffix).strip().strip(",")
    return (base_positive.strip() + suffix).strip().strip(",")

# =========================
# Orchestration
# =========================

async def generate_image_pollinations(
    user_text: str,
    cfg: StyleConfig,
    backend: Any,
    refs_dir: Union[str, Path],
    width: Optional[int] = None,
    height: Optional[int] = None,
    negative_override: Optional[str] = None,
    seed: Optional[int] = None,
) -> Optional[Path]:
    """
    Pollinations V1 Edits orchestration with optional Vision descriptors (enabled by default).
    Falls back to local style pipeline when Vision is unavailable. Lean prompt can be enabled via env.
    """
    try:
        STYLE_REFERENCE_LEAN_PROMPT: bool = _env_bool01("STYLE_REFERENCE_LEAN_PROMPT", 1)
        # For Pollinations, prefer cloud Vision by default to enrich the lean edit prompt.
        POLLINATIONS_PREFER_CLOUD: bool = True

        bname = type(backend).__name__.lower()
        w = int(width or APP_IMAGE_WIDTH)
        h = int(height or APP_IMAGE_HEIGHT)

        # If backend is not Pollinations, delegate as a generic text generation (keep compatibility).
        if "pollinations" not in bname:
            payload = build_pollinations_prompt(user_text, cfg, refs=[], negative_override=negative_override)
            try:
                path = await backend.generate(payload.merged_prompt, width=w, height=h)  # type: ignore[attr-defined]
                return Path(path) if isinstance(path, (str, Path)) else None
            except Exception as e:
                print(f"[STYLE] non-pollinations backend.generate failed: {e}")
                return None

        # Build base prompts
        base_prompts = build_prompt(user_text, cfg)
        final_negative = (negative_override or "").strip() or base_prompts.negative

        # Resolve local reference path if configured (securely via ReferenceStore)
        ref_path: Optional[Path] = None
        if cfg and getattr(cfg, "use_reference", False) and getattr(cfg, "reference_id", None):
            try:
                store = ReferenceStore(refs_dir)
                ref_path = store.get_path(cfg.reference_id)
            except Exception as e:
                print(f"[STYLE] reference resolution failed: {e}")
                ref_path = None

        # Preferred path: V1 edits with multipart local file
        if FEATURE_POLLINATIONS_V1_EDITS and POLLINATIONS_SECRET and ref_path and ref_path.exists():
            prefer_cloud = bool(POLLINATIONS_PREFER_CLOUD or cfg.reference_cloud)
            # Async descriptors: use cloud first when available, fallback local
            descriptors = await resolve_style_descriptors_for_reference_async(ref_path=ref_path, prefer_cloud=prefer_cloud)

            if STYLE_REFERENCE_LEAN_PROMPT:
                lean_prompt = user_text.strip() if user_text.strip() else base_prompts.positive or "high-quality image"
                prompt_for_edits = _merge_descriptors_into_prompt(user_text, lean_prompt, descriptors, lean=True)
            else:
                prompt_for_edits = _merge_descriptors_into_prompt(user_text, base_prompts.positive, descriptors, lean=False)

            print(f"[STYLE] pollinations: using multipart local reference | model={POLLINATIONS_IMAGE_MODEL} size={w}x{h}")
            if _debug():
                print(f"[STYLE] prompt(mode={'lean' if STYLE_REFERENCE_LEAN_PROMPT else 'full'}): {prompt_for_edits[:200]}")

            try:
                img_url, b64 = await _v1_edits_multipart(
                    prompt=prompt_for_edits,
                    image_file=ref_path,
                    width=w,
                    height=h,
                    negative_prompt=final_negative,
                    seed=seed,
                    model=POLLINATIONS_IMAGE_MODEL,
                )
                if hasattr(backend, "store_generated_result"):
                    try:
                        path = await backend.store_generated_result(image_url=img_url, b64=b64)  # type: ignore[attr-defined]
                        if path:
                            return Path(path)
                    except Exception as e:
                        print(f"[STYLE] backend.store_generated_result failed: {e}")
                return None
            except Exception as e:
                print(f"[STYLE] pollinations multipart failed: {e}")
                return None

        # Optional URL-mode path when enabled via FEATURE_POLLINATIONS_USE_URL_MODE
        if FEATURE_POLLINATIONS_USE_URL_MODE and ALLOW_CLOUD_IMAGE_BACKEND:
            refs = await resolve_reference_urls_for_pollinations(cfg, refs_dir)
            payload = build_pollinations_prompt(user_text, cfg, refs=refs, negative_override=negative_override)
            if refs:
                try:
                    img_url, b64 = await _v1_edits_url_mode(
                        prompt=payload.positive,
                        image_url=refs[0],
                        width=w,
                        height=h,
                        negative_prompt=payload.negative,
                        seed=seed,
                        model=POLLINATIONS_IMAGE_MODEL,
                    )
                    if hasattr(backend, "store_generated_result"):
                        try:
                            path = await backend.store_generated_result(image_url=img_url, b64=b64)  # type: ignore[attr-defined]
                            if path:
                                return Path(path)
                        except Exception as e:
                            print(f"[STYLE] backend.store_generated_result failed: {e}")
                    # Fallback to backend.generate if store failed
                except Exception as e:
                    print(f"[STYLE] pollinations url-mode failed: {e}")
            # Fallback to simple text-only generate on the Pollinations backend
            try:
                path = await backend.generate(payload.merged_prompt, width=w, height=h)  # type: ignore[attr-defined]
                return Path(path) if isinstance(path, (str, Path)) else None
            except Exception as e:
                print(f"[STYLE] backend.generate(text/url) failed: {e}")
                return None

        # Text-only generation via backend (no reference or features disabled)
        payload = build_pollinations_prompt(user_text, cfg, refs=[], negative_override=negative_override)
        try:
            path = await backend.generate(payload.merged_prompt, width=w, height=h)  # type: ignore[attr-defined]
            return Path(path) if isinstance(path, (str, Path)) else None
        except Exception as e:
            print(f"[STYLE] backend.generate(text-only) failed: {e}")
            return None

    except Exception as e:
        print(f"[STYLE] generate_image_pollinations failed: {e}")
        return None

# =========================
# Module init diagnostics
# =========================

def _log_style_env() -> None:
    """Print key style-related environment configuration for diagnostics."""
    print(
        "[STYLE-ENV]",
        f"api={POLLINATIONS_API_BASE or '(unset)'}",
        f"v1_edits={'on' if FEATURE_POLLINATIONS_V1_EDITS else 'off'}",
        f"url_mode={'on' if FEATURE_POLLINATIONS_USE_URL_MODE else 'off'}",
        f"cloud={'on' if ALLOW_CLOUD_IMAGE_BACKEND else 'off'}",
        f"uploads={'on' if POLLINATIONS_ENABLE_UPLOAD else 'off'}",
        f"secret_set={'yes' if bool(POLLINATIONS_SECRET) else 'no'}",
        f"model={POLLINATIONS_IMAGE_MODEL}",
        f"vision_cloud={'on' if FEATURE_STYLE_REF_VISION_CLOUD else 'off'}",
        f"ollama_host={_env_str('APP_OLLAMA_HOST', '(unset)')}",
        f"ollama_vision_model={APP_OLLAMA_VISION_MODEL or '(unset)'}",
    )

with contextlib.suppress(Exception):
    _log_style_env()
