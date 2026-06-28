# slAIdshow : style_engine.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import asyncio
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
        if not name:
            raise ValueError("empty name")
        if "/" in name or "\\" in name:
            raise ValueError("invalid path separator")
        if not re.fullmatch(r"[A-Za-z0-9._\-]+", name):
            raise ValueError("illegal characters")
        return name

    def put(self, filename: str, data: bytes) -> Tuple[str, Path]:
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

        target.write_bytes(bytes(data))
        return (target.name, target)

    def get_path(self, reference_id: str) -> Path:
        base = self._safe_basename(reference_id)
        p = (self.base / base).resolve()
        if p.parent != self.base or not p.exists() or not p.is_file():
            raise FileNotFoundError(f"reference not found: {reference_id}")
        return p

# =========================
# Style prompt builder
# =========================

def _compose_positive(user_topic: str, cfg: StyleConfig) -> str:
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
    try:
        if not cfg or not getattr(cfg, "use_reference", False) or not getattr(cfg, "reference_id", None):
            return
        store = ReferenceStore(refs_dir)
        ref_path = store.get_path(cfg.reference_id)
        if not ref_path.exists():
            print(f"[STYLE] reference file missing on disk: {ref_path}")
            return
        if hasattr(backend, "stage_reference_image"):
            try:
                backend.stage_reference_image(ref_path, float(cfg.reference_strength))  # type: ignore[attr-defined]
                print("[STYLE] staged reference into LocalComfyBackend.")
            except Exception as e:
                print(f"[STYLE] backend staging failed: {e}")
    except Exception as e:
        print(f"[STYLE] prepare_backend_style failed: {e}")

# =========================
# Pollinations integration (V1)
# =========================

POLLINATIONS_API_BASE = _env_str("POLLINATIONS_API_BASE", "https://gen.pollinations.ai")
POLLINATIONS_V1_IMAGES_EDITS_PATH = _env_str("POLLINATIONS_V1_IMAGES_EDITS_PATH", "/v1/images/edits")
POLLINATIONS_SECRET = _env_str("POLLINATIONS_SECRET", "")
POLLINATIONS_IMAGE_MODEL = _env_str("POLLINATIONS_IMAGE_MODEL", _env_str("POLLINATIONS_MODEL", "flux"))

FEATURE_POLLINATIONS_V1_EDITS = _env_bool01("FEATURE_POLLINATIONS_V1_EDITS", 1)
FEATURE_POLLINATIONS_USE_URL_MODE = _env_bool01("FEATURE_POLLINATIONS_USE_URL_MODE", 0)

ALLOW_CLOUD_IMAGE_BACKEND = _env_bool01("ALLOW_CLOUD_IMAGE_BACKEND", 0)
POLLINATIONS_ENABLE_UPLOAD = _env_bool01("POLLINATIONS_ENABLE_UPLOAD", 1)

APP_TIMEOUT_SEC = float(_env_float("APP_OLLAMA_TIMEOUT_SEC", 90.0))
APP_MAX_RETRIES = _env_int("APP_OLLAMA_MAX_RETRIES", 4)
APP_RETRY_BASE_DELAY = float(_env_float("APP_OLLAMA_RETRY_BASE_DELAY", 0.8))

APP_IMAGE_WIDTH = _env_int("APP_IMAGE_WIDTH", 1024)
APP_IMAGE_HEIGHT = _env_int("APP_IMAGE_HEIGHT", 1024)

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
    return httpx.Limits(max_keepalive_connections=6, max_connections=8, keepalive_expiry=20.0)

def _timeout_generic() -> httpx.Timeout:
    return httpx.Timeout(connect=8.0, read=APP_TIMEOUT_SEC, write=30.0, pool=8.0)

def _v1_edits_endpoint() -> str:
    path = POLLINATIONS_V1_IMAGES_EDITS_PATH
    if not path.startswith("/"):
        path = "/" + path
    return f"{POLLINATIONS_API_BASE.rstrip('/')}{path}"

def _auth_headers() -> Dict[str, str]:
    if not POLLINATIONS_SECRET:
        raise RuntimeError("POLLINATIONS_SECRET not set in environment")
    return {"Authorization": f"Bearer {POLLINATIONS_SECRET}"}

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

def _parse_v1_images_response(resp: httpx.Response) -> Tuple[Optional[str], Optional[str]]:
    try:
        js = resp.json()
    except Exception:
        return (None, None)
    try:
        parsed = V1ImagesResp(**js)
        if parsed.data:
            item = parsed.data[0]
            return (str(item.url) if item.url else None, item.b64_json)
    except ValidationError:
        try:
            url = js.get("url") or ((js.get("data") or [{}])[0] or {}).get("url")
            b64 = ((js.get("data") or [{}])[0] or {}).get("b64_json") or js.get("b64_json")
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

    async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_generic(), follow_redirects=True, http2=True) as client:
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
    # WICHTIG: Feld 'image' (nicht 'image_url'), um den Test-Aufruf exakt zu spiegeln.
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

    async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_generic(), follow_redirects=True, http2=True) as client:
        print(f"[STYLE] pollinations: POST {_v1_edits_endpoint()} (url-mode), model={model}")
        resp = await _post_with_retries(client, _v1_edits_endpoint(), files=files, headers=_auth_headers())
        if resp.status_code >= 400:
            raise RuntimeError(f"pollinations url-mode error {resp.status_code}: {resp.text[:512]}")
        return _parse_v1_images_response(resp)

# Legacy helper
POLLINATIONS_UPLOAD_ENDPOINT = _env_str("POLLINATIONS_UPLOAD_ENDPOINT", "/upload")

def validate_image_bytes_minimal(data: bytes) -> None:
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
    headers = {"Authorization": f"Bearer {POLLINATIONS_SECRET}"}
    files = {"file": (neutral_name, data, mime)}

    async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_generic(), follow_redirects=True, http2=True) as client:
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
    Pollinations V1 Edits orchestration with 'lean edit prompt' when a reference is used.
    Aligns parameters with the working test script to ensure the reference actually guides the output.
    """
    try:
        STYLE_REFERENCE_LEAN_PROMPT = _env_bool01("STYLE_REFERENCE_LEAN_PROMPT", 1)
        bname = type(backend).__name__.lower()
        w = int(width or APP_IMAGE_WIDTH)
        h = int(height or APP_IMAGE_HEIGHT)

        # If not pollinations backend, fall back to generic generation
        if "pollinations" not in bname:
            payload = build_pollinations_prompt(user_text, cfg, refs=[], negative_override=negative_override)
            try:
                path = await backend.generate(payload.merged_prompt, width=w, height=h)  # type: ignore[attr-defined]
                return Path(path) if isinstance(path, (str, Path)) else None
            except Exception as e:
                print(f"[STYLE] non-pollinations backend.generate failed: {e}")
                return None

        # Build prompts
        base_prompts = build_prompt(user_text, cfg)
        final_negative = (negative_override or "").strip() or base_prompts.negative

        # Reference resolution
        ref_path: Optional[Path] = None
        if cfg and getattr(cfg, "use_reference", False) and getattr(cfg, "reference_id", None):
            try:
                store = ReferenceStore(refs_dir)
                ref_path = store.get_path(cfg.reference_id)
            except Exception as e:
                print(f"[STYLE] reference resolution failed: {e}")
                ref_path = None

        # If we have a local reference and V1 edits are enabled, go multipart
        if FEATURE_POLLINATIONS_V1_EDITS and POLLINATIONS_SECRET and ref_path and ref_path.exists():
            # Lean prompt (mimic the successful test): do not prepend style preset verbiage, just the user's intent
            if STYLE_REFERENCE_LEAN_PROMPT:
                # Knapper Prompt wie im Test – dein Test hatte explizit "Donald Duck as a pencil sketch..."
                lean_prompt = user_text.strip() if user_text.strip() else base_prompts.positive
                if not lean_prompt:
                    lean_prompt = "high-quality image"
                prompt_for_edits = lean_prompt
            else:
                # Fallback: benutze die positive Komposition
                prompt_for_edits = base_prompts.positive

            print(f"[STYLE] pollinations: using multipart local reference | model={POLLINATIONS_IMAGE_MODEL} size={w}x{h}")
            print(f"[STYLE] prompt(mode={'lean' if STYLE_REFERENCE_LEAN_PROMPT else 'full'}): {prompt_for_edits[:180]}")

            # Perform multipart call using the exact contract that worked in your test
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
                # Transparenter Fehler: 402, 4xx, 5xx inkl. body snippet
                print(f"[STYLE] pollinations multipart failed: {e}")
                return None

        # If no local reference (or disabled), fall back to text-only via backend
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
    print(
        "[STYLE-ENV]",
        f"api={POLLINATIONS_API_BASE or '(unset)'}",
        f"v1_edits={'on' if FEATURE_POLLINATIONS_V1_EDITS else 'off'}",
        f"url_mode={'on' if FEATURE_POLLINATIONS_USE_URL_MODE else 'off'}",
        f"cloud={'on' if ALLOW_CLOUD_IMAGE_BACKEND else 'off'}",
        f"uploads={'on' if POLLINATIONS_ENABLE_UPLOAD else 'off'}",
        f"secret_set={'yes' if bool(POLLINATIONS_SECRET) else 'no'}",
        f"model={POLLINATIONS_IMAGE_MODEL}",
    )

with contextlib.suppress(Exception):
    _log_style_env()
