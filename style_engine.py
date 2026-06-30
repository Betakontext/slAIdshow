#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
style-engine.py

Purpose:
- Provide a single, production-grade style enrichment module that:
  - Accepts a free-form "style text prompt" from the UI
  - Accepts an optional reference image either from local file upload or by downloading from a URL
  - Optionally analyzes the reference locally (OpenCV/skimage) to extract style descriptors
  - Optionally uses an Ollama Vision model to summarize the reference image into concise style text
  - Returns additive "style_positive" text and structured details that the existing image_backend.py
    can consume for prompt synthesis WITHOUT changing the existing negative prompt logic.

Key design constraints:
- Do NOT change existing negative prompt logic in image_backend.py. This engine only enriches positive style inputs.
- Keep cloud access opt-in and policy-controlled (localhost by default; remote only if allowed by env).
- Be fully async (I/O) and offload CPU-bound analysis to threads to avoid blocking the event loop.
- Robust error handling and retries (for network operations).
- Safe local file handling and URL downloads with type/size guards.

Integration points:
- FastAPI controller can expose /api/style/build that calls build_styles() with the user request data.
- image_backend.py remains the "single source of truth" for final negative prompt composition.
- The style_positive returned here should be merged additively into positive prompt synthesis in image_backend.py,
  mirroring how the existing negative prompt blueprint is handled.

Testing checklist (manual):
1) Enable only style_text_prompt and verify it is appended to style_positive (no image).
2) Upload a local reference image, enable "use_local_style_features", verify descriptors appended.
3) Provide a reference image URL, enable "use_ollama_vision", verify concise vision text appended.
4) Toggle "deactivate_all_styles" and confirm style_positive and all style-derived fields are empty.
5) Confirm that Pollinations/Comfy generation still work with the additive style_positive combined by image_backend.py.

Notes:
- For Ollama Vision, ensure a compatible multimodal model is available (e.g., `llava:latest`, `moondream:latest`, `bakllava:latest`, etc.)
  and that ollama serve is running. The API used is /api/generate with images: [base64-encoded content].

"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import mimetypes
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from pydantic import BaseModel, Field, ConfigDict, field_validator, HttpUrl, ValidationError

# Optional style features dependencies
# We import lazily when used; helpers will raise friendly errors if unavailable
# Required: opencv-python, scikit-image; sklearn is optional.


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
# Core configuration
# =========================

APP_IMAGE_WIDTH = _env_int("APP_IMAGE_WIDTH", 1280)
APP_IMAGE_HEIGHT = _env_int("APP_IMAGE_HEIGHT", 720)

# Reference directory where images are stored for subsequent use
# We use the main outputs/images folder so UI can serve them via /static.
# Optionally a subfolder for refs can be used; here we go with outputs/images/refs.
DEFAULT_REFS_DIR = Path(_env_str("APP_REFS_DIR", "./outputs/images/refs")).resolve()

# Network and retry defaults (reuse existing app defaults if present)
HTTP_TIMEOUT_SEC = float(_env_float("APP_OLLAMA_TIMEOUT_SEC", 90.0))
HTTP_MAX_RETRIES = _env_int("APP_OLLAMA_MAX_RETRIES", 4)
HTTP_RETRY_BASE_DELAY = float(_env_float("APP_OLLAMA_RETRY_BASE_DELAY", 0.8))

# Pollinations cloud policy (this module doesn't call their generation endpoints,
# but may download a URL or later be extended; leave policy hints here)
ALLOW_CLOUD_IMAGE_BACKEND = _env_bool01("ALLOW_CLOUD_IMAGE_BACKEND", 0)

# Ollama Vision configuration
OLLAMA_HOST = _env_str("APP_OLLAMA_HOST", "127.0.0.1")
OLLAMA_PORT = _env_int("APP_OLLAMA_PORT", 11434)
OLLAMA_VISION_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
OLLAMA_VISION_MODEL = _env_str("APP_OLLAMA_VISION_MODEL", "llava:latest")
OLLAMA_VISION_TEMPERATURE = float(_env_float("APP_OLLAMA_TEMPERATURE", 0.2))
OLLAMA_VISION_NUM_CTX = _env_int("APP_OLLAMA_NUM_CTX", 1024)
OLLAMA_VISION_NUM_PREDICT = _env_int("APP_OLLAMA_NUM_PREDICT", 256)
OLLAMA_VISION_TOP_K = _env_int("APP_OLLAMA_TOP_K", 40)
OLLAMA_VISION_TOP_P = float(_env_float("APP_OLLAMA_TOP_P", 0.9))
OLLAMA_VISION_REPEAT_PENALTY = float(_env_float("APP_OLLAMA_REPEAT_PENALTY", 1.1))

# Vision remote/cloud policy:
# - By default, enforce localhost only. Optional remote allowed flag:
APP_ALLOW_REMOTE_VISION = _env_bool01("APP_ALLOW_REMOTE_VISION", 0)
APP_ALLOWED_SUBNETS_VISION = set((os.getenv("APP_ALLOWED_SUBNETS_VISION", "") or "").split()) - {""}

# File download limits
MAX_DOWNLOAD_BYTES = _env_int("APP_STYLE_MAX_DOWNLOAD_BYTES", 15_000_000)  # 15 MB
ALLOWED_IMAGE_MIME = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/gif",
}

# Feature toggles
FEATURE_LOCAL_STYLE_ANALYSIS_DEFAULT = _env_bool01("FEATURE_LOCAL_STYLE_ANALYSIS_DEFAULT", 1)
FEATURE_OLLAMA_VISION_DEFAULT = _env_bool01("FEATURE_OLLAMA_VISION_DEFAULT", 0)


# =========================
# Pydantic data models
# =========================

class StyleEngineRequest(BaseModel):
    """
    A single enriched-style build request. This module DOES NOT generate the image.
    It only returns additive style prompt components and structured metadata that
    image_backend.py can use in its existing synthesis pipeline.
    """
    model_config = ConfigDict(extra="forbid")

    # Core prompts (content-positive is the main user/LLM prompt outside styles)
    content_positive: str = Field(default="")
    # New free-form style text prompt from UI (additive)
    style_text_prompt: str = Field(default="")

    # Reference source selection
    reference_source: str = Field(default="none")  # "none" | "local_file" | "url"
    # If "local_file": UI will POST a file; controller should save it by calling
    #   style-engine.save_reference_file(bytes, filename) to get a reference_id,
    #   then set reference_id here.
    # If "url": controller sets reference_url here and we download it.
    reference_id: Optional[str] = Field(default=None)
    reference_url: Optional[str] = Field(default=None)
    reference_strength: float = Field(default=0.6, ge=0.0, le=1.0)

    # Optional analysis toggles
    use_local_style_features: bool = Field(default=FEATURE_LOCAL_STYLE_ANALYSIS_DEFAULT)
    use_ollama_vision: bool = Field(default=FEATURE_OLLAMA_VISION_DEFAULT)

    # UI kill switch for all styles
    deactivate_all_styles: bool = Field(default=False)

    # Optional hints for later generation (not used here, passed-through to caller)
    width: Optional[int] = Field(default=None)
    height: Optional[int] = Field(default=None)
    seed: Optional[int] = Field(default=None)

    # Which backend is targeted downstream (for reference only; no generation here)
    target_backend_name: str = Field(default="comfyui")  # "comfyui" | "pollinations" | others

    @field_validator("reference_source")
    @classmethod
    def _validate_source(cls, v: str) -> str:
        vv = (v or "").strip().lower()
        if vv not in {"none", "local_file", "url"}:
            raise ValueError("reference_source must be one of: none | local_file | url")
        return vv


class StyleEngineResponse(BaseModel):
    """
    Result of style enrichment for downstream prompt synthesis.
    - style_positive: additive style text to be merged into the positive prompt.
    - negative_hint: optional text with negative style hints; NOT auto-injected.
    - artifacts and metadata are included for UI and debugging.
    """
    model_config = ConfigDict(extra="ignore")

    # Additive style content to be appended by image_backend.py
    style_positive: str = Field(default="")

    # The following fields are metadata for transparency and UI display:
    descriptors_used: List[str] = Field(default_factory=list)
    vision_text_used: str = Field(default="")
    reference_file_saved: Optional[str] = Field(default=None)  # local path
    reference_id: Optional[str] = Field(default=None)
    warnings: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)

    # Passthrough (caller can use these to assemble final requests)
    content_positive: str = Field(default="")
    style_text_prompt_raw: str = Field(default="")
    negative_hint: str = Field(default="")  # blueprint only; DO NOT auto-inject


# =========================
# Reference store (safe local file handling)
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
        # Allow only safe characters to avoid traversal or script injection
        if not re.fullmatch(r"[A-Za-z0-9._\-]+", name):
            raise ValueError("illegal characters")
        return name

    def put(self, filename: str, data: bytes) -> Tuple[str, Path]:
        """Store bytes under a sanitized filename; de-duplicate by suffix _N if needed."""
        if not isinstance(data, (bytes, bytearray)):
            raise ValueError("data must be bytes")
        try:
            base_raw = filename or "ref.png"
            base = self._safe_basename(Path(base_raw).name)
        except Exception:
            base = "ref.png"

        # Add default extension if missing
        if "." not in base:
            base = base + ".png"

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
# HTTP helpers (timeouts, retries)
# =========================

def _httpx_limits() -> httpx.Limits:
    return httpx.Limits(max_keepalive_connections=6, max_connections=8, keepalive_expiry=20.0)

def _timeout_generic() -> httpx.Timeout:
    return httpx.Timeout(connect=8.0, read=HTTP_TIMEOUT_SEC, write=30.0, pool=8.0)

async def _get_with_retries(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    max_retries: int = HTTP_MAX_RETRIES,
    base_delay: float = HTTP_RETRY_BASE_DELAY,
) -> httpx.Response:
    last_exc: Optional[Exception] = None
    delay = float(base_delay)
    for attempt in range(1, int(max_retries) + 1):
        try:
            return await client.get(url, headers=headers)
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            last_exc = e
            if attempt >= max_retries:
                break
            await asyncio.sleep(delay)
            delay *= 1.8
    raise RuntimeError(f"GET failed after {max_retries} attempts: {last_exc}")

async def _post_json_with_retries(
    client: httpx.AsyncClient,
    url: str,
    *,
    json: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    max_retries: int = HTTP_MAX_RETRIES,
    base_delay: float = HTTP_RETRY_BASE_DELAY,
) -> httpx.Response:
    last_exc: Optional[Exception] = None
    delay = float(base_delay)
    for attempt in range(1, int(max_retries) + 1):
        try:
            return await client.post(url, json=json, headers=headers)
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            last_exc = e
            if attempt >= max_retries:
                break
            await asyncio.sleep(delay)
            delay *= 1.8
    raise RuntimeError(f"POST failed after {max_retries} attempts: {last_exc}")


# =========================
# Minimal image validation
# =========================

MAGIC_HEADERS = (
    b"\x89PNG",        # PNG
    b"\xFF\xD8\xFF",   # JPEG
    b"RIFF",           # WEBP (requires 'WEBP' within first bytes)
    b"BM",             # BMP
    b"GIF87a",         # GIF
    b"GIF89a",         # GIF
)

def validate_image_bytes_minimal(data: bytes) -> None:
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError("invalid bytes")
    if len(data) < 128:
        raise ValueError("image too small")
    head = data[:16]
    if head.startswith(b"RIFF"):
        if b"WEBP" not in data[:64]:
            raise ValueError("invalid WEBP magic")
        return
    if not any(head.startswith(m) for m in MAGIC_HEADERS):
        # Allow unknown headers but warn (still accept to be format-agnostic)
        print("[STYLE] unknown image magic header; continuing")

def sniff_mime_from_bytes(data: bytes, fallback: str = "application/octet-stream") -> str:
    # Heuristic by magic
    head = data[:16]
    if head.startswith(b"\x89PNG"):
        return "image/png"
    if head.startswith(b"\xFF\xD8\xFF"):
        return "image/jpeg"
    if head.startswith(b"RIFF") and b"WEBP" in data[:64]:
        return "image/webp"
    if head[:2] == b"BM":
        return "image/bmp"
    if head.startswith(b"GIF87a") or head.startswith(b"GIF89a"):
        return "image/gif"
    return fallback

def ensure_image_extension(name: str, mime: str) -> str:
    ext = Path(name).suffix.lower()
    if ext:
        return name
    if mime == "image/jpeg":
        return name + ".jpg"
    if mime == "image/png":
        return name + ".png"
    if mime == "image/webp":
        return name + ".webp"
    if mime == "image/bmp":
        return name + ".bmp"
    if mime == "image/gif":
        return name + ".gif"
    return name + ".png"


# =========================
# Public reference helpers (to be called by controller)
# =========================

async def save_reference_file(data: bytes, filename: str, refs_dir: Union[str, Path] = DEFAULT_REFS_DIR) -> Tuple[str, Path]:
    """
    Save a reference image uploaded by the UI. Returns (reference_id, path).
    Accepts any common image format; performs minimal validation and safe naming.
    """
    validate_image_bytes_minimal(data)
    mime = sniff_mime_from_bytes(data, fallback=mimetypes.guess_type(filename or "")[0] or "application/octet-stream")
    safe_name = Path(filename or "reference").name
    safe_name = re.sub(r"[^A-Za-z0-9._\-]", "_", safe_name)
    safe_name = ensure_image_extension(safe_name, mime)
    store = ReferenceStore(refs_dir)
    return store.put(safe_name, data)

async def save_reference_from_url(url: str, refs_dir: Union[str, Path] = DEFAULT_REFS_DIR) -> Tuple[str, Path]:
    """
    Download an image by URL and store locally. Returns (reference_id, path).
    Applies content-type and size restrictions to avoid abuse.
    """
    url = (url or "").strip()
    if not url or not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("invalid url")

    async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_generic(), follow_redirects=True, http2=True) as client:
        # HEAD might be blocked; use GET and abort if content-length too large
        resp = await _get_with_retries(client, url)
        if resp.status_code >= 400:
            raise RuntimeError(f"download error {resp.status_code}")

        cl = int(resp.headers.get("Content-Length", "0") or "0")
        if cl and cl > MAX_DOWNLOAD_BYTES:
            raise ValueError(f"file too large: {cl} bytes")

        ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype and ctype not in ALLOWED_IMAGE_MIME:
            # Some servers mislabel; still allow if magic is valid later
            pass

        data = resp.content
        if len(data) > MAX_DOWNLOAD_BYTES:
            raise ValueError(f"file too large after download: {len(data)} bytes")

    validate_image_bytes_minimal(data)
    mime = sniff_mime_from_bytes(data, fallback=ctype or "application/octet-stream")
    filename = Path(httpx.URL(url).path or "reference").name or "reference"
    filename = re.sub(r"[^A-Za-z0-9._\-]", "_", filename)
    filename = ensure_image_extension(filename, mime)
    store = ReferenceStore(refs_dir)
    return store.put(filename, data)


# =========================
# Local style features (OpenCV + skimage)
# =========================

# We inline essential parts of the previously separate style_features.py to keep a single-file module.
# Import lazily to avoid import cost if not used.

@dataclass
class _StyleAnalysis:
    edge_density: float
    edge_coherence: float
    edge_thickness_score: float
    saturation_mean: float
    saturation_std: float
    color_clusters: int
    color_silhouette: float
    contrast: float
    grayscale_ratio: float
    grain_score: float
    hf_ratio: float
    dot_pattern_score: float
    straight_line_score: float
    brush_texture_score: float
    bokeh_score: float
    class_scores: Dict[str, float]
    primary_class: str

def _lazy_import_cv_stack():
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    from skimage.feature import canny  # type: ignore
    from skimage.color import rgb2gray  # type: ignore
    try:
        from sklearn.metrics import silhouette_score  # type: ignore
        _have_sklearn = True
    except Exception:
        silhouette_score = None  # type: ignore
        _have_sklearn = False
    return cv2, np, canny, rgb2gray, silhouette_score, _have_sklearn

def _base_descriptors_for_class(a: _StyleAnalysis) -> List[str]:
    c = a.primary_class
    d: List[str] = []

    if c == "comic":
        d.append("clear line art" if a.edge_coherence < 0.34 or a.edge_thickness_score < 0.03 else "bold outlines")
        d.append("flat colors" if a.saturation_mean < 0.18 and a.saturation_std < 0.06 and a.color_clusters <= 5 else "balanced palette")
        if a.contrast > 0.16:
            d.append("high contrast")
        if a.dot_pattern_score > 0.9:
            d.append("screen-tone dots")
        if a.hf_ratio < 1.1 and a.grain_score < 70.0:
            d.append("clean finish")

    elif c == "manga":
        d += ["monochrome", "clear line art"]
        if a.dot_pattern_score > 0.8:
            d.append("halftone shading")
        if a.contrast > 0.15:
            d.append("high contrast")
        if a.hf_ratio < 1.15:
            d.append("clean finish")

    elif c == "photo":
        d.append("natural lighting")
        if a.hf_ratio > 1.20:
            d.append("fine detail")
        d.append("smooth gradients")
        if a.saturation_std > 0.1:
            d.append("rich colors")
        if a.bokeh_score > 0.30:
            d.append("shallow depth of field")

    elif c == "science illustration/infographic":
        d += ["thin precise lines", "flat colors", "clean layout"]
        if a.straight_line_score > 0.35:
            d.append("geometric accuracy")
        if a.saturation_std < 0.08:
            d.append("limited palette")

    elif c == "classical oil painting":
        d += ["brush textures", "rich tones", "soft transitions"]
        if a.brush_texture_score > 0.45:
            d.append("impasto strokes")

    elif c == "watercolor":
        d += ["soft washes", "bleeding edges", "delicate tones"]
        if a.saturation_mean < 0.2:
            d.append("pastel palette")

    elif c == "children sketches":
        d += ["simple shapes", "thin uneven lines", "playful composition"]

    elif c == "technical drawing/technical sketch":
        d += ["precise straight lines", "monochrome", "high clarity"]

    else:
        d += ["loose strokes", "expressive lines", "dynamic texture"]

    out: List[str] = []
    for x in d:
        if x not in out:
            out.append(x)
    return out[:5]

def _analyze_style(image_path: Path, debug: bool = False) -> _StyleAnalysis:
    # Inline implementation adapted for single-file usage (condensed; behavior consistent)
    cv2, np, canny, rgb2gray, silhouette_score, _HAVE_SKLEARN = _lazy_import_cv_stack()

    def _load_bgr(path: Path) -> "np.ndarray":
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"cannot read image: {path}")
        return img

    def _downscale(img: "np.ndarray", max_side: int = 640) -> "np.ndarray":
        h, w = img.shape[:2]
        ms = max(h, w)
        if ms <= max_side:
            return img
        s = max_side / ms
        return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    def _edge_metrics(bgr: "np.ndarray") -> Tuple[float, float, float]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = _downscale(gray, 640)
        edges = canny(gray.astype("float32") / 255.0, sigma=1.2)
        edge_density = float(edges.mean())

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy) + 1e-6
        ori = np.arctan2(gy, gx)

        mask = (mag > (mag.mean() + mag.std()))
        if int(mask.sum()) > 100:
            ori_edges = ori[mask]
            var_ori = float(np.var(np.sin(ori_edges)) + np.var(np.cos(ori_edges)))
            edge_coherence = max(0.0, 1.0 - min(1.0, var_ori / 1.0))
        else:
            edge_coherence = 0.0

        edges_u = (edges.astype("uint8") * 255)
        dil = cv2.dilate(edges_u, np.ones((3, 3), "uint8"), 1)
        thickness_score = float((dil > 0).mean() - edge_density)
        return edge_density, edge_coherence, thickness_score

    def _estimate_silhouette_fallback(X: "np.ndarray", labels: "np.ndarray") -> float:
        try:
            labs = labels.reshape(-1)
            ks = np.unique(labs)
            if ks.size < 2:
                return -0.05
            cents = []
            intras = []
            for k in ks:
                pts = X[labs == k]
                if pts.size == 0:
                    continue
                c = pts.mean(axis=0)
                cents.append(c)
                d = np.linalg.norm(pts - c, axis=1).mean()
                intras.append(d if np.isfinite(d) else 0.0)
            if len(cents) < 2:
                return -0.03
            cents = np.vstack(cents)
            dsum = 0.0
            cnt = 0
            for i in range(len(cents)):
                for j in range(i + 1, len(cents)):
                    dsum += float(np.linalg.norm(cents[i] - cents[j]))
                    cnt += 1
            inter = (dsum / cnt) if cnt else 0.0
            intra = float(np.mean(intras)) if intras else 0.0
            if intra <= 1e-6:
                return 0.2
            ratio = inter / intra
            val = (ratio - 1.0) * 0.15
            return float(np.clip(val, -0.1, 0.5))
        except Exception:
            return -0.05

    def _color_metrics(bgr: "np.ndarray") -> Tuple[float, float, int, float]:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1].astype("float32") / 255.0
        s_mean = float(s.mean())
        s_std = float(s.std())

        sample = _downscale(bgr, 480)
        flat = sample.reshape(-1, 3).astype("float32")
        if flat.shape[0] > 40000:
            idx = np.random.choice(flat.shape[0], 40000, replace=False)
            flat = flat[idx]

        best_k = 3
        best_sil = -1.0
        prev_compact = None
        for k in range(3, 9):
            criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 20, 1.0)
            compactness, labels, _ = cv2.kmeans(flat, k, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
            if _HAVE_SKLEARN and silhouette_score is not None:
                try:
                    if flat.shape[0] > 2000:
                        idx = np.random.choice(flat.shape[0], 2000, replace=False)
                        sil = silhouette_score(flat[idx], labels[idx].ravel(), metric='euclidean')  # type: ignore
                    else:
                        sil = silhouette_score(flat, labels.ravel(), metric='euclidean')  # type: ignore
                except Exception:
                    sil = -1.0
            else:
                sil = _estimate_silhouette_fallback(flat, labels)

            if sil > best_sil:
                best_sil = sil
                best_k = k

            if prev_compact is None:
                prev_compact = compactness
            else:
                if compactness > prev_compact * 0.98:
                    break
                prev_compact = compactness

        return s_mean, s_std, int(best_k), float(best_sil)

    def _contrast_metric(bgr: "np.ndarray") -> float:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
        return float(gray.std())

    def _grayscale_ratio(bgr: "np.ndarray") -> float:
        b, g, r = cv2.split(bgr.astype("float32"))
        diff = (np.abs(r - g) + np.abs(g - b) + np.abs(r - b)) / (3.0 * 255.0)
        return float((diff < 0.03).mean())

    def _grain_and_hf(bgr: "np.ndarray") -> Tuple[float, float]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
        g_small = _downscale((gray * 255).astype("uint8"), 512).astype("float32") / 255.0
        lap = cv2.Laplacian(g_small, cv2.CV_32F)
        grain = float(lap.var())

        G = _downscale(gray, 512)
        F = np.fft.fftshift(np.fft.fft2(G))
        mag = np.log1p(np.abs(F))
        H, W = mag.shape
        yy, xx = np.ogrid[:H, :W]
        cy, cx = H // 2, W // 2
        r = np.hypot(yy - cy, xx - cx)
        hf_mask = (r > min(H, W) * 0.22) & (r < min(H, W) * 0.48)
        lf_mask = (r < min(H, W) * 0.12)
        hf_ratio = float(mag[hf_mask].mean() / (mag[lf_mask].mean() + 1e-6))
        return grain, hf_ratio

    def _dot_pattern_score(bgr: "np.ndarray") -> float:
        gray = rgb2gray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        gray = _downscale((gray * 255).astype("uint8"), 512).astype("float32") / 255.0
        F = np.fft.fftshift(np.fft.fft2(gray))
        mag = np.log1p(np.abs(F))
        H, W = mag.shape
        yy, xx = np.ogrid[:H, :W]
        cy, cx = H // 2, W // 2
        r = np.hypot(yy - cy, xx - cx)
        ring = (r > 26) & (r < 46)
        neigh = ((r > 18) & (r < 24)) | ((r > 48) & (r < 56))
        ring_mean = float(mag[ring].mean())
        neigh_mean = float(mag[neigh].mean() + 1e-6)
        z = (ring_mean - neigh_mean) / (neigh_mean + 1e-6)
        return float(max(0.0, z * 3.0))

    def _straight_line_score(bgr: "np.ndarray") -> float:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = _downscale(gray, 800)
        edges = cv2.Canny(gray, 80, 160, apertureSize=3, L2gradient=True)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=40, maxLineGap=3)
        if lines is None or len(lines) == 0:
            return 0.0
        h, w = gray.shape
        per = float(h + w)
        total_len = 0.0
        for l in lines:
            x1, y1, x2, y2 = l[0]
            total_len += float(np.hypot(x2 - x1, y2 - y1))
        return float(min(1.0, total_len / (per * 15.0)))

    def _brush_texture_score(bgr: "np.ndarray") -> float:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
        L = lab[:, :, 0].astype("float32") / 255.0
        Ls = _downscale(L, 512)
        g1 = cv2.GaussianBlur(Ls, (0, 0), 1.0)
        g2 = cv2.GaussianBlur(Ls, (0, 0), 3.0)
        dog = cv2.absdiff(g1, g2)
        mean = cv2.blur(dog, (9, 9))
        sq = cv2.blur(dog * dog, (9, 9))
        var = sq - mean * mean
        return float(np.clip(var.mean() * 50.0, 0.0, 1.0))

    def _bokeh_score(bgr: "np.ndarray") -> float:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
        gs = _downscale(gray, 512)
        lap = cv2.Laplacian(gs, cv2.CV_32F)
        sharp = cv2.GaussianBlur(lap * lap, (0, 0), 1.0)
        h, w = sharp.shape
        tiles = 6
        th, tw = h // tiles, w // tiles
        vals = []
        for i in range(tiles):
            for j in range(tiles):
                patch = sharp[i*th:(i+1)*th, j*tw:(j+1)*tw]
                if patch.size > 0:
                    vals.append(float(patch.mean()))
        vals = np.array(vals, dtype="float32")
        if vals.size < 4:
            return 0.0
        v = float(np.std(vals))
        return float(np.clip(v * 10.0, 0.0, 1.5))

    bgr = _load_bgr(image_path)

    ed, ecoh, th = _edge_metrics(bgr)
    s_mean, s_std, k, sil = _color_metrics(bgr)
    contr = _contrast_metric(bgr)
    gray_ratio = _grayscale_ratio(bgr)
    grain, hf_ratio = _grain_and_hf(bgr)
    dots = _dot_pattern_score(bgr)
    straight_score = _straight_line_score(bgr)
    brush_score = _brush_texture_score(bgr)
    bokeh = _bokeh_score(bgr)

    # Score classes (condensed; behavior closely mirrors provided version)
    photo = 0.0
    if hf_ratio > 1.15:
        photo += 0.55
    if bokeh > 0.26:
        photo += 0.30
    if s_mean > 0.20 and s_std > 0.09:
        photo += 0.18
    if k >= 7:
        photo += 0.12
    if sil < 0.05:
        photo += 0.08
    if (hf_ratio > 1.15) and (bokeh > 0.26):
        photo += 0.12
    if contr > 0.17:
        photo += 0.07
    if grain > 80.0:
        photo += 0.05
    photo_like_gate = (
        (1.07 <= hf_ratio <= 1.15) and
        (bokeh >= 0.18 or contr >= 0.15) and
        ((k >= 6) or (sil <= 0.08)) and
        (straight_score < 0.18) and
        not (ecoh > 0.28 and 0.012 < th < 0.060) and
        (dots < 0.8)
    )
    if photo_like_gate:
        photo += 0.20
    photo_like_gate_relaxed = (
        (hf_ratio < 1.00) and
        (bokeh < 0.12) and
        (k <= 4) and (sil >= 0.45) and
        (s_mean >= 0.32) and (s_std >= 0.18) and
        (straight_score < 0.12) and
        (ecoh <= 0.06) and
        (th >= 0.10) and
        (dots < 0.6)
    )
    if photo_like_gate_relaxed:
        photo += 0.32
    if ecoh > 0.32 and 0.012 < th < 0.050 and bokeh < 0.22 and hf_ratio < 1.14:
        photo -= 0.35
    if (ecoh <= 0.06 and th >= 0.10 and hf_ratio < 0.95 and k <= 4 and sil >= 0.45):
        photo -= 0.05
    if dots > 1.1 and gray_ratio > 0.45:
        photo -= 0.20
    photo = max(0.0, photo)

    comic = 0.0
    if ecoh >= 0.20 and 0.012 < th < 0.060 and hf_ratio < 1.12:
        comic += 0.50
    if bokeh < 0.22:
        comic += 0.12
    if k <= 7 and sil > 0.06:
        comic += 0.10
    if ed > 0.040:
        comic += 0.06
    if dots > 0.9:
        comic += 0.08
    if (ecoh <= 0.05 and k <= 4 and sil >= 0.45 and th >= 0.10 and hf_ratio < 0.95):
        comic -= 0.16
    if (hf_ratio > 1.15 and bokeh > 0.26) or (s_mean > 0.22 and s_std > 0.10 and k >= 7):
        comic -= 0.08
    if s_mean > 0.28 and s_std > 0.12 and k >= 7:
        comic -= 0.10
    if (0.16 <= s_mean <= 0.42) and (0.12 <= s_std <= 0.35) and (4 <= k <= 7) \
       and (th >= 0.10) and (ecoh <= 0.10) and (bokeh < 0.24) and (hf_ratio < 1.10) \
       and (straight_score < 0.10) and (dots < 1.1):
        comic += 0.24
    comic = max(0.0, comic)

    manga = 0.0
    if gray_ratio > 0.60 and ed > 0.032 and ecoh > 0.24 and hf_ratio < 1.08:
        manga += 0.60
    if dots > 1.0:
        manga += 0.15
    if s_mean > 0.15:
        manga -= 0.10
    manga = max(0.0, manga)

    child_sketch = 0.0
    if s_mean < 0.16:
        child_sketch += 0.20
    else:
        child_sketch -= 0.15
    if k <= 4:
        child_sketch += 0.15
    else:
        child_sketch -= 0.10
    if hf_ratio < 1.06:
        child_sketch += 0.15
    else:
        child_sketch -= 0.15
    if th < 0.016:
        child_sketch += 0.15
    else:
        child_sketch -= 0.10
    if ed > 0.030 and ecoh < 0.18:
        child_sketch += 0.20
    else:
        child_sketch -= 0.10
    if bokeh > 0.22 or (s_mean > 0.18 and s_std > 0.08):
        child_sketch -= 0.25
    child_sketch = max(0.0, child_sketch)

    oil = 0.0
    if brush_score > 0.46:
        oil += 0.30
    if hf_ratio < 1.26 and s_std > 0.06:
        oil += 0.18
    if contr > 0.12:
        oil += 0.10
    if sil < 0.07 and k >= 5:
        oil += 0.12
    if hf_ratio > 1.18:
        oil -= 0.14
    if bokeh > 0.30:
        oil -= 0.12
    if s_mean > 0.20 and s_std > 0.09:
        oil -= 0.08
    if k >= 7:
        oil -= 0.08
    oil = max(0.0, oil)

    sci_infog = 0.0
    if ecoh > 0.36:
        sci_infog += 0.30
    if sil > 0.12 and k <= 6:
        sci_infog += 0.20
    if straight_score > 0.28:
        sci_infog += 0.35
    if s_std < 0.09:
        sci_infog += 0.15
    poster_like = (
        (ecoh <= 0.06) and
        (th >= 0.10) and
        (hf_ratio < 0.90) and
        (k <= 4) and
        (sil >= 0.45) and
        (s_mean >= 0.32) and
        (s_std >= 0.18) and
        (straight_score < 0.12) and
        (dots < 0.6)
    )
    if poster_like:
        sci_infog += 0.28
        comic = max(0.0, comic - 0.12)
    sci_infog = max(0.0, sci_infog)

    watercolor = 0.0
    if s_mean < 0.22 and s_std < 0.07:
        watercolor += 0.30
    if th < 0.018 and ed < 0.04:
        watercolor += 0.25
    if hf_ratio < 1.12:
        watercolor += 0.20
    if sil < 0.08 and k <= 6:
        watercolor += 0.25
    watercolor = max(0.0, watercolor)

    technical = 0.0
    if straight_score > 0.32:
        technical += 0.45
    if ecoh > 0.32:
        technical += 0.25
    if (gray_ratio > 0.45) or (s_std < 0.06):
        technical += 0.20
    if ed > 0.028 and th < 0.02:
        technical += 0.10
    technical = max(0.0, technical)

    scribble = 0.0
    if s_mean < 0.14 and s_std < 0.055 and k <= 4:
        scribble += 0.22
    if ed > 0.052 and ecoh < 0.18 and th < 0.018:
        scribble += 0.20
    if hf_ratio < 1.06:
        scribble += 0.15
    if bokeh < 0.16:
        scribble += 0.08

    photo_like = int(hf_ratio > 1.15) + int(bokeh > 0.26) + int(s_mean > 0.20 and s_std > 0.09) + int(k >= 7)
    comic_like = int(ecoh >= 0.20) + int(0.012 < th < 0.060) + int(hf_ratio < 1.12) + int(bokeh < 0.22)
    if photo_like >= 2:
        scribble -= 0.30
    if comic_like >= 3 and s_mean >= 0.16:
        scribble -= 0.25
    scribble = max(0.0, scribble)

    scores = {
        "comic": float(min(1.0, comic)),
        "manga": float(min(1.0, manga)),
        "photo": float(min(1.0, photo)),
        "science illustration/infographic": float(min(1.0, sci_infog)),
        "classical oil painting": float(min(1.0, oil)),
        "watercolor": float(min(1.0, watercolor)),
        "children sketches": float(min(1.0, child_sketch)),
        "technical drawing/technical sketch": float(min(1.0, technical)),
        "scribble/sketches": float(min(1.0, scribble)),
    }
    primary = max(scores, key=scores.get)

    return _StyleAnalysis(
        edge_density=ed,
        edge_coherence=ecoh,
        edge_thickness_score=th,
        saturation_mean=s_mean,
        saturation_std=s_std,
        color_clusters=k,
        color_silhouette=sil,
        contrast=contr,
        grayscale_ratio=gray_ratio,
        grain_score=grain,
        hf_ratio=hf_ratio,
        dot_pattern_score=dots,
        straight_line_score=straight_score,
        brush_texture_score=brush_score,
        bokeh_score=bokeh,
        class_scores=scores,
        primary_class=primary,
    )

async def _extract_style_descriptors_async(image_path: Path, debug: bool = False) -> List[str]:
    def _work() -> List[str]:
        a = _analyze_style(image_path, debug=debug)
        return _base_descriptors_for_class(a)
    return await asyncio.to_thread(_work)


# =========================
# Ollama Vision integration
# =========================

VISION_SYS_PROMPT = (
    "You are an expert visual stylist. Describe the visual style of the provided image "
    "concisely, in short comma-separated tags suitable for image generation prompts. "
    "Focus on style, materials, technique, composition, lighting, and color mood. "
    "Avoid meta commentary, avoid long sentences, no punctuation except commas."
)

def _validate_vision_policy(url: str) -> None:
    # Default policy: force localhost unless explicitly allowed
    if APP_ALLOW_REMOTE_VISION:
        # Optional subnet checks would be implemented in the caller-side host controls.
        return
    # Enforce localhost access only
    if not (url.startswith("http://127.0.0.1:") or url.startswith("http://localhost:")):
        raise PermissionError("Remote Ollama vision access is disabled by policy (APP_ALLOW_REMOTE_VISION=0).")

def _b64_from_file(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("ascii")

async def _ollama_vision_describe(path: Path) -> str:
    """
    Call Ollama /api/generate with 'images' containing the base64 image.
    Returns a concise comma-separated style description.
    """
    model = OLLAMA_VISION_MODEL
    url = f"{OLLAMA_VISION_URL.rstrip('/')}/api/generate"
    _validate_vision_policy(url)

    img_b64 = await asyncio.to_thread(_b64_from_file, path)
    prompt = "Describe the visual style of this image in compact, comma-separated tags."

    body = {
        "model": model,
        "prompt": prompt,
        "images": [img_b64],
        "options": {
            "temperature": OLLAMA_VISION_TEMPERATURE,
            "num_ctx": OLLAMA_VISION_NUM_CTX,
            "num_predict": OLLAMA_VISION_NUM_PREDICT,
            "top_k": OLLAMA_VISION_TOP_K,
            "top_p": OLLAMA_VISION_TOP_P,
            "repeat_penalty": OLLAMA_VISION_REPEAT_PENALTY,
        },
        "system": VISION_SYS_PROMPT,
        "stream": False,
    }

    async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_generic(), follow_redirects=True, http2=True) as client:
        resp = await _post_json_with_retries(client, url, json=body)
        if resp.status_code >= 400:
            raise RuntimeError(f"Ollama vision error {resp.status_code}: {resp.text[:512]}")
        js = resp.json()
        # Ollama /api/generate returns {"response": "...", "done": true, ...}
        text = (js.get("response") or "").strip()
        # Normalize: keep only compact comma-separated tags
        text = re.sub(r"\s+", " ", text)
        # Split by commas, trim and deduplicate
        tags = [t.strip(" ,.;-") for t in text.split(",") if t.strip(" ,.;-")]
        out: List[str] = []
        seen = set()
        for t in tags:
            tl = t.lower()
            if tl not in seen:
                out.append(t)
                seen.add(tl)
        return ", ".join(out[:20])


# =========================
# Style fusion helpers
# =========================

def _dedup_join(parts: List[str]) -> str:
    out: List[str] = []
    seen: set[str] = set()
    for p in parts:
        pp = (p or "").strip().strip(",;.")
        if not pp:
            continue
        # Split by commas to normalize granular tags
        tokens = [t.strip() for t in re.split(r"[;,]", pp) if t.strip()]
        for t in tokens:
            tl = t.lower()
            if tl not in seen:
                seen.add(tl)
                out.append(t)
    # Keep concise
    return ", ".join(out[:40])

def _sanitize_style_text(text: str) -> str:
    t = (text or "").strip()
    # Avoid prefixes or meta sentences; keep tags concise
    t = re.sub(r"\s+", " ", t)
    return t

def _apply_deactivate_all_styles(resp: StyleEngineResponse) -> None:
    resp.style_positive = ""
    resp.descriptors_used = []
    resp.vision_text_used = ""
    resp.notes.append("All style inputs deactivated by user.")
    # Keep passthrough fields for transparency


# =========================
# Public orchestration
# =========================

async def build_styles(req: StyleEngineRequest, refs_dir: Union[str, Path] = DEFAULT_REFS_DIR) -> StyleEngineResponse:
    """
    Build additive style text (style_positive) and metadata from:
    - free-form style_text_prompt
    - optional reference image (local file or URL)
    - optional local style analysis (OpenCV/skimage)
    - optional Ollama Vision summary

    This function DOES NOT generate images. It prepares style inputs for image_backend.py,
    which remains responsible for final prompt synthesis (including negative prompts).
    """
    resp = StyleEngineResponse(
        content_positive=(req.content_positive or "").strip(),
        style_text_prompt_raw=(req.style_text_prompt or "").strip(),
    )

    # Resolve reference (if any)
    resolved_path: Optional[Path] = None
    if req.reference_source == "local_file":
        # Expect reference_id provided (previously saved via save_reference_file)
        if req.reference_id:
            try:
                store = ReferenceStore(refs_dir)
                resolved_path = store.get_path(req.reference_id)
                resp.reference_id = req.reference_id
                resp.reference_file_saved = str(resolved_path)
            except Exception as e:
                resp.warnings.append(f"local reference not found: {e}")
        else:
            resp.warnings.append("reference_id missing for local_file source")

    elif req.reference_source == "url":
        if req.reference_url:
            try:
                rid, rpath = await save_reference_from_url(req.reference_url, refs_dir=refs_dir)
                resolved_path = rpath
                resp.reference_id = rid
                resp.reference_file_saved = str(rpath)
            except Exception as e:
                resp.warnings.append(f"download failed: {e}")
        else:
            resp.warnings.append("reference_url missing for url source")

    # Collect style parts
    style_parts: List[str] = []
    if req.style_text_prompt:
        style_parts.append(_sanitize_style_text(req.style_text_prompt))

    # Optional: local style analysis
    if req.use_local_style_features and resolved_path is not None:
        try:
            descriptors = await _extract_style_descriptors_async(resolved_path, debug=False)
            if descriptors:
                resp.descriptors_used = descriptors
                style_parts.append(", ".join(descriptors))
        except Exception as e:
            resp.warnings.append(f"local style analysis failed: {e}")

    # Optional: Ollama vision
    if req.use_ollama_vision and resolved_path is not None:
        try:
            vision_text = await _ollama_vision_describe(resolved_path)
            if vision_text:
                resp.vision_text_used = vision_text
                style_parts.append(vision_text)
        except PermissionError as pe:
            resp.warnings.append(str(pe))
        except Exception as e:
            resp.warnings.append(f"ollama vision failed: {e}")

    # Deduplicate and finalize style_positive
    resp.style_positive = _dedup_join(style_parts)

    # Deactivate all styles (kill-switch)
    if req.deactivate_all_styles:
        _apply_deactivate_all_styles(resp)

    return resp


# =========================
# Module init diagnostics
# =========================

def _log_style_env() -> None:
    print(
        "[STYLE-ENGINE-ENV]",
        f"refs_dir={str(DEFAULT_REFS_DIR)}",
        f"ollama_vision_model={OLLAMA_VISION_MODEL}",
        f"ollama_host={OLLAMA_HOST}:{OLLAMA_PORT}",
        f"vision_remote_allowed={'yes' if APP_ALLOW_REMOTE_VISION else 'no'}",
        f"size_default={APP_IMAGE_WIDTH}x{APP_IMAGE_HEIGHT}",
        f"timeout={HTTP_TIMEOUT_SEC}s",
        f"retries={HTTP_MAX_RETRIES}",
    )

with contextlib.suppress(Exception):
    _log_style_env()


# =========================
# Minimal self-test (optional)
# =========================

if __name__ == "__main__":
    async def _quick_test():
        # This quick test only builds styles; it does not generate images.
        req = StyleEngineRequest(
            content_positive="A cozy reading nook with plants.",
            style_text_prompt="soft natural light, warm tones, minimalist, Scandinavian, matte finish",
            reference_source="none",
            use_local_style_features=False,
            use_ollama_vision=False,
            deactivate_all_styles=False,
            target_backend_name="comfyui",
        )
        out = await build_styles(req)
        print("style_positive:", out.style_positive)
        print("warnings:", out.warnings)
        print("notes:", out.notes)

    asyncio.run(_quick_test())
