# style_engine.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import base64
import contextlib
import mimetypes
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from pydantic import BaseModel, Field, ConfigDict, field_validator

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

# Reference directory used by the app UI (/static serves outputs/images)
DEFAULT_REFS_DIR = Path(_env_str("APP_REFS_DIR", "./outputs/images/refs")).resolve()

# HTTP retry/timeouts aligned with app.py defaults
HTTP_TIMEOUT_SEC = float(_env_float("APP_OLLAMA_TIMEOUT_SEC", 90.0))
HTTP_MAX_RETRIES = int(_env_int("APP_OLLAMA_MAX_RETRIES", 4))
HTTP_RETRY_BASE_DELAY = float(_env_float("APP_OLLAMA_RETRY_BASE_DELAY", 0.8))

# Vision defaults (local by default; policy can forbid remote)
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

# Remote vision policy: disabled unless explicitly enabled
APP_ALLOW_REMOTE_VISION = _env_bool01("APP_ALLOW_REMOTE_VISION", 0)

# File download limits
MAX_DOWNLOAD_BYTES = _env_int("APP_STYLE_MAX_DOWNLOAD_BYTES", 15_000_000)  # 15 MB
ALLOWED_IMAGE_MIME = {
    "image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif",
}

# Feature toggles
FEATURE_LOCAL_STYLE_ANALYSIS_DEFAULT = _env_bool01("FEATURE_LOCAL_STYLE_ANALYSIS_DEFAULT", 1)
# Default Vision ON for better UX unless overridden in .env
FEATURE_OLLAMA_VISION_DEFAULT = _env_bool01("FEATURE_OLLAMA_VISION_DEFAULT", 1)
# Safety: force local analysis whenever a reference exists (ignores UI flag). Default OFF.
FEATURE_LOCAL_STYLE_ANALYSIS_FORCE_ON = _env_bool01("FEATURE_LOCAL_STYLE_ANALYSIS_FORCE_ON", 0)

# Token caps per source to keep prompt concise and user-prioritized
MAX_TOKENS_STYLE_TEXT = _env_int("STYLE_ENGINE_MAX_TOKENS_STYLE_TEXT", 40)
MAX_TOKENS_VISION = _env_int("STYLE_ENGINE_MAX_TOKENS_VISION", 24)
MAX_TOKENS_DESCRIPTORS = _env_int("STYLE_ENGINE_MAX_TOKENS_DESCRIPTORS", 12)
MAX_TOKENS_FINAL = _env_int("STYLE_ENGINE_MAX_TOKENS_FINAL", 40)

# Local debug toggle controlling [style:debug] lines during local analysis
STYLE_LOCAL_DEBUG = _env_bool01("STYLE_LOCAL_DEBUG", 1)


# =========================
# Pydantic data models
# =========================

class StyleEngineRequest(BaseModel):
    """
    Input to style builder. This module does NOT generate images.
    It returns an additive style string plus metadata.
    """
    model_config = ConfigDict(extra="forbid")

    content_positive: str = Field(default="")
    style_text_prompt: str = Field(default="")

    reference_source: str = Field(default="none")  # "none" | "local_file" | "url"
    reference_id: Optional[str] = Field(default=None)
    reference_url: Optional[str] = Field(default=None)
    reference_strength: float = Field(default=0.6, ge=0.0, le=1.0)

    use_local_style_features: bool = Field(default=FEATURE_LOCAL_STYLE_ANALYSIS_DEFAULT)
    use_ollama_vision: bool = Field(default=FEATURE_OLLAMA_VISION_DEFAULT)

    # UI "Reset styles" control
    deactivate_all_styles: bool = Field(default=False)

    width: Optional[int] = Field(default=None)
    height: Optional[int] = Field(default=None)
    seed: Optional[int] = Field(default=None)

    target_backend_name: str = Field(default="comfyui")

    # Optional routing hints for Ollama Vision
    ollama_vision_mode: Optional[str] = Field(default=None)  # "local" | "remote" | "cloud"
    ollama_vision_local_url: Optional[str] = Field(default=None)
    ollama_vision_remote_url: Optional[str] = Field(default=None)
    ollama_vision_cloud_url: Optional[str] = Field(default=None)

    @field_validator("reference_source")
    @classmethod
    def _validate_source(cls, v: str) -> str:
        vv = (v or "").strip().lower()
        if vv == "url_file":
            vv = "url"
        if vv not in {"none", "local_file", "url"}:
            raise ValueError("reference_source must be one of: none | local_file | url")
        return vv


class StyleEngineResponse(BaseModel):
    """
    Output: additive style text and metadata for transparency.
    The caller (app/image_backend) merges style_positive with the content prompt.
    """
    model_config = ConfigDict(extra="ignore")

    style_positive: str = Field(default="")  # Final additive style tags
    descriptors_used: List[str] = Field(default_factory=list)
    vision_text_used: str = Field(default="")
    reference_file_saved: Optional[str] = Field(default=None)
    reference_id: Optional[str] = Field(default=None)
    warnings: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)

    # passthrough
    content_positive: str = Field(default="")
    style_text_prompt_raw: str = Field(default="")
    negative_hint: str = Field(default="")  # blueprint only; not injected automatically


# =========================
# Reference store (safe local file handling)
# =========================

class ReferenceStore:
    """Safe local store under DEFAULT_REFS_DIR."""

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
        base_raw = Path(filename or "ref.png").name
        try:
            base = self._safe_basename(base_raw)
        except Exception:
            base = "ref.png"

        if "." not in base:
            base += ".png"

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
        safe = self._safe_basename(reference_id)
        p = (self.base / safe).resolve()
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
            r = await client.get(url, headers=headers)
            r.raise_for_status()
            return r
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError, httpx.HTTPStatusError) as e:
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
            r = await client.post(url, json=json, headers=headers)
            r.raise_for_status()
            return r
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError, httpx.HTTPStatusError) as e:
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
    b"RIFF",           # WEBP
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
        print("[STYLE] unknown image magic header; continuing")

def sniff_mime_from_bytes(data: bytes, fallback: str = "application/octet-stream") -> str:
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
# Public reference helpers (controller-compat)
# =========================

def ensure_dirs() -> None:
    """Ensure refs dir exists and is writable."""
    DEFAULT_REFS_DIR.mkdir(parents=True, exist_ok=True)
    test = DEFAULT_REFS_DIR / ".__touch_test__"
    test.write_text("ok", encoding="utf-8")
    with contextlib.suppress(Exception):
        test.unlink()

async def save_reference_from_bytes(filename: str, data: bytes) -> Dict[str, str]:
    """
    Save uploaded bytes to refs dir. Returns dict with reference_id, path, url_path.
    """
    ensure_dirs()
    validate_image_bytes_minimal(data)
    mime = sniff_mime_from_bytes(data, fallback=mimetypes.guess_type(filename or "")[0] or "application/octet-stream")
    safe_name = Path(filename or "reference").name
    safe_name = re.sub(r"[^A-Za-z0-9._\-]", "_", safe_name)
    safe_name = ensure_image_extension(safe_name, mime)
    store = ReferenceStore(DEFAULT_REFS_DIR)
    rid, path = store.put(safe_name, data)
    return {"reference_id": rid, "path": str(path), "url_path": f"/static/{Path(path).name}"}

async def save_reference_from_url(url: str, filename_hint: Optional[str] = None) -> Dict[str, str]:
    """
    Download an image by URL and store locally. Returns same dict as save_reference_from_bytes.
    """
    ensure_dirs()
    url = (url or "").strip()
    if not url or not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("invalid url")

    async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_generic(), follow_redirects=True, http2=True) as client:
        r = await _get_with_retries(client, url)
        cl = int(r.headers.get("Content-Length", "0") or "0")
        if cl and cl > MAX_DOWNLOAD_BYTES:
            raise ValueError(f"file too large: {cl} bytes")
        data = r.content
        if len(data) > MAX_DOWNLOAD_BYTES:
            raise ValueError(f"file too large after download: {len(data)} bytes")
        ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()

    validate_image_bytes_minimal(data)
    mime = sniff_mime_from_bytes(data, fallback=ctype or "application/octet-stream")

    name = filename_hint or Path(httpx.URL(url).path or "reference").name or "reference"
    name = re.sub(r"[^A-Za-z0-9._\-]", "_", name)
    name = ensure_image_extension(name, mime)

    store = ReferenceStore(DEFAULT_REFS_DIR)
    rid, path = store.put(name, data)
    return {"reference_id": rid, "path": str(path), "url_path": f"/static/{Path(path).name}"}


# =========================
# Local style features (strict 1:1 integration)
# =========================

# We embed the exact logic from style_features.py so that metrics, gates,
# class scoring, and descriptor mapping are identical and reproducible.

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    from skimage.feature import canny  # type: ignore
    from skimage.color import rgb2gray  # type: ignore
    try:
        from sklearn.metrics import silhouette_score  # type: ignore
        _HAVE_SKLEARN = True
    except Exception:
        _HAVE_SKLEARN = False
except Exception as _e:
    # Defer errors until analysis is actually called to avoid import-time failures in environments without OpenCV.
    cv2 = None  # type: ignore
    np = None  # type: ignore
    canny = None  # type: ignore
    rgb2gray = None  # type: ignore
    silhouette_score = None  # type: ignore
    _HAVE_SKLEARN = False
    _IMPORT_ERR = _e
else:
    _IMPORT_ERR = None

@dataclass
class StyleAnalysis:
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

def _ensure_cv_stack_ready() -> None:
    if _IMPORT_ERR is not None or cv2 is None or np is None or canny is None or rgb2gray is None:
        raise RuntimeError(f"CV stack not available: {_IMPORT_ERR}")

def _load_bgr(path: Path) -> "np.ndarray":
    _ensure_cv_stack_ready()
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return img  # BGR

def _downscale(img: "np.ndarray", max_side: int = 640) -> "np.ndarray":
    h, w = img.shape[:2]
    ms = max(h, w)
    if ms <= max_side:
        return img
    s = max_side / ms
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

def _edge_metrics(bgr: "np.ndarray") -> Tuple[float, float, float]:
    # Kanten-Dichte, Orientierungs-Kohärenz (Proxy), und Kanten-"Dicke"
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = _downscale(gray, 640)
    edges = canny(gray.astype(np.float32) / 255.0, sigma=1.2)
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

    edges_u = (edges.astype(np.uint8) * 255)
    dil = cv2.dilate(edges_u, np.ones((3, 3), np.uint8), 1)
    thickness_score = float((dil > 0).mean() - edge_density)
    return edge_density, edge_coherence, thickness_score

def _estimate_silhouette_fallback(X: "np.ndarray", labels: "np.ndarray") -> float:
    # Lightweight Silhouette-Approximation ohne sklearn
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
    # Sättigungs-Statistik + einfache KMeans-Paletten-Clustering-Qualität
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    s_mean = float(s.mean())
    s_std = float(s.std())

    sample = _downscale(bgr, 480)
    flat = sample.reshape(-1, 3).astype(np.float32)
    if flat.shape[0] > 40000:
        idx = np.random.choice(flat.shape[0], 40000, replace=False)
        flat = flat[idx]

    best_k = 3
    best_sil = -1.0
    prev_compact = None
    for k in range(3, 9):
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 20, 1.0)
        compactness, labels, _ = cv2.kmeans(flat, k, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
        if _HAVE_SKLEARN:
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
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return float(gray.std())

def _grayscale_ratio(bgr: "np.ndarray") -> float:
    b, g, r = cv2.split(bgr.astype(np.float32))
    diff = (np.abs(r - g) + np.abs(g - b) + np.abs(r - b)) / (3.0 * 255.0)
    return float((diff < 0.03).mean())

def _grain_and_hf(bgr: "np.ndarray") -> Tuple[float, float]:
    # Korn/Noise via Laplacian-Varianz; HF/LF Spektrum via FFT-Ringe
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    g_small = _downscale((gray * 255).astype(np.uint8), 512).astype(np.float32) / 255.0
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
    # Halftone-Ring-Detektor (Frequenzbereich)
    gray = rgb2gray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    gray = _downscale((gray * 255).astype(np.uint8), 512).astype(np.float32) / 255.0
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
    # Hough-Linienlänge relativ zum Umfang
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
    # DoG-Varianz als grober Pinseltextur-Hinweis
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    L = lab[:, :, 0].astype(np.float32) / 255.0
    Ls = _downscale(L, 512)
    g1 = cv2.GaussianBlur(Ls, (0, 0), 1.0)
    g2 = cv2.GaussianBlur(Ls, (0, 0), 3.0)
    dog = cv2.absdiff(g1, g2)
    mean = cv2.blur(dog, (9, 9))
    sq = cv2.blur(dog * dog, (9, 9))
    var = sq - mean * mean
    return float(np.clip(var.mean() * 50.0, 0.0, 1.0))

def _bokeh_score(bgr: "np.ndarray") -> float:
    # Varianz lokaler Schärfe → Hinweis auf DoF/Bokeh
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
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
    vals = np.array(vals, dtype=np.float32)
    if vals.size < 4:
        return 0.0
    v = float(np.std(vals))
    return float(np.clip(v * 10.0, 0.0, 1.5))

def analyze_style(path: Path) -> StyleAnalysis:
    """
    Full analysis strictly identical to style_features.py:
    - Computes metrics
    - Applies the same gates and class scoring
    - Returns StyleAnalysis with class scores and primary_class
    """
    bgr = _load_bgr(path)

    ed, ecoh, th = _edge_metrics(bgr)
    s_mean, s_std, k, sil = _color_metrics(bgr)
    contr = _contrast_metric(bgr)
    gray_ratio = _grayscale_ratio(bgr)
    grain, hf_ratio = _grain_and_hf(bgr)
    dots = _dot_pattern_score(bgr)
    straight_score = _straight_line_score(bgr)
    brush_score = _brush_texture_score(bgr)
    bokeh = _bokeh_score(bgr)

    # Photo
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

    # Photo-like gate
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

    # Relaxed photo-like gate (Policy)
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

    # Anti-photo Penalties
    if ecoh > 0.32 and 0.012 < th < 0.050 and bokeh < 0.22 and hf_ratio < 1.14:
        photo -= 0.35
    if (ecoh <= 0.06 and th >= 0.10 and hf_ratio < 0.95 and k <= 4 and sil >= 0.45):
        photo -= 0.05
    if dots > 1.1 and gray_ratio > 0.45:
        photo -= 0.20

    photo = max(0.0, photo)

    # Comic (UNVERÄNDERT LASSEN)
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

    # Colored cartoon guard (tightened)
    if (0.16 <= s_mean <= 0.42) and (0.12 <= s_std <= 0.35) and (4 <= k <= 7) \
       and (th >= 0.10) and (ecoh <= 0.10) and (bokeh < 0.24) and (hf_ratio < 1.10) \
       and (straight_score < 0.10) and (dots < 1.1):
        comic += 0.24

    comic = max(0.0, comic)

    # Manga
    manga = 0.0
    if gray_ratio > 0.60 and ed > 0.032 and ecoh > 0.24 and hf_ratio < 1.08:
        manga += 0.60
    if dots > 1.0:
        manga += 0.15
    if s_mean > 0.15:
        manga -= 0.10
    manga = max(0.0, manga)

    # Children sketches
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

    # Classical oil painting
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

    # Science infographic / poster-like graphics
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

    # Watercolor
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

    # Technical drawing
    technical = 0.0
    if straight_score > 0.32:
        technical += 0.45
    if ecoh > 0.32:
        technical += 0.25
    if gray_ratio > 0.45 or s_std < 0.06:
        technical += 0.20
    if ed > 0.028 and th < 0.02:
        technical += 0.10
    technical = max(0.0, technical)

    # Scribble/sketches
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

    return StyleAnalysis(
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

def _base_descriptors_for_class(a: StyleAnalysis) -> List[str]:
    """
    Exact descriptor mapping copied from style_features.py to ensure parity.
    """
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

def extract_style_descriptors(image_path: Path, debug: bool = False) -> List[str]:
    """
    Return short style descriptors for a given image.
    When debug=True, print logs identical to style_features.py:
      - [style:debug] primary_class
      - [style:debug] class_scores
      - [style:debug] metrics line
      - [style:debug] descriptors
    """
    a = analyze_style(image_path)
    desc = _base_descriptors_for_class(a)
    if debug:
        # metrics line format must match test script expectations
        print("[style:debug] primary_class:", a.primary_class)
        print("[style:debug] class_scores:", a.class_scores)
        print(
            "[style:debug] metrics:",
            f"ed={a.edge_density:.3f}",
            f"ecoh={a.edge_coherence:.3f}",
            f"th={a.edge_thickness_score:.3f}",
            f"s_mean={a.saturation_mean:.3f}",
            f"s_std={a.saturation_std:.3f}",
            f"k={a.color_clusters}",
            f"sil={a.color_silhouette:.3f}",
            f"contrast={a.contrast:.3f}",
            f"gray_ratio={a.grayscale_ratio:.3f}",
            f"grain={a.grain_score:.1f}",
            f"hf_ratio={a.hf_ratio:.2f}",
            f"dots={a.dot_pattern_score:.2f}",
            f"straight={a.straight_line_score:.2f}",
            f"brush={a.brush_texture_score:.2f}",
            f"bokeh={a.bokeh_score:.2f}",
        )
        print("[style:debug] descriptors:", desc)
    return desc

def detect_primary_style_label(image_path: Path) -> str:
    return analyze_style(image_path).primary_class

def extract_style_with_label(image_path: Path, debug: bool = False) -> Tuple[str, List[str]]:
    a = analyze_style(image_path)
    desc = _base_descriptors_for_class(a)
    if debug:
        print("[style:debug] primary_class:", a.primary_class)
        print("[style:debug] class_scores:", a.class_scores)
        print("[style:debug] descriptors:", desc)
    return a.primary_class, desc


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
    if APP_ALLOW_REMOTE_VISION:
        return
    # Enforce localhost if remote vision not allowed
    if not (url.startswith("http://127.0.0.1:") or url.startswith("http://localhost:")):
        raise PermissionError("Remote Ollama vision access is disabled by policy (APP_ALLOW_REMOTE_VISION=0).")

def _b64_from_file(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("ascii")

def _tokenize(text: str) -> List[str]:
    parts = [t.strip(" ,.;-") for t in (text or "").split(",")]
    return [t for t in parts if t]

def _retokenize_limit(text: str, limit: int) -> str:
    toks = _tokenize(text)
    return ", ".join(toks[:max(0, limit)])

async def _ollama_vision_describe(path: Path, mode_url: Optional[str] = None) -> str:
    """
    Use Ollama /api/generate with 'images' to extract compact comma-separated style tags.
    mode_url can override default OLLAMA_VISION_URL (app may pass via request).
    """
    url_base = (mode_url or OLLAMA_VISION_URL).rstrip("/")
    _validate_vision_policy(url_base)
    url = f"{url_base}/api/generate"
    img_b64 = await asyncio.to_thread(_b64_from_file, path)

    body = {
        "model": OLLAMA_VISION_MODEL,
        "prompt": "Describe the visual style of this image in compact, comma-separated tags.",
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
        r = await _post_json_with_retries(client, url, json=body)
        js = r.json()
        text = (js.get("response") or "").strip()
        text = re.sub(r"\s+", " ", text)
        tags = [t.strip(" ,.;-") for t in text.split(",") if t.strip(" ,.;-")]
        out: List[str] = []
        seen: set[str] = set()
        for t in tags:
            tl = t.lower()
            if tl not in seen:
                seen.add(tl)
                out.append(t)
        return ", ".join(out[:MAX_TOKENS_VISION])


# =========================
# Style fusion helpers
# =========================

def _dedup_ordered_tokens(parts: List[str]) -> List[str]:
    """
    Split by commas, trim and deduplicate preserving order across multiple inputs.
    """
    out: List[str] = []
    seen: set[str] = set()
    for p in parts:
        if not p:
            continue
        tokens = [t.strip() for t in re.split(r",", p) if t.strip()]
        for t in tokens:
            tl = t.lower()
            if tl not in seen:
                seen.add(tl)
                out.append(t)
    return out

def _limit_join(tokens: List[str], limit: int) -> str:
    return ", ".join(tokens[:max(0, limit)])

def _sanitize_style_text(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"^(style|stil)[:\s-]+", "", t, flags=re.I)
    return _retokenize_limit(t, MAX_TOKENS_STYLE_TEXT)

def _apply_deactivate_all_styles(resp: StyleEngineResponse) -> None:
    resp.style_positive = ""
    resp.descriptors_used = []
    resp.vision_text_used = ""
    resp.notes.append("All style inputs deactivated (reset).")


# =========================
# Public orchestration
# =========================

def _format_metrics_line(a: StyleAnalysis) -> str:
    """
    Produce the exact compact metrics line expected by tests.
    """
    return " ".join([
        f"ed={a.edge_density:.3f}",
        f"ecoh={a.edge_coherence:.3f}",
        f"th={a.edge_thickness_score:.3f}",
        f"s_mean={a.saturation_mean:.3f}",
        f"s_std={a.saturation_std:.3f}",
        f"k={a.color_clusters}",
        f"sil={a.color_silhouette:.3f}",
        f"contrast={a.contrast:.3f}",
        f"gray_ratio={a.grayscale_ratio:.3f}",
        f"grain={a.grain_score:.1f}",
        f"hf_ratio={a.hf_ratio:.2f}",
        f"dots={a.dot_pattern_score:.2f}",
        f"straight={a.straight_line_score:.2f}",
        f"brush={a.brush_texture_score:.2f}",
        f"bokeh={a.bokeh_score:.2f}",
    ])

async def build_styles(req: StyleEngineRequest, refs_dir: Union[str, Path] = DEFAULT_REFS_DIR) -> StyleEngineResponse:
    """
    Build additive style text from:
      1) User style text (priority source)
      2) Ollama Vision tags (if enabled and reference provided)
      3) Local descriptors from image analysis (if enabled and reference provided)

    Merge policy:
      - Order-preserving deduplication across sources
      - Per-source token caps: style_text (40), vision (24), descriptors (12)
      - Final cap MAX_TOKENS_FINAL (40)

    Reset policy:
      - If deactivate_all_styles==True AND no new inputs in this request:
          -> clear styles and return empty
      - If deactivate_all_styles==True BUT new inputs present:
          -> ignore reset and apply new inputs

    Remote vision policy:
      - If APP_ALLOW_REMOTE_VISION=0, only localhost URLs are permitted
      - Otherwise, remote/cloud URLs are allowed
    """
    ensure_dirs()
    resp = StyleEngineResponse(
        content_positive=(req.content_positive or "").strip(),
        style_text_prompt_raw=(req.style_text_prompt or "").strip(),
    )

    # Resolve reference (if any)
    resolved_path: Optional[Path] = None
    if req.reference_source == "local_file":
        if req.reference_id:
            try:
                store = ReferenceStore(refs_dir)
                p = store.get_path(req.reference_id)
                resolved_path = p
                resp.reference_id = req.reference_id
                resp.reference_file_saved = str(p)
            except Exception as e:
                resp.warnings.append(f"local reference not found: {e}")
        else:
            resp.warnings.append("reference_id missing for local_file source")
    elif req.reference_source == "url":
        if req.reference_url:
            try:
                saved = await save_reference_from_url(req.reference_url, filename_hint=None)
                resp.reference_id = saved.get("reference_id")
                spath = saved.get("path")
                if spath:
                    resolved_path = Path(spath)
                resp.reference_file_saved = spath
            except Exception as e:
                resp.warnings.append(f"download failed: {e}")
        else:
            resp.warnings.append("reference_url missing for url source")
    else:
        resp.notes.append("No reference image provided.")

    # Determine whether this request carries new inputs
    has_new_style_text = bool((req.style_text_prompt or "").strip())
    has_new_reference = resolved_path is not None
    has_new_inputs = has_new_style_text or has_new_reference

    # Deactivate handling
    if req.deactivate_all_styles and not has_new_inputs:
        _apply_deactivate_all_styles(resp)
        resp.notes.append("Deactivate applied (no new inputs in request).")
        return resp
    elif req.deactivate_all_styles and has_new_inputs:
        resp.notes.append("Deactivate requested but new inputs detected; applying new styles (auto-reactivate).")

    # Safety mode toggles
    if has_new_reference:
        if FEATURE_LOCAL_STYLE_ANALYSIS_FORCE_ON and not req.use_local_style_features:
            req.use_local_style_features = True
            resp.notes.append("Local style analysis forced on by env (FEATURE_LOCAL_STYLE_ANALYSIS_FORCE_ON=1).")
        if (not req.use_ollama_vision) and (not req.use_local_style_features):
            req.use_local_style_features = True
            resp.notes.append("Local style safety-net enabled: reference provided, vision disabled.")

    # Collect style parts in priority order
    style_parts_in_order: List[str] = []

    # 1) User style text
    if has_new_style_text:
        st = _sanitize_style_text(req.style_text_prompt)
        if st:
            style_parts_in_order.append(st)
            resp.notes.append(f"Applied style_text_prompt (up to {MAX_TOKENS_STYLE_TEXT} tokens).")
    else:
        resp.notes.append("No style_text_prompt provided.")

    # 2) Ollama vision
    if req.use_ollama_vision:
        if resolved_path is None:
            resp.notes.append("Ollama vision requested but no reference image resolved.")
        else:
            ov_mode_url = None
            mode = (req.ollama_vision_mode or "local").lower()
            if mode in {"remote", "cloud"}:
                ov_mode_url = (req.ollama_vision_remote_url or req.ollama_vision_cloud_url or req.ollama_vision_local_url or OLLAMA_VISION_URL)
            else:
                ov_mode_url = (req.ollama_vision_local_url or OLLAMA_VISION_URL)
            try:
                print(f"[STYLE][vision] mode={mode} url={ov_mode_url} allow_remote={APP_ALLOW_REMOTE_VISION} ref={resolved_path}")
                txt = await _ollama_vision_describe(resolved_path, mode_url=ov_mode_url)
                print(f"[STYLE][vision] response_len={len(txt) if txt else 0} sample='{(txt or '')[:160]}'")
                if txt:
                    resp.vision_text_used = _retokenize_limit(txt, MAX_TOKENS_VISION)
                    style_parts_in_order.append(resp.vision_text_used)
                    resp.notes.append(f"Ollama vision used via {ov_mode_url} (mode={mode}, max {MAX_TOKENS_VISION} tokens).")
                else:
                    resp.notes.append("Ollama vision returned empty text.")
            except PermissionError as pe:
                print(f"[STYLE][vision][policy] {pe}")
                resp.warnings.append(str(pe))
            except Exception as e:
                print(f"[STYLE][vision][error] {e}")
                resp.warnings.append(f"ollama vision failed: {e}")
    else:
        resp.notes.append("Ollama vision disabled by request.")

    # 3) Local descriptors with strict debug output parity
    if req.use_local_style_features:
        if resolved_path is None:
            resp.notes.append("Local style features requested but no reference image resolved.")
        else:
            try:
                # Compute full analysis once for logging parity and descriptor generation
                a = analyze_style(resolved_path)
                if STYLE_LOCAL_DEBUG:
                    print("[style:debug] primary_class:", a.primary_class)
                    print("[style:debug] class_scores:", a.class_scores)
                    print("[style:debug] metrics:", _format_metrics_line(a))
                # Use exact descriptor mapping
                try:
                    desc = _base_descriptors_for_class(a)
                except Exception:
                    # Fallback to extract_style_with_label if something unexpected happens
                    _, desc = extract_style_with_label(resolved_path, debug=False)
                if STYLE_LOCAL_DEBUG:
                    print("[style:debug] descriptors:", desc)
                # Cap and deduplicate descriptors locally before merging
                if desc:
                    dedup_desc: List[str] = []
                    seen: set[str] = set()
                    for d in desc:
                        dl = d.lower().strip()
                        if dl and dl not in seen:
                            seen.add(dl)
                            dedup_desc.append(d.strip())
                        if len(dedup_desc) >= MAX_TOKENS_DESCRIPTORS:
                            break
                    resp.descriptors_used = dedup_desc
                    if dedup_desc:
                        style_parts_in_order.append(", ".join(dedup_desc))
                        resp.notes.append(f"Local descriptors used (max {MAX_TOKENS_DESCRIPTORS} tokens).")
                else:
                    resp.notes.append("Local analysis produced no descriptors.")
            except Exception as e:
                print(f"[STYLE][local][error] {e}")
                resp.warnings.append(f"local style analysis failed: {e}")
    else:
        resp.notes.append("Local style features disabled by request.")

    # Finalize style: deduplicate tokens across sources (order-preserving)
    tokens = _dedup_ordered_tokens(style_parts_in_order)
    if tokens:
        resp.style_positive = _limit_join(tokens, MAX_TOKENS_FINAL)
        resp.notes.append(f"Final style merged with cap {MAX_TOKENS_FINAL} tokens.")
    else:
        reasons: List[str] = []
        if not has_new_style_text:
            reasons.append("no style_text_prompt")
        if resolved_path is None:
            reasons.append("no reference image")
        if not req.use_local_style_features:
            reasons.append("local_features_disabled")
        if not req.use_ollama_vision:
            reasons.append("vision_disabled")
        if reasons:
            resp.warnings.append("style_positive is empty: " + ", ".join(reasons))

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
        f"caps(style={MAX_TOKENS_STYLE_TEXT}, vision={MAX_TOKENS_VISION}, desc={MAX_TOKENS_DESCRIPTORS}, final={MAX_TOKENS_FINAL})",
        f"feature_defaults(local={FEATURE_LOCAL_STYLE_ANALYSIS_DEFAULT}, vision={FEATURE_OLLAMA_VISION_DEFAULT}, force_local={FEATURE_LOCAL_STYLE_ANALYSIS_FORCE_ON})",
        f"local_debug={'on' if STYLE_LOCAL_DEBUG else 'off'}",
    )

with contextlib.suppress(Exception):
    _log_style_env()


# =========================
# Minimal self-test (optional)
# =========================

if __name__ == "__main__":
    async def _quick_test():
        ensure_dirs()

        # Case A: pure reset (no new inputs) -> empty
        req_reset = StyleEngineRequest(
            content_positive="",
            style_text_prompt="",
            reference_source="none",
            use_local_style_features=False,
            use_ollama_vision=False,
            deactivate_all_styles=True,
            target_backend_name="comfyui",
        )
        out_reset = await build_styles(req_reset)
        print("A reset style_positive:", out_reset.style_positive, "| notes:", out_reset.notes, "| warnings:", out_reset.warnings)

        # Case B: reset + new style text -> should apply style (ignore reset)
        req_apply = StyleEngineRequest(
            content_positive="",
            style_text_prompt="thin precise lines, flat colors, clean layout",
            reference_source="none",
            use_local_style_features=False,
            use_ollama_vision=False,
            deactivate_all_styles=True,  # request reset, but new input present
            target_backend_name="comfyui",
        )
        out_apply = await build_styles(req_apply)
        print("B apply style_positive:", out_apply.style_positive, "| notes:", out_apply.notes, "| warnings:", out_apply.warnings)

    asyncio.run(_quick_test())
