# style_engine.py
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
FEATURE_OLLAMA_VISION_DEFAULT = _env_bool01("FEATURE_OLLAMA_VISION_DEFAULT", 0)

# Token caps per source to keep prompt concise and user-prioritized
MAX_TOKENS_STYLE_TEXT = _env_int("STYLE_ENGINE_MAX_TOKENS_STYLE_TEXT", 40)
MAX_TOKENS_VISION = _env_int("STYLE_ENGINE_MAX_TOKENS_VISION", 24)
MAX_TOKENS_DESCRIPTORS = _env_int("STYLE_ENGINE_MAX_TOKENS_DESCRIPTORS", 12)
MAX_TOKENS_FINAL = _env_int("STYLE_ENGINE_MAX_TOKENS_FINAL", 40)


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

    # UI "Reset styles" control: if true AND no new inputs supplied, clear styles.
    # If true BUT new inputs are present in this same request (style_text or usable reference),
    # we IGNORE the reset and apply the new inputs (auto-reactivation).
    deactivate_all_styles: bool = Field(default=False)

    width: Optional[int] = Field(default=None)
    height: Optional[int] = Field(default=None)
    seed: Optional[int] = Field(default=None)

    target_backend_name: str = Field(default="comfyui")

    # Optional routing hints for Ollama Vision (provided by app; not strictly required)
    ollama_vision_mode: Optional[str] = Field(default=None)  # "local" | "remote" | "cloud"
    ollama_vision_local_url: Optional[str] = Field(default=None)
    ollama_vision_remote_url: Optional[str] = Field(default=None)
    ollama_vision_cloud_url: Optional[str] = Field(default=None)

    @field_validator("reference_source")
    @classmethod
    def _validate_source(cls, v: str) -> str:
        vv = (v or "").strip().lower()
        # Legacy "url_file" maps to "url" for compatibility
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
        # Permit unknown but warn, to be robust with broader formats
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
        # Basic size & type checks
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
# Local style features (OpenCV + skimage), lazy
# =========================

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

# Condensed analysis consistent with user's provided version
def _analyze_style(image_path: Path, debug: bool = False) -> _StyleAnalysis:
    cv2, np, canny, rgb2gray, silhouette_score, _HAVE_SKLEARN = _lazy_import_cv_stack()

    def _downscale(img, max_side=640):
        h, w = img.shape[:2]
        ms = max(h, w)
        if ms <= max_side:
            return img
        s = max_side / ms
        return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")

    # Edges
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_s = _downscale(gray, 640)
    edges = canny(gray_s.astype("float32") / 255.0, sigma=1.2)
    edge_density = float(edges.mean())
    gx = cv2.Sobel(gray_s, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_s, cv2.CV_32F, 0, 1, ksize=3)
    mag = (gx * gx + gy * gy) ** 0.5 + 1e-6
    ori = np.arctan2(gy, gx)
    mask = (mag > (mag.mean() + mag.std()))
    if int(mask.sum()) > 100:
        ori_edges = ori[mask]
        var_ori = float(np.var(np.sin(ori_edges)) + np.var(np.cos(ori_edges)))
        edge_coherence = max(0.0, 1.0 - min(1.0, var_ori / 1.0))
    else:
        edge_coherence = 0.0
    edges_u = (edges.astype("uint8") * 255)
    dil = np.maximum(edges_u, cv2.dilate(edges_u, np.ones((3, 3), "uint8"), 1))
    edge_thickness_score = float((dil > 0).mean() - edge_density)

    # Color stats + kmeans
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype("float32") / 255.0
    s_mean = float(s.mean())
    s_std = float(s.std())
    sample = _downscale(bgr, 480)
    flat = sample.reshape(-1, 3).astype("float32")
    if flat.shape[0] > 30000:
        idx = np.random.choice(flat.shape[0], 30000, replace=False)
        flat = flat[idx]
    best_k = 3
    best_sil = -1.0
    prev_compact = None
    for k in range(3, 9):
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 20, 1.0)
        compactness, labels, centers = cv2.kmeans(flat, k, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
        if silhouette_score is not None:
            try:
                sub = flat
                labs = labels.ravel()
                if flat.shape[0] > 2000:
                    ridx = np.random.choice(flat.shape[0], 2000, replace=False)
                    sub = flat[ridx]
                    labs = labs[ridx]
                sil = float(silhouette_score(sub, labs, metric="euclidean"))  # type: ignore
            except Exception:
                sil = -1.0
        else:
            sil = -1.0
        if sil > best_sil:
            best_sil = sil
            best_k = k
        if prev_compact is None:
            prev_compact = compactness
        else:
            if compactness > prev_compact * 0.98:
                break
            prev_compact = compactness

    # Contrast
    gray_f = gray.astype("float32") / 255.0
    contrast = float(gray_f.std())

    # Grayscale ratio
    b, g, r = cv2.split(bgr.astype("float32"))
    diff = (abs(r - g) + abs(g - b) + abs(r - b)) / (3.0 * 255.0)
    grayscale_ratio = float((diff < 0.03).mean())

    # Grain + HF ratio
    gs = _downscale(gray_f, 512)
    lap = cv2.Laplacian(gs, cv2.CV_32F)
    grain = float(lap.var())
    F = np.fft.fftshift(np.fft.fft2(gs))
    mag = np.log1p(np.abs(F))
    H, W = mag.shape
    yy, xx = np.ogrid[:H, :W]
    cy, cx = H // 2, W // 2
    r = ((yy - cy) ** 2 + (xx - cx) ** 2) ** 0.5
    hf_mask = (r > min(H, W) * 0.22) & (r < min(H, W) * 0.48)
    lf_mask = (r < min(H, W) * 0.12)
    hf_ratio = float(mag[hf_mask].mean() / (mag[lf_mask].mean() + 1e-6))

    # Dot pattern
    from skimage.color import rgb2gray as _rgb2gray  # type: ignore
    gray_rgb = _rgb2gray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    grs = _downscale((gray_rgb * 255).astype("uint8"), 512).astype("float32") / 255.0
    F2 = np.fft.fftshift(np.fft.fft2(grs))
    mag2 = np.log1p(np.abs(F2))
    H2, W2 = mag2.shape
    yy2, xx2 = np.ogrid[:H2, :W2]
    cy2, cx2 = H2 // 2, W2 // 2
    r2 = ((yy2 - cy2) ** 2 + (xx2 - cx2) ** 2) ** 0.5
    ring = (r2 > 26) & (r2 < 46)
    neigh = ((r2 > 18) & (r2 < 24)) | ((r2 > 48) & (r2 < 56))
    ring_mean = float(mag2[ring].mean())
    neigh_mean = float(mag2[neigh].mean() + 1e-6)
    dot_pattern_score = float(max(0.0, (ring_mean - neigh_mean) / (neigh_mean + 1e-6) * 3.0))

    # Straight line score
    edges_c = cv2.Canny(_downscale(gray, 800), 80, 160, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(edges_c, 1, 3.14159 / 180.0, threshold=80, minLineLength=40, maxLineGap=3)
    if lines is None or len(lines) == 0:
        straight_line_score = 0.0
    else:
        h, w = edges_c.shape
        per = float(h + w)
        total = 0.0
        for l in lines:
            x1, y1, x2, y2 = l[0]
            total += float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        straight_line_score = float(min(1.0, total / (per * 15.0)))

    # Brush texture (DoG variance)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    L = lab[:, :, 0].astype("float32") / 255.0
    Ls = _downscale(L, 512)
    g1 = cv2.GaussianBlur(Ls, (0, 0), 1.0)
    g2 = cv2.GaussianBlur(Ls, (0, 0), 3.0)
    dog = cv2.absdiff(g1, g2)
    mean = cv2.blur(dog, (9, 9))
    sq = cv2.blur(dog * dog, (9, 9))
    var = sq - mean * mean
    brush_texture_score = float(max(0.0, min(1.0, var.mean() * 50.0)))

    # Bokeh (variance across tiles)
    gs2 = _downscale(gray_f, 512)
    lap2 = cv2.Laplacian(gs2, cv2.CV_32F)
    sharp = cv2.GaussianBlur(lap2 * lap2, (0, 0), 1.0)
    H3, W3 = sharp.shape
    tiles = 6
    th, tw = H3 // tiles, W3 // tiles
    vals = []
    for i in range(tiles):
        for j in range(tiles):
            patch = sharp[i*th:(i+1)*th, j*tw:(j+1)*tw]
            if patch.size:
                vals.append(float(patch.mean()))
    import numpy as np  # type: ignore
    v = float(np.std(np.array(vals, dtype="float32"))) if vals else 0.0
    bokeh_score = float(max(0.0, min(1.5, v * 10.0)))

    # Scoring simple heuristic classes (photo/comic/...); condensed
    photo = 0.0
    if hf_ratio > 1.15: photo += 0.55
    if bokeh_score > 0.26: photo += 0.30
    if s_mean > 0.20 and s_std > 0.09: photo += 0.18
    if contrast > 0.17: photo += 0.07
    photo = max(0.0, photo)

    comic = 0.0
    if edge_coherence >= 0.20 and 0.012 < edge_thickness_score < 0.060 and hf_ratio < 1.12: comic += 0.50
    if bokeh_score < 0.22: comic += 0.12
    if edge_density > 0.040: comic += 0.06
    if dot_pattern_score > 0.9: comic += 0.08
    comic = max(0.0, comic)

    manga = 0.0
    if grayscale_ratio > 0.60 and edge_density > 0.032 and edge_coherence > 0.24 and hf_ratio < 1.08: manga += 0.60
    if dot_pattern_score > 1.0: manga += 0.15
    manga = max(0.0, manga)

    child_sketch = max(0.0, (0.2 if s_mean < 0.16 else 0.0) + (0.15 if hf_ratio < 1.06 else 0.0))

    scores = {
        "photo": float(min(1.0, photo)),
        "comic": float(min(1.0, comic)),
        "manga": float(min(1.0, manga)),
        "children sketches": float(min(1.0, child_sketch)),
    }
    primary = max(scores, key=scores.get)
    return _StyleAnalysis(
        edge_density=edge_density,
        edge_coherence=edge_coherence,
        edge_thickness_score=edge_thickness_score,
        saturation_mean=s_mean,
        saturation_std=s_std,
        color_clusters=best_k,
        color_silhouette=best_sil,
        contrast=contrast,
        grayscale_ratio=grayscale_ratio,
        grain_score=grain,
        hf_ratio=hf_ratio,
        dot_pattern_score=dot_pattern_score,
        straight_line_score=straight_line_score,
        brush_texture_score=brush_texture_score,
        bokeh_score=bokeh_score,
        class_scores=scores,
        primary_class=primary,
    )

def _base_descriptors_for_class(a: _StyleAnalysis) -> List[str]:
    c = a.primary_class
    out: List[str] = []
    if c == "photo":
        out += ["natural lighting", "smooth gradients"]
        if a.hf_ratio > 1.20: out.append("fine detail")
        if a.bokeh_score > 0.30: out.append("shallow depth of field")
        if a.saturation_std > 0.1: out.append("rich colors")
    elif c == "comic":
        out += ["bold outlines" if a.edge_thickness_score >= 0.03 else "clear line art"]
        out.append("flat colors" if a.saturation_mean < 0.18 and a.saturation_std < 0.06 and a.color_clusters <= 5 else "balanced palette")
        if a.contrast > 0.16: out.append("high contrast")
        if a.dot_pattern_score > 0.9: out.append("screen-tone dots")
    elif c == "manga":
        out += ["monochrome", "clear line art"]
        if a.dot_pattern_score > 0.8: out.append("halftone shading")
        if a.contrast > 0.15: out.append("high contrast")
    elif c == "children sketches":
        out += ["simple shapes", "thin uneven lines", "playful composition"]
    else:
        out += ["clean finish"]
    # deduplicate with order
    dedup: List[str] = []
    seen: set[str] = set()
    for t in out:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            dedup.append(t)
    return dedup[:6]

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
    # Remove leading phrases like "style:" or similar
    t = re.sub(r"^(style|stil)[:\s-]+", "", t, flags=re.I)
    # Cap tokens to avoid dominance
    return _retokenize_limit(t, MAX_TOKENS_STYLE_TEXT)

def _apply_deactivate_all_styles(resp: StyleEngineResponse) -> None:
    resp.style_positive = ""
    resp.descriptors_used = []
    resp.vision_text_used = ""
    resp.notes.append("All style inputs deactivated (reset).")


# =========================
# Public orchestration
# =========================

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
      - No synthetic fallback: if inputs are empty or disabled, result can be empty

    Reset policy (deactivate_all_styles):
      - If deactivate_all_styles==True AND no new inputs (no style text, no usable reference) in this request:
          -> clear styles (reset) and return empty style_positive
      - If deactivate_all_styles==True BUT new inputs are present in this request:
          -> ignore reset and apply new inputs (auto-reactivate)

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

    # If deactivate requested and NO new inputs in this very request -> perform reset and return early
    if req.deactivate_all_styles and not has_new_inputs:
        _apply_deactivate_all_styles(resp)
        resp.notes.append("Deactivate applied (no new inputs in request).")
        return resp
    elif req.deactivate_all_styles and has_new_inputs:
        resp.notes.append("Deactivate requested but new inputs detected; applying new styles (auto-reactivate).")

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

    # 2) Ollama vision (only if reference is available)
    if req.use_ollama_vision:
        if resolved_path is None:
            resp.notes.append("Ollama vision requested but no reference image resolved.")
        else:
            # Decide URL based on hints; app supplies via request for "remote" or "cloud"
            ov_mode_url = None
            mode = (req.ollama_vision_mode or "local").lower()
            if mode in {"remote", "cloud"}:
                ov_mode_url = (req.ollama_vision_remote_url or req.ollama_vision_cloud_url or req.ollama_vision_local_url or OLLAMA_VISION_URL)
            else:
                ov_mode_url = (req.ollama_vision_local_url or OLLAMA_VISION_URL)
            try:
                txt = await _ollama_vision_describe(resolved_path, mode_url=ov_mode_url)
                if txt:
                    resp.vision_text_used = _retokenize_limit(txt, MAX_TOKENS_VISION)
                    style_parts_in_order.append(resp.vision_text_used)
                    resp.notes.append(f"Ollama vision used via {ov_mode_url} (mode={mode}, max {MAX_TOKENS_VISION} tokens).")
                else:
                    resp.notes.append("Ollama vision returned empty text.")
            except PermissionError as pe:
                resp.warnings.append(str(pe))
            except Exception as e:
                resp.warnings.append(f"ollama vision failed: {e}")
    else:
        resp.notes.append("Ollama vision disabled by request.")

    # 3) Local descriptors (only if reference is available)
    if req.use_local_style_features:
        if resolved_path is None:
            resp.notes.append("Local style features requested but no reference image resolved.")
        else:
            try:
                desc = await _extract_style_descriptors_async(resolved_path, debug=False)
                # Keep only up to MAX_TOKENS_DESCRIPTORS descriptor tokens
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
                resp.warnings.append(f"local style analysis failed: {e}")
    else:
        resp.notes.append("Local style features disabled by request.")

    # Finalize style: deduplicate tokens across sources (order-preserving: style_text, vision, descriptors)
    tokens = _dedup_ordered_tokens(style_parts_in_order)
    if tokens:
        resp.style_positive = _limit_join(tokens, MAX_TOKENS_FINAL)
        resp.notes.append(f"Final style merged with cap {MAX_TOKENS_FINAL} tokens.")
    else:
        # Explain why empty
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

    # Do NOT apply deactivate here anymore (it was already handled):
    # The reset has priority only when no new inputs are supplied in the same request.

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
        print("A reset style_positive:", out_reset.style_positive, "| notes:", out_reset.notes)

        # Case B: reset + new style text -> should apply style (ignore reset)
        req_apply = StyleEngineRequest(
            content_positive="",
            style_text_prompt="Comic",
            reference_source="none",
            use_local_style_features=False,
            use_ollama_vision=False,
            deactivate_all_styles=True,  # request reset, but new input present
            target_backend_name="comfyui",
        )
        out_apply = await build_styles(req_apply)
        print("B apply style_positive:", out_apply.style_positive, "| notes:", out_apply.notes)

    asyncio.run(_quick_test())
