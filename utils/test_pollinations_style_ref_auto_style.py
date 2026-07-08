# test_pollinations_style_ref_auto_style.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import base64
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import httpx
from pydantic import BaseModel, HttpUrl, ValidationError

# We import the analyzer so we can print a consistent "metrics" line.
# If your style_features.py exposes `analyze_style`, we will use it to print metrics.
# Otherwise, we still call extract_style_with_label(debug=True), which prints basic logs.
try:
    from style_features import analyze_style, extract_style_with_label  # type: ignore
    _HAVE_ANALYZE = True
except Exception:
    from style_features import extract_style_with_label  # type: ignore
    analyze_style = None  # type: ignore
    _HAVE_ANALYZE = False


def load_dotenv_inline(dotenv_path: Optional[str] = None) -> None:
    """
    Minimal .env loader to avoid extra dependencies.
    Loads key=value pairs unless key already exists in environment.
    """
    candidates: List[Path] = []
    if dotenv_path:
        candidates.append(Path(dotenv_path))
    else:
        candidates.append(Path(".env").resolve())
        here = Path(__file__).resolve().parent
        candidates.append(here / ".env")
        candidates.append(here.parent / ".env")
    chosen: Optional[Path] = None
    for p in candidates:
        if p.exists():
            chosen = p
            break
    if not chosen:
        return
    try:
        for raw in chosen.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if " #" in val:
                val = val.split(" #", 1)[0].strip()
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        # Keep silent on dotenv parse errors to avoid breaking tests
        pass


load_dotenv_inline()

# Pollinations API configuration (read from .env when available)
POLL_BASE = (os.getenv("POLLINATIONS_API_BASE", "https://gen.pollinations.ai") or "").rstrip("/")
POLL_KEY = (os.getenv("POLLINATIONS_SECRET", "") or "").strip()
V1_EDITS_PATH = os.getenv("POLLINATIONS_V1_IMAGES_EDITS_PATH", "/v1/images/edits")
IMAGE_MODEL = (os.getenv("POLLINATIONS_IMAGE_MODEL", os.getenv("POLLINATIONS_MODEL", "flux")) or "flux").strip()

# Render dimensions and seed
APP_IMAGE_WIDTH = int(os.getenv("APP_IMAGE_WIDTH", "1280") or "1280")
APP_IMAGE_HEIGHT = int(os.getenv("APP_IMAGE_HEIGHT", "720") or "720")
SEED = int(os.getenv("TEST_SEED", "1234") or "1234")

# I/O locations
STYLE_REFS_DIR = Path(os.getenv("APP_INPUT_DIR", "./outputs/images/refs")).resolve()
STYLE_FILENAME = os.getenv("TEST_STYLE_REFERENCE", "./donald-duck-102-768x576.jpg")
STYLE_FILE = (STYLE_REFS_DIR / STYLE_FILENAME).resolve()

OUTPUT_DIR = Path(os.getenv("APP_OUTPUT_DIR", "./outputs/images")).resolve()

# Prompts
CONTENT_PROMPT = os.getenv("TEST_CONTENT_PROMPT", "Hunde im Weltall").strip()
NEG_PROMPT = os.getenv("TEST_NEG_PROMPT", "").strip()

# Optional: inject explicit primary style name in the prompt (use responsibly)
ENABLE_EXPLICIT_STYLE_NAME = os.getenv("TEST_ENABLE_EXPLICIT_STYLE_NAME", "0").strip() not in {"0", "false", "False", ""}


class V1ImagesItem(BaseModel):
    url: Optional[HttpUrl] = None
    b64_json: Optional[str] = None


class V1ImagesResp(BaseModel):
    data: List[V1ImagesItem] = []


def _headers() -> Dict[str, str]:
    """
    Build authorization headers for Pollinations.
    """
    if not POLL_KEY:
        raise RuntimeError("POLLINATIONS_SECRET missing (ENV)")
    return {"Authorization": f"Bearer {POLL_KEY}"}


async def _client() -> httpx.AsyncClient:
    """
    Create a tuned AsyncClient with HTTP/2, limits, and timeouts.
    """
    limits = httpx.Limits(max_connections=8, max_keepalive_connections=8, keepalive_expiry=30.0)
    timeout = httpx.Timeout(connect=8.0, read=90.0, write=30.0, pool=8.0)
    return httpx.AsyncClient(limits=limits, timeout=timeout, follow_redirects=True, http2=True)


def _edits_endpoint() -> str:
    """
    Build full /v1/images/edits endpoint.
    """
    path = V1_EDITS_PATH if V1_EDITS_PATH.startswith("/") else "/" + V1_EDITS_PATH
    return f"{POLL_BASE}{path}"


def _mime_for(name: str) -> str:
    """
    Return a suitable MIME type for a given filename.
    """
    n = name.lower()
    if n.endswith(".jpg") or n.endswith(".jpeg"):
        return "image/jpeg"
    if n.endswith(".png"):
        return "image/png"
    if n.endswith(".webp"):
        return "image/webp"
    return "application/octet-stream"


def _parse_v1_images_response(resp: httpx.Response) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse the Pollinations V1 images response. Prefer Pydantic parsing, but
    handle relaxed shapes as well.
    """
    try:
        js = resp.json()
        parsed = V1ImagesResp(**js)
        if parsed.data:
            return (str(parsed.data[0].url) if parsed.data[0].url else None, parsed.data[0].b64_json)
    except ValidationError:
        try:
            js = resp.json()
            image_url = js.get("url") or ((js.get("data") or [{}])[0] or {}).get("url")
            b64 = ((js.get("data") or [{}])[0] or {}).get("b64_json") or js.get("b64_json")
            return (image_url, b64)
        except Exception:
            pass
    return (None, None)


def _ensure_dir(p: Path) -> None:
    """
    Ensure the directory exists (mkdir -p).
    """
    p.mkdir(parents=True, exist_ok=True)


async def save_result(image_url: Optional[str], b64: Optional[str], out_dir: Path) -> Path:
    """
    Save the generated image from either a URL or a base64 payload. Returns output path.
    """
    _ensure_dir(out_dir)
    if image_url:
        async with await _client() as client:
            r = await client.get(image_url)
            r.raise_for_status()
            data = r.content
    elif b64:
        data = base64.b64decode(b64.split(",", 1)[1] if "," in b64 else b64, validate=True)
    else:
        raise RuntimeError("No image_url or b64_json returned")

    import hashlib
    digest = hashlib.sha1(data).hexdigest()
    out_path = out_dir / f"{digest}.png"
    out_path.write_bytes(data)
    print(f"[save] wrote {out_path}")
    return out_path


async def call_v1_images_edits_multipart(
    prompt: str,
    image_file: Path,
    *,
    width: int,
    height: int,
    negative_prompt: str = "",
    seed: int = 1234,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Call the V1 /images/edits endpoint using multipart form data with the style reference image.
    """
    if not image_file.exists():
        raise FileNotFoundError(image_file)
    mime = _mime_for(image_file.name)
    files: Dict[str, Any] = {
        "prompt": (None, prompt),
        "response_format": (None, "url"),
        "n": (None, "1"),
        "model": (None, IMAGE_MODEL),
        "size": (None, f"{width}x{height}"),
        "image": (image_file.name, image_file.read_bytes(), mime),
        "seed": (None, str(seed)),
    }
    if negative_prompt:
        files["negative_prompt"] = (None, negative_prompt)

    async with await _client() as client:
        print(f"[pollinations:mp] POST {_edits_endpoint()} model={IMAGE_MODEL} size={width}x{height}")
        print(f"[pollinations:mp] prompt(content-only + auto-style)={prompt[:200]}")
        resp = await client.post(_edits_endpoint(), headers=_headers(), files=files)
        print(f"[pollinations:mp] status {resp.status_code}")
        if resp.status_code >= 400:
            raise RuntimeError(f"pollinations multipart error {resp.status_code}: {resp.text[:500]}")
        return _parse_v1_images_response(resp)


def _format_metrics_line(a: Any) -> str:
    """
    Format a compact metrics line from the StyleAnalysis dataclass.
    Requires 'analyze_style' to be available and to return a StyleAnalysis.
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


def build_auto_style_prompt(content: str, ref_path: Path) -> str:
    """
    Build the final prompt by combining content and detected style descriptors.
    Also prints a consistent 'metrics' line when possible.
    """
    # If we have analyze_style, compute full metrics and print them in a single line
    if _HAVE_ANALYZE and analyze_style is not None:
        a = analyze_style(ref_path)
        print("[style:debug] primary_class:", a.primary_class)
        print("[style:debug] class_scores:", a.class_scores)
        # Always print the compact metrics line for threshold tuning
        print("[style:debug] metrics:", _format_metrics_line(a))
        # Derive short descriptors mapped to class (re-use existing helper via label+descriptors)
        # We still call extract_style_with_label for descriptor generation (and optional extra logs)
        primary_label, desc = a.primary_class, []  # fill below
        try:
            # Call without debug spam to avoid duplicate lines; we already printed metrics above
            from style_features import _base_descriptors_for_class  # type: ignore
            desc = _base_descriptors_for_class(a)
        except Exception:
            # Fallback: use extract_style_with_label(debug=False) to obtain descriptors
            try:
                primary_label, desc = extract_style_with_label(ref_path, debug=False)
            except Exception:
                primary_label, desc = a.primary_class, []
    else:
        # Fallback: rely on extract_style_with_label to print basic logs
        primary_label, desc = extract_style_with_label(ref_path, debug=True)

    # Optionally inject explicit style name (use responsibly)
    explicit_name = primary_label
    parts: List[str] = [content]
    if desc:
        parts.append(", ".join(desc))
    if explicit_name and ENABLE_EXPLICIT_STYLE_NAME:
        parts.append(f"(style: {explicit_name})")
    return ". ".join([p for p in parts if p])


async def main() -> None:
    """
    Entrypoint: detect style, build prompt, call Pollinations, save result.
    """
    print("=== Pollinations V1 Auto-Style Test (multipart) ===")
    print(f"- Using: MODEL={IMAGE_MODEL} ENDPOINT={_edits_endpoint()}")
    print(f"- Ref: {STYLE_FILE}")
    if not STYLE_FILE.exists():
        raise FileNotFoundError(f"Style file not found at {STYLE_FILE}")
    _ = _headers()

    prompt = build_auto_style_prompt(CONTENT_PROMPT, STYLE_FILE)

    img_url, b64 = await call_v1_images_edits_multipart(
        prompt=prompt,
        image_file=STYLE_FILE,
        width=APP_IMAGE_WIDTH,
        height=APP_IMAGE_HEIGHT,
        negative_prompt=NEG_PROMPT,
        seed=SEED,
    )
    print(f"[result] url={bool(img_url)} b64={bool(b64)}")
    out = await save_result(img_url, b64, OUTPUT_DIR)
    print(f"[result] saved: {out}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
