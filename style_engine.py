# style_engine.py
from __future__ import annotations

import io
import os
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

from pydantic import BaseModel, Field, ValidationError, field_validator

# Pillow is required for robust image validation (imghdr was removed in Python 3.13)
try:
    from PIL import Image
except Exception as e:
    raise RuntimeError(
        "Pillow (PIL) is required. Please install in your active venv: pip install Pillow"
    ) from e


# Allowed image formats (uppercased as Pillow reports them)
ALLOWED_FORMATS: set[str] = {"PNG", "JPEG", "JPG", "WEBP"}

# Static root and style directory (local only)
STATIC_ROOT = Path(os.environ.get("STATIC_ROOT", "static")).resolve()
STYLE_DIR = (STATIC_ROOT / "style").resolve()


def ensure_style_dir() -> None:
    """Ensure the style directory exists."""
    STYLE_DIR.mkdir(parents=True, exist_ok=True)


def safe_filename(name: str) -> str:
    """Sanitize filename to avoid path traversal and unsafe characters."""
    name = os.path.basename(name)
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name or "style_ref"


def detect_image_format(data: bytes) -> Tuple[bool, Optional[str], Optional[Tuple[int, int]]]:
    """
    Validate image bytes via Pillow and return (ok, format, size).
    - Uses verify() for integrity, then re-opens for metadata.
    """
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()  # integrity check
        with Image.open(io.BytesIO(data)) as img2:
            fmt = (img2.format or "").upper()
            size = img2.size
        return True, fmt, size
    except Exception:
        return False, None, None


class StyleState(BaseModel):
    """
    Runtime style state used by the engine and backends.
    - enabled: whether style reference should be applied
    - strength: 0..1 influence factor (backend-specific mapping)
    - rel: relative path under /static (e.g., "style/ref.png")
    - use_reference: mirror of config toggle; not strictly required but useful
    - style_preset: optional preset name/id
    """
    enabled: bool = Field(default=False)
    strength: float = Field(ge=0.0, le=1.0, default=0.6)
    rel: Optional[str] = None
    use_reference: bool = False
    style_preset: Optional[str] = None

    @field_validator("rel")
    @classmethod
    def _validate_rel(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.lstrip("/")
        if ".." in v:
            raise ValueError("invalid relative path")
        return v


class StyleConfig(BaseModel):
    """
    Configuration model mirroring StyleState and extended for app startup/shutdown usage.

    Fields:
    - enabled: whether style reference should be applied
    - strength: 0..1 influence factor
    - rel: relative path under /static (e.g., "style/ref.png")
    - persisted_path: optional filesystem path where this config was loaded/saved
    - style_preset: optional name/id of a style preset selected in the UI
    - use_reference: UI toggle whether to use the reference image
    """
    enabled: bool = Field(default=False)
    strength: float = Field(ge=0.0, le=1.0, default=0.6)
    rel: Optional[str] = None
    persisted_path: Optional[str] = None
    style_preset: Optional[str] = None
    use_reference: bool = False

    @field_validator("rel")
    @classmethod
    def _validate_rel(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.lstrip("/")
        if ".." in v:
            raise ValueError("invalid relative path")
        return v


class StyleEngine:
    """
    Manage the active style reference image and parameters.
    - Stores reference safely under static/style
    - Exposes current StyleState/StyleConfig
    - Validates uploaded images with Pillow
    """

    def __init__(self, static_root: Path | str = STATIC_ROOT) -> None:
        self._static_root = Path(static_root).resolve()
        ensure_style_dir()
        self._state = StyleState(enabled=False, strength=0.6, rel=None)

    def get_state(self) -> StyleState:
        """Return current runtime style state."""
        return self._state

    def get_config(self) -> StyleConfig:
        """Return current style configuration (alias view)."""
        # Keep fields that exist on StyleState; persist-only fields remain None
        return StyleConfig(**self._state.model_dump())

    def update_params(
        self,
        enabled: bool,
        strength: float,
        use_reference: Optional[bool] = None,
        style_preset: Optional[str] = None,
    ) -> StyleState:
        """Update style params with validation; optional toggles can be passed."""
        kwargs = self._state.model_dump()
        kwargs.update({"enabled": enabled, "strength": strength})
        if use_reference is not None:
            kwargs["use_reference"] = bool(use_reference)
        if style_preset is not None:
            kwargs["style_preset"] = style_preset
        try:
            valid = StyleState(**kwargs)
        except ValidationError as e:
            raise ValueError(f"Invalid style params: {e}") from e
        self._state = valid
        return self._state

    def set_reference_file(self, filename: str, content: bytes) -> StyleState:
        """
        Persist a validated reference image to static/style and update state.rel.
        - Accepts only allowed formats (PNG/JPEG/WEBP)
        - Writes atomically via a temporary file replace
        """
        ok, fmt, _ = detect_image_format(content)
        if not ok or not fmt:
            raise ValueError("Invalid image file")
        fmt = fmt.upper()
        if fmt == "JPG":
            fmt = "JPEG"
        if fmt not in ALLOWED_FORMATS:
            raise ValueError(f"Unsupported format: {fmt}")

        base = os.path.splitext(safe_filename(filename))[0]
        ext = ".jpg" if fmt == "JPEG" else f".{fmt.lower()}"
        final_name = f"{base}{ext}"

        ensure_style_dir()
        tmp_path = STYLE_DIR / (final_name + ".tmp")
        dst_path = STYLE_DIR / final_name

        with open(tmp_path, "wb") as f:
            f.write(content)
        # Atomic replace to avoid partial files
        tmp_path.replace(dst_path)

        # Compute relative path under /static
        try:
            rel_path = dst_path.relative_to(self._static_root)
        except ValueError:
            rel_path = Path("style") / final_name  # fallback if style dir is outside static

        self._state.rel = str(rel_path).replace(os.sep, "/")
        return self._state

    def clear_reference(self) -> StyleState:
        """Clear only the reference pointer; keep the file on disk."""
        self._state.rel = None
        return self._state

    def current_file_path(self) -> Optional[Path]:
        """Return absolute path to the current reference file, if any."""
        if not self._state.rel:
            return None
        return (self._static_root / self._state.rel).resolve()


class ReferenceMeta(BaseModel):
    """Lightweight metadata for a stored reference image."""
    rel: str  # relative path under /static (e.g. "style/foo.png")
    format: str
    width: int
    height: int
    created_at: float = Field(default_factory=lambda: datetime.now().timestamp())

    @field_validator("rel")
    @classmethod
    def _validate_rel(cls, v: str) -> str:
        v = v.lstrip("/")
        if ".." in v:
            raise ValueError("invalid relative path")
        if not v.startswith("style/"):
            # Enforce namespace under static/style
            raise ValueError("reference must reside under 'style/'")
        return v

    @field_validator("format")
    @classmethod
    def _validate_fmt(cls, v: str) -> str:
        return v.upper()


class ReferenceStore:
    """
    Store and manage reference images under static/style.
    - Validates images via Pillow
    - Returns only relative paths under /static
    - Provides simple list/get/remove helpers
    """

    def __init__(self, static_root: Path | str = STATIC_ROOT) -> None:
        self._static_root = Path(static_root).resolve()
        ensure_style_dir()
        self._lock = threading.Lock()

    def _unique_name(self, base: str, ext: str) -> str:
        """
        Generate a unique filename if the target already exists.
        Uses a short UUID suffix to avoid collisions.
        """
        candidate = f"{base}{ext}"
        dst = STYLE_DIR / candidate
        if not dst.exists():
            return candidate
        short = uuid.uuid4().hex[:8]
        return f"{base}_{short}{ext}"

    def save(self, filename: str, content: bytes) -> ReferenceMeta:
        """
        Validate and save an image under static/style, returning its metadata.
        - Only PNG/JPEG/WEBP
        - Atomic write via temporary file
        """
        ok, fmt, size = detect_image_format(content)
        if not ok or not fmt or not size:
            raise ValueError("Invalid image file")
        fmt = fmt.upper()
        if fmt == "JPG":
            fmt = "JPEG"
        if fmt not in ALLOWED_FORMATS:
            raise ValueError(f"Unsupported format: {fmt}")

        base = os.path.splitext(safe_filename(filename))[0]
        ext = ".jpg" if fmt == "JPEG" else f".{fmt.lower()}"
        final_name = self._unique_name(base, ext)

        tmp_path = STYLE_DIR / (final_name + ".tmp")
        dst_path = STYLE_DIR / final_name

        with self._lock:
            with open(tmp_path, "wb") as f:
                f.write(content)
            tmp_path.replace(dst_path)

        try:
            rel_path = dst_path.relative_to(self._static_root)
        except ValueError:
            rel_path = Path("style") / final_name

        w, h = int(size[0]), int(size[1])
        return ReferenceMeta(
            rel=str(rel_path).replace(os.sep, "/"),
            format=fmt,
            width=w,
            height=h,
        )

    def get_path(self, rel: str) -> Path:
        """Return absolute filesystem path for a given relative 'style/...' path."""
        # Validate namespace using ReferenceMeta schema
        ReferenceMeta(rel=rel, format="PNG", width=1, height=1)
        abs_path = (self._static_root / rel.lstrip("/")).resolve()
        # Ensure the resolved path is still inside STYLE_DIR
        if not str(abs_path).startswith(str(STYLE_DIR)):
            raise ValueError("path escapes style directory")
        return abs_path

    def list(self) -> List[ReferenceMeta]:
        """
        List reference images from static/style.
        Reads minimal metadata (format via Pillow open, size via Pillow).
        """
        items: List[ReferenceMeta] = []
        for p in STYLE_DIR.glob("*"):
            if not p.is_file():
                continue
            ext = p.suffix.lower()
            if ext not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            try:
                with Image.open(p) as im:
                    fmt = (im.format or "").upper()
                    w, h = im.size
            except Exception:
                continue
            if fmt == "JPG":
                fmt = "JPEG"
            if fmt not in ALLOWED_FORMATS:
                continue
            try:
                rel_path = p.relative_to(self._static_root)
            except ValueError:
                rel_path = Path("style") / p.name
            items.append(
                ReferenceMeta(
                    rel=str(rel_path).replace(os.sep, "/"),
                    format=fmt,
                    width=int(w),
                    height=int(h),
                )
            )
        return items

    def remove(self, rel: str) -> bool:
        """Remove a stored reference by its relative path. Returns True if deleted."""
        try:
            target = self.get_path(rel)
        except Exception:
            return False
        if target.exists() and target.is_file():
            try:
                target.unlink()
                return True
            except Exception:
                return False
        return False

def build_prompt(
    prompt: str,
    negative_prompt: Optional[str],
    state: Union[StyleState, StyleConfig, None] = None,
) -> dict:

    """
    Build a unified payload for image backends.
    - Adds a gentle style hint to the positive prompt when style is enabled and a reference is present.
    - Returns a dict containing prompt, negative_prompt, and style metadata.

    Returns:
      {
        "prompt": str,
        "negative_prompt": Optional[str],
        "style": {"enabled": bool, "strength": float, "rel": Optional[str]}
      }
    """
    base_prompt = (prompt or "").strip()
    neg = (negative_prompt or "").strip() or None

    # Normalize state to StyleState
    if state is None:
        st = StyleState(enabled=False, strength=0.6, rel=None)
    elif isinstance(state, StyleConfig):
        st = StyleState(**state.model_dump())
    else:
        st = state

    final_prompt = base_prompt
    # Only add hint when reference is intended to be used
    if st.enabled and st.use_reference and st.rel:
        style_hint = f" Use the visual style from the provided reference image. Style strength: {st.strength:.2f}."
        final_prompt = f"{final_prompt}{style_hint}"

    return {
        "prompt": final_prompt,
        "negative_prompt": neg,
        "style": {
            "enabled": bool(st.enabled),
            "strength": float(st.strength),
            "rel": st.rel,
            "use_reference": bool(st.use_reference),
            "style_preset": st.style_preset,
        },
    }


# Global instances for convenient import in app.py
style_engine = StyleEngine()
reference_store = ReferenceStore()


if __name__ == "__main__":
    # Simple self-test to verify core functionality works in isolation
    print("Selftest style_engine...")
    se = StyleEngine()
    print("Initial:", se.get_state().model_dump())
    # Create a tiny red PNG in-memory
    buf = io.BytesIO()
    with Image.new("RGB", (1, 1), (255, 0, 0)) as im:
        im.save(buf, format="PNG")
    st = se.set_reference_file("test.png", buf.getvalue())
    print("After upload:", st.model_dump())
    st = se.update_params(True, 0.8, use_reference=True, style_preset="soft-illustration")
    print("After update:", st.model_dump())
    payload = build_prompt("A castle at sunset", "low quality, blurry", st)
    print("Payload:", payload)
    # Exercise ReferenceStore
    rs = ReferenceStore()
    meta = rs.save("another.png", buf.getvalue())
    print("Saved meta:", meta.model_dump())
    all_refs = rs.list()
    print("List:", [m.rel for m in all_refs])
    removed = rs.remove(meta.rel)
    print("Removed:", removed)
    st = se.clear_reference()
    print("After clear:", st.model_dump())
