# file: style_engine.py
from __future__ import annotations

import io
import json
import os
import re
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator

# Bildvalidierung
try:
    from PIL import Image
except Exception as e:
    raise RuntimeError("Pillow ist erforderlich: pip install Pillow") from e

# Zulässige Formate
ALLOWED_FORMATS: set[str] = {"PNG", "JPEG", "JPG", "WEBP"}

# Default-Verzeichnisse (können via ENV überschrieben werden)
STATIC_ROOT = Path(os.environ.get("STATIC_ROOT", "static")).resolve()
STYLE_DIR_STATIC = (STATIC_ROOT / "style").resolve()
STYLE_REFS_DIR_DEFAULT = Path(os.environ.get("APP_STYLE_REF_DIR", "./outputs/style_refs")).resolve()

# ComfyUI IP-Adapter/Referenz-Node IDs & Keys (über ENV konfigurierbar)
# Beispiel:
#   APP_COMFY_NODE_REF_IMAGE=23
#   APP_COMFY_NODE_IPADAPTER=45
#   APP_COMFY_KEY_REF_IMAGE_PATH=image
#   APP_COMFY_KEY_REF_WEIGHT=weight
REF_IMAGE_NODE_ID = os.environ.get("APP_COMFY_NODE_REF_IMAGE", "").strip()
IPADAPTER_NODE_ID = os.environ.get("APP_COMFY_NODE_IPADAPTER", "").strip()
REF_IMAGE_KEY = os.environ.get("APP_COMFY_KEY_REF_IMAGE_PATH", "image").strip()
REF_WEIGHT_KEY = os.environ.get("APP_COMFY_KEY_REF_WEIGHT", "weight").strip()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_filename(name: str) -> str:
    """Säubere Dateinamen gegen Traversal und ungültige Zeichen."""
    name = os.path.basename(name or "")
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name or f"file_{uuid.uuid4().hex[:8]}"


def detect_image_format(data: bytes) -> Tuple[bool, Optional[str], Optional[Tuple[int, int]]]:
    """
    Prüfe Bildbytes via Pillow. Gibt (ok, format, size) zurück.
    """
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
        with Image.open(io.BytesIO(data)) as img2:
            fmt = (img2.format or "").upper()
            size = img2.size
        return True, fmt, size
    except Exception:
        return False, None, None


class PromptBuildResult(BaseModel):
    positive: str
    negative: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class StyleState(BaseModel):
    # Historische Felder
    enabled: bool = Field(default=False)
    strength: float = Field(ge=0.0, le=1.0, default=0.6)
    rel: Optional[str] = None  # relativ unter /static (legacy)
    use_reference: bool = False
    style_preset: Optional[str] = None

    # Erweiterungen, wie von app.py erwartet
    style_details: str = Field(default="")
    negative_base: str = Field(default="")
    color_scheme: str = Field(default="")
    reference_id: Optional[str] = None
    reference_strength: float = Field(ge=0.0, le=1.0, default=0.6)

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
    Persistente UI-Konfiguration (1:1 wie die Felder, die app.py nutzt).
    """
    enabled: bool = Field(default=False)
    strength: float = Field(ge=0.0, le=1.0, default=0.6)
    rel: Optional[str] = None
    persisted_path: Optional[Path] = None
    style_preset: Optional[str] = None
    use_reference: bool = False

    style_details: str = Field(default="")
    negative_base: str = Field(default="")
    color_scheme: str = Field(default="")
    reference_id: Optional[str] = None
    reference_strength: float = Field(ge=0.0, le=1.0, default=0.6)

    @field_validator("rel")
    @classmethod
    def _validate_rel(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.lstrip("/")
        if ".." in v:
            raise ValueError("invalid relative path")
        return v


class StaticReferenceStore:
    """
    Legacy: Speicherung unter static/style mit Rückgabe eines relativen Pfads.
    """
    def __init__(self, static_root: Path | str = STATIC_ROOT) -> None:
        self._static_root = Path(static_root).resolve()
        ensure_dir(STYLE_DIR_STATIC)
        self._lock = threading.Lock()

    def save(self, filename: str, content: bytes) -> str:
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
        final = f"{base}{ext}"
        tmp = STYLE_DIR_STATIC / (final + ".tmp")
        dst = STYLE_DIR_STATIC / final
        with self._lock:
            tmp.write_bytes(content)
            tmp.replace(dst)
        try:
            rel = dst.relative_to(self._static_root)
        except ValueError:
            rel = Path("style") / final
        return str(rel).replace(os.sep, "/")


class ReferenceStore:
    """
    Store für Referenzbilder unter outputs/style_refs.
    Bietet ID-basierte Speicherung und Lookup.
    """
    def __init__(self, root: Path | str = STYLE_REFS_DIR_DEFAULT) -> None:
        self._root = Path(root).resolve()
        ensure_dir(self._root)
        self._lock = threading.Lock()

    def _id_to_path(self, rid: str) -> Path:
        fname = safe_filename(rid)
        return (self._root / fname).resolve()

    def put(self, filename: str, content: bytes) -> tuple[str, Path]:
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
        rid = f"{base}_{uuid.uuid4().hex[:8]}{ext}"
        tmp = self._root / (rid + ".tmp")
        dst = self._root / rid
        with self._lock:
            tmp.write_bytes(content)
            tmp.replace(dst)
        return rid, dst

    def get(self, rid: str) -> Optional[Path]:
        p = self._id_to_path(rid)
        if p.exists() and p.is_file():
            return p
        return None

    # Alias, falls bestehende Aufrufer get_path erwarten
    def get_path(self, rid: str) -> Optional[Path]:
        return self.get(rid)

    def remove(self, rid: str) -> bool:
        p = self._id_to_path(rid)
        try:
            if p.exists() and p.is_file():
                p.unlink()
                return True
        except Exception:
            return False
        return False

    def list(self) -> List[str]:
        return [f.name for f in self._root.iterdir() if f.is_file()]


def _compose_positive(base: str, sc: StyleConfig) -> str:
    """
    Positive Prompt setzt sich zusammen aus:
    - Benutzer- oder LLM-Text (base)
    - optionale style_preset/styledetails/farbvorgaben
    - deklarativer Hinweis auf Referenz-Style (Text-Hinweis für Prompt-Modelle ohne IP-Adapter)
    """
    base = (base or "").strip()
    parts: List[str] = [base]
    if sc.style_preset:
        parts.append(f"style preset: {sc.style_preset}")
    if sc.style_details:
        parts.append(sc.style_details)
    if sc.color_scheme:
        parts.append(f"color scheme: {sc.color_scheme}")
    if sc.use_reference and sc.reference_id:
        parts.append(f"visual style guided by reference image (strength {sc.reference_strength:.2f})")
    # Schlicht zusammenführen, kurz halten:
    return ", ".join([p for p in parts if p])


def _compose_negative(sc: StyleConfig) -> str:
    neg = (sc.negative_base or "").strip()
    return neg


def build_prompt(
    prompt: str,
    state: Union[StyleState, StyleConfig, None] = None,
) -> PromptBuildResult:
    """
    Baue positive/negative Prompts aus Nutzereingabe und StyleConfig.
    """
    if state is None:
        sc = StyleConfig()
    elif isinstance(state, StyleConfig):
        sc = state
    else:
        sc = StyleConfig(**state.model_dump())

    positive = _compose_positive((prompt or "").strip(), sc)
    negative = _compose_negative(sc)
    meta = {
        "use_reference": sc.use_reference,
        "reference_id": sc.reference_id,
        "reference_strength": sc.reference_strength,
        "style_preset": sc.style_preset,
    }
    return PromptBuildResult(positive=positive, negative=negative, meta=meta)


# ===== ComfyUI IP-Adapter Patching =====

def _parse_node_id(s: str) -> Optional[str]:
    s = (s or "").strip()
    if not s:
        return None
    # ComfyUI JSON nutzt meist numerische IDs als Strings
    return s


def apply_ip_adapter_to_workflow(
    workflow_payload: Dict[str, Any],
    reference_path: Path,
    reference_strength: float,
    *,
    ref_image_node_id: Optional[str] = None,
    ipadapter_node_id: Optional[str] = None,
    ref_image_key: Optional[str] = None,
    ref_weight_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Patches das bestehende ComfyUI-Workflow-Payload (dict), um:
    - den LoadImage-Node (ref_image_node_id) mit 'image': reference_path zu setzen
    - den IP-Adapter-Node (ipadapter_node_id) mit 'weight': reference_strength zu setzen

    Die Node IDs/Keys kommen standardmäßig aus ENV, können aber überschrieben werden.
    Gibt das modifizierte Payload zurück.
    """
    if not isinstance(workflow_payload, dict):
        return workflow_payload

    rid_ref = _parse_node_id(ref_image_node_id or REF_IMAGE_NODE_ID)
    rid_ip = _parse_node_id(ipadapter_node_id or IPADAPTER_NODE_ID)
    key_img = (ref_image_key or REF_IMAGE_KEY) or "image"
    key_w = (ref_weight_key or REF_WEIGHT_KEY) or "weight"

    if not rid_ref and not rid_ip:
        # Keine Konfiguration vorhanden → nichts zu patchen
        return workflow_payload

    # ComfyUI Workflow ist i.d.R. dict mit Node-ID als Schlüssel → Node-Objekt mit "inputs"
    nodes: Dict[str, Any] = {**workflow_payload}

    # Patch: Referenz-Image-Node
    if rid_ref and rid_ref in nodes:
        node = nodes.get(rid_ref)
        if isinstance(node, dict):
            inputs = node.setdefault("inputs", {})
            # Lokaler Pfad als String; ComfyUI erwartet string path
            inputs[key_img] = str(reference_path)

    # Patch: IP-Adapter-Node Gewicht
    if rid_ip and rid_ip in nodes:
        node = nodes.get(rid_ip)
        if isinstance(node, dict):
            inputs = node.setdefault("inputs", {})
            try:
                val = float(reference_strength)
            except Exception:
                val = 0.6
            inputs[key_w] = val

    return nodes


# ===== Backend-Bridge =====

def prepare_backend_style(
    backend: Any,
    style_cfg: StyleConfig,
    refs_dir: Path | str = STYLE_REFS_DIR_DEFAULT,
) -> None:
    """
    Stelle backend-seitig die Style-Referenz bereit.
    - Für LocalComfyBackend: setze StyleRuntime(reference_path, reference_strength) falls verfügbar.
    - Alternativ-Backends können diese Funktion ignorieren; der textuelle Stil bleibt erhalten.

    Diese Funktion darf gefahrlos mehrfach vor Generierung aufgerufen werden.
    """
    if backend is None:
        return
    # Lazy-Import, um harte Abhängigkeit auf image_backend zu vermeiden
    try:
        from image_backend import LocalComfyBackend, StyleRuntime  # type: ignore
    except Exception:
        LocalComfyBackend = None  # type: ignore
        StyleRuntime = None  # type: ignore

    if LocalComfyBackend is None or StyleRuntime is None:
        return
    if not isinstance(backend, LocalComfyBackend):
        return

    # Kein Referenzstil aktiv: StyleRuntime leeren (falls Backend das unterstützt)
    if not style_cfg.use_reference or not style_cfg.reference_id:
        try:
            backend.set_style_runtime(StyleRuntime(reference_path=None, reference_strength=style_cfg.reference_strength))  # type: ignore[attr-defined]
        except Exception:
            pass
        return

    store = ReferenceStore(refs_dir)
    p = store.get(style_cfg.reference_id)
    if p is None:
        # Falls Referenz nicht existiert → leeren
        try:
            backend.set_style_runtime(StyleRuntime(reference_path=None, reference_strength=style_cfg.reference_strength))  # type: ignore[attr-defined]
        except Exception:
            pass
        return

    # Lokale Referenz setzen
    try:
        backend.set_style_runtime(StyleRuntime(reference_path=p, reference_strength=style_cfg.reference_strength))  # type: ignore[attr-defined]
    except Exception:
        # Optional: Falls Backend stattdessen einen Workflow-Payload-Patcher exposed,
        # könnte man hier fallbacken. Standardmäßig genügt StyleRuntime.
        pass


# Bequeme Exporte
reference_store = ReferenceStore()

if __name__ == "__main__":
    # Minimaler Selbsttest
    buf = io.BytesIO()
    with Image.new("RGB", (4, 4), (120, 60, 200)) as im:
        im.save(buf, format="PNG")
    rs = ReferenceStore()
    rid, p = rs.put("ref.png", buf.getvalue())
    print("Saved:", rid, "->", p)
    sc = StyleConfig(
        style_preset="photo",
        style_details="soft bokeh, 35mm lens",
        color_scheme="warm tones",
        negative_base="text, watermark, logo, low quality",
        use_reference=True,
        reference_id=rid,
        reference_strength=0.65,
    )
    built = build_prompt("Ein roter Fuchs im Wald", sc)
    print("Positive:", built.positive)
    print("Negative:", built.negative)
