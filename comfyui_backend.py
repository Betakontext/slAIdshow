# comfyui_backend.py
# English instructions and comments. German comments annotate complex logic succinctly.
# Asynchronous ComfyUI backends (local and remote) for slAIdshow.
from __future__ import annotations

import asyncio
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

# We only import httpx inside comfyui_bridge; this module itself stays lean.
# comfy_backend focuses on workflow preparation and bridge invocation.


# =========================
# Env + small helpers
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

def _debug() -> bool:
    return (_env_str("APP_IMAGE_BACKEND_DEBUG", "0").lower() in {"1","true","yes","on"})

def _app_root_dir() -> Path:
    return Path(_env_str("APP_OUTPUT_DIR", ".")).resolve()

def _outputs_images_dir() -> Path:
    # Contract: FastAPI mounts this dir at /static
    return (_app_root_dir() / "outputs" / "images").resolve()

def _ensure_outputs_dir() -> Path:
    p = _outputs_images_dir()
    p.mkdir(parents=True, exist_ok=True)
    return p

def _clamp8(v: int) -> int:
    # Deutsch: Viele Latent-Workflows erwarten Vielfache von 8
    v = max(64, min(4096, int(v)))
    return v - (v % 8)

def _resolve_size(req_w: Optional[int], req_h: Optional[int]) -> Tuple[int, int]:
    if isinstance(req_w, int) and req_w > 0 and isinstance(req_h, int) and req_h > 0:
        return _clamp8(req_w), _clamp8(req_h)
    w = _env_int("APP_COMFY_WIDTH", _env_int("APP_IMAGE_WIDTH", 512))
    h = _env_int("APP_COMFY_HEIGHT", _env_int("APP_IMAGE_HEIGHT", 512))
    return _clamp8(w), _clamp8(h)


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
# Comfy base + impls
# =========================

@dataclass
class _ComfyConfig:
    host: str = "127.0.0.1"
    port: int = 8188
    timeout_sec: float = 180.0
    reference_mode: str = "file"  # "file" | "url"

class _BaseComfyBackend:
    """
    Shared logic for Local/Remote Comfy backends.
    - Lazy import comfyui_bridge to avoid hard dependency at import time.
    - If no workflow provided in extra, build a minimal shell (production should pass a real workflow).
    """

    def __init__(self, cfg: _ComfyConfig) -> None:
        self._cfg = cfg

    async def close(self) -> None:
        return None

    def _minimal_workflow(self, *, prompt: str, negative_prompt: Optional[str], width: int, height: int) -> Dict[str, Any]:
        """
        Minimal placeholder workflow to remain schema-compatible with Comfy prompt dicts.
        Deutsch: In der Praxis gib ein echtes Workflow-Dict via extra['workflow'] mit.
        """
        return {
            "prompt": {
                "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt}},
                "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt or ""}},
                "4": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height}},
            }
        }

    def _apply_reference_url_mode(self, prompt_dict: Dict[str, Any], reference: Path) -> Dict[str, Any]:
        # Deutsch: URL-Modus injiziert signierte URL via Bridge-Helfer
        try:
            from comfyui_bridge import stage_reference_url_and_patch_prompt_sync
        except Exception as e:
            raise RuntimeError(f"reference_url_mode_not_available: {e}")
        return stage_reference_url_and_patch_prompt_sync(prompt_dict=prompt_dict, reference_local_path=reference)

    async def _generate_core(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str],
        width: int,
        height: int,
        style: Optional[Dict[str, Any]],
        reference: Optional[Path],
        extra: Optional[Dict[str, Any]],
    ) -> Path:
        # 1) Load/build workflow dict
        if extra and isinstance(extra.get("workflow"), dict):
            prompt_dict: Dict[str, Any] = extra["workflow"]
        else:
            prompt_dict = self._minimal_workflow(prompt=prompt, negative_prompt=negative_prompt, width=width, height=height)

        # 2) Inject positive/negative prompts into CLIP nodes (IDs 2/3, else heuristic scan)
        def _override_text_nodes(pd: dict, positive: str, negative: str) -> None:
            node_pos = pd.get("2")
            if isinstance(node_pos, dict) and node_pos.get("class_type") == "CLIPTextEncode":
                ins = node_pos.get("inputs")
                if isinstance(ins, dict) and "text" in ins:
                    ins["text"] = positive
            node_neg = pd.get("3")
            if isinstance(node_neg, dict) and node_neg.get("class_type") == "CLIPTextEncode":
                ins = node_neg.get("inputs")
                if isinstance(ins, dict) and "text" in ins:
                    ins["text"] = negative
            # Fallback scan
            clip_nodes = []
            for node in pd.values():
                if isinstance(node, dict) and node.get("class_type") == "CLIPTextEncode":
                    clip_nodes.append(node)
            if clip_nodes:
                ins = clip_nodes[0].get("inputs", {})
                if isinstance(ins, dict) and "text" in ins:
                    ins["text"] = positive
            if len(clip_nodes) > 1:
                ins = clip_nodes[1].get("inputs", {})
                if isinstance(ins, dict) and "text" in ins:
                    ins["text"] = negative

        pos = (prompt or "").strip()
        neg = (negative_prompt or "").strip()
        if isinstance(style, dict) and style.get("descriptors"):
            ds = [d for d in style["descriptors"] if isinstance(d, str) and d.strip()]
            if ds:
                pos = (pos.rstrip(",") + ", " + ", ".join(ds)).strip().strip(",")
        _override_text_nodes(prompt_dict, pos, neg)

        # 3) Override dimensions on common nodes
        def _override_dimensions(pd: dict, w: int, h: int) -> None:
            for node in pd.values():
                if not isinstance(node, dict):
                    continue
                cls = str(node.get("class_type") or node.get("class", "")).strip()
                ins = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
                if cls in {"EmptyLatentImage", "EmptyLatentImageBatch", "LatentImage", "CreateLatentImage"}:
                    if "width" in ins:
                        ins["width"] = w
                    if "height" in ins:
                        ins["height"] = h
                if cls.startswith("KSampler"):
                    if "width" in ins:
                        ins["width"] = w
                    if "height" in ins:
                        ins["height"] = h

        _override_dimensions(prompt_dict, width, height)

        # 4) Reference handling
        if reference and reference.exists():
            mode = (_env_str("APP_COMFY_REF_MODE", self._cfg.reference_mode) or "file").strip().lower()
            if mode == "url":
                try:
                    prompt_dict = self._apply_reference_url_mode(prompt_dict, reference)
                except Exception as e:
                    if _debug():
                        print(f"[COMFY][ref:url] failed -> keep file-mode. err={e}")
            else:
                # file-mode: assume the provided workflow already reads from a file path
                pass

        # 5) Dispatch to Comfy bridge
        try:
            from comfyui_bridge import generate_from_prompt_dict
        except Exception as e:
            raise RuntimeError(f"comfy_bridge_import_failed: {e}")

        out_dir = _ensure_outputs_dir()
        paths: List[Path] = await generate_from_prompt_dict(  # type: ignore[misc]
            prompt_dict=prompt_dict,
            out_dir=out_dir,
            host=self._cfg.host,
            port=self._cfg.port,
            max_wait_sec=self._cfg.timeout_sec,
        )
        if paths:
            p = Path(paths[0]).resolve()
            if p.exists() and p.is_file() and p.stat().st_size >= 1024:
                return p

        # 6) Optional fallback: copy from Comfy output dir if configured
        comfy_out = _env_str("APP_COMFY_OUTPUT_DIR", "")
        if comfy_out:
            base = Path(comfy_out).resolve()
            if base.exists():
                ts = time.time()
                candidates = sorted(
                    [p for p in base.glob("**/*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}],
                    key=lambda x: x.stat().st_mtime,
                    reverse=True,
                )
                for p in candidates[:6]:
                    if p.stat().st_mtime >= ts - 5 and p.stat().st_size >= 1024:
                        target = out_dir / f"img_{uuid.uuid4().hex}{p.suffix.lower()}"
                        try:
                            shutil.copy2(p, target)
                            return target
                        except Exception:
                            return p

        raise RuntimeError("comfy_generation_failed")

class LocalComfyBackend(_BaseComfyBackend, ImageBackend):
    """Local ComfyUI backend (127.0.0.1)."""
    def __init__(self) -> None:
        cfg = _ComfyConfig(
            host=_env_str("APP_COMFY_HOST", "127.0.0.1") or "127.0.0.1",
            port=_env_int("APP_COMFY_PORT", 8188),
            timeout_sec=_env_float("APP_COMFY_TIMEOUT_SEC", 180.0),
            reference_mode=_env_str("APP_COMFY_REF_MODE", "file") or "file",
        )
        super().__init__(cfg)

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
        w, h = _resolve_size(width, height)
        return await self._generate_core(
            prompt=prompt, negative_prompt=negative_prompt,
            width=w, height=h, style=style, reference=reference, extra=extra,
        )

class RemoteComfyBackend(_BaseComfyBackend, ImageBackend):
    """Remote ComfyUI backend (LAN/other network via VPN/WireGuard)."""
    def __init__(self, *, host: str, port: int) -> None:
        cfg = _ComfyConfig(
            host=host,
            port=port,
            timeout_sec=_env_float("APP_COMFY_TIMEOUT_SEC", 180.0),
            reference_mode=_env_str("APP_COMFY_REF_MODE", "file") or "file",
        )
        super().__init__(cfg)

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
        w, h = _resolve_size(width, height)
        return await self._generate_core(
            prompt=prompt, negative_prompt=negative_prompt,
            width=w, height=h, style=style, reference=reference, extra=extra,
        )
