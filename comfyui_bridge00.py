# slAIDshow : comfyui_bridge.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable, Union
from urllib.parse import urlencode, quote

import httpx
from pydantic import BaseModel, Field

# -----------------------------
# Env helpers & host policy
# -----------------------------

def _env_str(k: str, d: str = "") -> str:
    return (os.getenv(k, d) or "").strip()

def _env_bool01(k: str, d: int = 0) -> bool:
    v = (os.getenv(k, str(d)) or "").strip().lower()
    return v in {"1", "true", "yes", "on"}

def _is_in_allowed_subnets(ip: str, subnets_str: str) -> bool:
    """Check if an IP is within any CIDR from a comma/space-separated allow-list."""
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
    """
    Privacy policy:
    - Always allow loopback.
    - Remote only if APP_ALLOW_REMOTE_BACKENDS=1 and optional subnets allow-list matches.
    """
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
        # Hostname without IP: cannot check; allow when explicit remote is enabled
        return
    if not _is_in_allowed_subnets(host, subnets):
        raise AssertionError(f"Remote host {host} not in allowed subnets ({subnets})")

# Optional filesystem fallback when /view is unavailable or unsuitable (mounted output dir).
_COMFY_OUTPUT_DIR: Optional[Path] = Path(_env_str("APP_COMFY_OUTPUT_DIR", "")).resolve() if _env_str("APP_COMFY_OUTPUT_DIR", "") else None

# Remote upload endpoint configuration (optional)
_COMFY_UPLOAD_ENDPOINT: str = _env_str("APP_COMFY_UPLOAD_ENDPOINT", "/upload/image")
_COMFY_INPUT_EXPECTS_OBJECT: bool = _env_bool01("APP_COMFY_INPUT_EXPECTS_OBJECT", 0)
# LoadImage node input key override (e.g., "image" or "image_upload")
_COMFY_REF_IMAGE_KEY: str = _env_str("APP_COMFY_KEY_REF_IMAGE_PATH", "image")
# IP-Adapter weight key
_COMFY_REF_WEIGHT_KEY: str = _env_str("APP_COMFY_KEY_REF_WEIGHT", "weight")
# Node-IDs (optional)
_COMFY_NODE_REF_IMAGE: str = _env_str("APP_COMFY_NODE_REF_IMAGE", "")
_COMFY_NODE_IPADAPTER: str = _env_str("APP_COMFY_NODE_IPADAPTER", "")

# Local input staging directory (default mirrors ComfyUI)
_COMFY_INPUT_DIR: Path = Path(_env_str("APP_COMFY_INPUT_DIR", "./ComfyUI/input")).resolve()

# URL-mode configuration for reference injection via URL loader node
_COMFY_NODE_REF_URL: str = _env_str("APP_COMFY_NODE_REF_URL", "")
_COMFY_KEY_REF_URL: str = _env_str("APP_COMFY_KEY_REF_URL", "url")

# Optional overrides for standard text/latent/ksampler nodes (if your workflow differs)
_POS_NODE_ID: str = _env_str("APP_COMFY_NODE_POSITIVE", "2")
_NEG_NODE_ID: str = _env_str("APP_COMFY_NODE_NEGATIVE", "3")
_LATENT_NODE_ID: str = _env_str("APP_COMFY_NODE_LATENT", "4")
_KSAMPLER_NODE_ID: str = _env_str("APP_COMFY_NODE_KSAMPLER", "5")

# -----------------------------
# Connection / helpers
# -----------------------------

class ComfyConnection(BaseModel):
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8188)

    @property
    def base(self) -> str:
        _assert_image_backend_host_policy(self.host)
        return f"http://{self.host}:{self.port}"

def _limits() -> httpx.Limits:
    return httpx.Limits(max_keepalive_connections=8, max_connections=16, keepalive_expiry=30.0)

def _timeout(total: float = 150.0) -> httpx.Timeout:
    total = max(30.0, min(total, 300.0))
    return httpx.Timeout(connect=5.0, read=total, write=8.0, pool=8.0)

def _clamp_dim(v: Optional[int]) -> Optional[int]:
    """Clamp dimension to [64, 2048] and multiple of 8."""
    if v is None:
        return None
    x = int(v)
    x = max(64, min(2048, x))
    return x - (x % 8)

def _select_view_mode(host: str) -> str:
    """
    Decide view mode: 'path' for local, 'query' for remote.
    Override via APP_COMFY_FORCE_VIEW_MODE in {'auto','path','query'}.
    """
    override = _env_str("APP_COMFY_FORCE_VIEW_MODE", "auto").lower()
    if override in {"path", "query"}:
        return override
    # auto
    if host in {"127.0.0.1", "localhost"}:
        return "path"
    return "query"

# -----------------------------
# Prompt manipulation
# -----------------------------

def _ensure_api_prompt_dict(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize payload to API format: {'prompt': {...}}.
    Accept either full payload or raw node mapping.
    """
    if isinstance(body, dict) and "prompt" in body and isinstance(body["prompt"], dict):
        return body
    if isinstance(body, dict) and body:
        if all(isinstance(k, str) and isinstance(v, dict) and "class_type" in v for k, v in body.items()):
            return {"prompt": body}
    raise RuntimeError("invalid_prompt_format: expected a 'prompt' dict or a node mapping")

def _get_node(prompt: Dict[str, Any], node_id: str) -> Optional[Dict[str, Any]]:
    n = prompt.get(node_id)
    return n if isinstance(n, dict) else None

def _set_input_if_present(node: Dict[str, Any], key: str, value: Any) -> None:
    ins = node.get("inputs")
    if isinstance(ins, dict) and key in ins:
        ins[key] = value

def override_prompt_inplace(
    body: Dict[str, Any],
    *,
    positive_text: Optional[str] = None,
    negative_text: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    steps: Optional[int] = None,
    cfg: Optional[float] = None,
    sampler_name: Optional[str] = None,
    scheduler: Optional[str] = None,
    denoise: Optional[float] = None,
    seed: Optional[int] = None,
    node_id_positive: str = _POS_NODE_ID,
    node_id_negative: str = _NEG_NODE_ID,
    node_id_latent: str = _LATENT_NODE_ID,
    node_id_ksampler: str = _KSAMPLER_NODE_ID,
) -> Dict[str, Any]:
    """
    Override standard text prompts and sampler/latent parameters in-place.
    Only keys present in the workflow are changed.
    """
    payload = _ensure_api_prompt_dict(body)
    prompt: Dict[str, Any] = payload["prompt"]

    if positive_text is not None:
        n2 = _get_node(prompt, node_id_positive)
        if n2:
            _set_input_if_present(n2, "text", positive_text)
    if negative_text is not None:
        n3 = _get_node(prompt, node_id_negative)
        if n3:
            _set_input_if_present(n3, "text", negative_text)

    w = _clamp_dim(width) if width is not None else None
    h = _clamp_dim(height) if height is not None else None
    if w is not None or h is not None:
        n4 = _get_node(prompt, node_id_latent)
        if n4:
            if w is not None:
                _set_input_if_present(n4, "width", int(w))
            if h is not None:
                _set_input_if_present(n4, "height", int(h))

    n5 = _get_node(prompt, node_id_ksampler)
    if n5:
        if steps is not None:
            _set_input_if_present(n5, "steps", int(steps))
        if cfg is not None:
            _set_input_if_present(n5, "cfg", float(cfg))
        if sampler_name is not None:
            _set_input_if_present(n5, "sampler_name", sampler_name)
        if scheduler is not None:
            _set_input_if_present(n5, "scheduler", scheduler)
        if denoise is not None:
            _set_input_if_present(n5, "denoise", float(denoise))
        if seed is not None:
            _set_input_if_present(n5, "seed", int(seed))

    return payload

# -----------------------------
# Remote/local reference staging (FILE / UPLOAD)
# -----------------------------

async def _upload_reference_to_remote_comfy(
    client: httpx.AsyncClient,
    base_url: str,
    local_path: Path,
    *,
    upload_endpoint: str = _COMFY_UPLOAD_ENDPOINT,
    timeout_s: float = 20.0,
) -> Optional[str]:
    """
    Upload a file via HTTP to the Comfy host and return the filename
    that a LoadImage node can reference (usually the basename).
    Requires a custom endpoint on the Comfy host writing into Comfy input dir.
    """
    try:
        if not local_path.exists() or not local_path.is_file():
            return None
        url = f"{base_url}{upload_endpoint}"
        filename = local_path.name
        files = {"file": (filename, local_path.open("rb"), "application/octet-stream")}
        r = await client.post(url, files=files, timeout=timeout_s)
        if 200 <= r.status_code < 300:
            try:
                data = r.json()
                # Common keys: filename, name; fallback: basename
                return data.get("filename") or data.get("name") or filename
            except Exception:
                return filename
        else:
            print(f"DEBUG comfy upload failed: status={r.status_code} text={r.text[:300]}")
            return None
    except Exception as e:
        print(f"DEBUG comfy upload exception: {e}")
        return None

def _stage_reference_into_local_input(local_path: Path, input_dir: Path = _COMFY_INPUT_DIR) -> Optional[str]:
    """
    Copy the referenced file into ComfyUI/input (if necessary) and
    return the basename for the LoadImage node.
    """
    try:
        if not local_path.exists() or not local_path.is_file():
            return None
        input_dir.mkdir(parents=True, exist_ok=True)
        target = input_dir / local_path.name
        try:
            # Overwrite only if source is newer or target missing
            if (not target.exists()) or (local_path.stat().st_mtime > target.stat().st_mtime):
                data = local_path.read_bytes()
                if len(data) < 10:
                    # Minimal sanity-check to avoid creating junk files
                    return None
                target.write_bytes(data)
        except Exception:
            # Fallback: copy2
            import shutil
            shutil.copy2(str(local_path), str(target))
        return local_path.name
    except Exception as e:
        print(f"DEBUG comfy local stage exception: {e}")
        return None

def _expected_loadimage_value(image_key_value: Any, filename_only: str) -> Union[str, Dict[str, Any]]:
    """
    Some workflows set inputs[image] as a string; others as an object {"image":"..."}.
    Preserve the original type while replacing the filename.
    """
    if isinstance(image_key_value, dict):
        d = dict(image_key_value)
        d["image"] = filename_only
        return d
    return filename_only

def override_reference_inplace(
    body: Dict[str, Any],
    *,
    reference_filename: str,
    ref_image_node_id: Optional[str] = None,
    ipadapter_node_id: Optional[str] = None,
    reference_strength: Optional[float] = None,
    ref_image_key: Optional[str] = None,
    ref_weight_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Patch the prompt dict:
      - At the LoadImage node (ref_image_node_id) set the image name (basename only).
      - At the IP-Adapter node (ipadapter_node_id) set the weight.
    Missing nodes/keys are ignored gracefully.
    """
    payload = _ensure_api_prompt_dict(body)
    prompt: Dict[str, Any] = payload["prompt"]

    rid_ref = (ref_image_node_id or _COMFY_NODE_REF_IMAGE or "").strip()
    rid_ip = (ipadapter_node_id or _COMFY_NODE_IPADAPTER or "").strip()
    key_img = (ref_image_key or _COMFY_REF_IMAGE_KEY or "image")
    key_w = (ref_weight_key or _COMFY_REF_WEIGHT_KEY or "weight")

    if rid_ref:
        n = _get_node(prompt, rid_ref)
        if n and isinstance(n.get("inputs"), dict):
            ins = n["inputs"]
            old_val = ins.get(key_img)
            ins[key_img] = _expected_loadimage_value(old_val, reference_filename)

    if rid_ip and reference_strength is not None:
        n = _get_node(prompt, rid_ip)
        if n and isinstance(n.get("inputs"), dict):
            try:
                val = float(reference_strength)
            except Exception:
                val = 0.6
            n["inputs"][key_w] = val

    return payload

# -----------------------------
# URL-mode reference injection
# -----------------------------

def _build_signed_url_for_basename(basename: str, ttl_sec: Optional[int] = None) -> str:
    """
    Build a signed URL for a given basename using app.build_signed_url.
    Lazy-import to avoid hard dependency when not available.
    """
    try:
        from app import build_signed_url  # provided by the app layer
    except Exception as e:
        raise RuntimeError(f"build_signed_url_unavailable: {e}")
    return build_signed_url(basename, ttl_sec) if ttl_sec is not None else build_signed_url(basename)

def inject_reference_url_inplace(
    body: Dict[str, Any],
    *,
    url: str,
    ref_url_node_id: Optional[str] = None,
    ref_url_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Set URL value on a URL-loader node (e.g., ImageFromURL).
    If node/key is not found by ID, falls back to first node having the given key.
    """
    payload = _ensure_api_prompt_dict(body)
    prompt: Dict[str, Any] = payload["prompt"]

    rid = (ref_url_node_id or _COMFY_NODE_REF_URL or "").strip()
    key = (ref_url_key or _COMFY_KEY_REF_URL or "url")

    if rid:
        n = _get_node(prompt, rid)
        if n and isinstance(n.get("inputs"), dict):
            if key in n["inputs"]:
                n["inputs"][key] = url
                return payload

    # Heuristic fallback: first node with matching input key
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        ins = node.get("inputs")
        if isinstance(ins, dict) and (key in ins):
            ins[key] = url
            return payload

    return payload

def stage_reference_url_and_patch_prompt(
    prompt_dict: Dict[str, Any],
    reference_local_path: Path,
    *,
    ref_url_node_id: Optional[str] = None,
    ref_url_key: Optional[str] = None,
    ttl_sec: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build a signed URL from the local reference basename and inject it into the prompt (URL node).
    Assumes the app serves the file under /ref/<basename>.
    """
    if not reference_local_path.exists() or not reference_local_path.is_file():
        raise FileNotFoundError(f"reference not found: {reference_local_path}")
    basename = reference_local_path.name
    signed = _build_signed_url_for_basename(basename, ttl_sec=ttl_sec)
    return inject_reference_url_inplace(
        prompt_dict,
        url=signed,
        ref_url_node_id=ref_url_node_id,
        ref_url_key=ref_url_key,
    )

# -----------------------------
# HTTP calls
# -----------------------------

async def _post_prompt(
    client: httpx.AsyncClient,
    base_url: str,
    body: Dict[str, Any],
) -> str:
    """
    POST /prompt with retries/backoff and extract prompt_id.
    Raises descriptive errors on 400 or missing id.
    """
    delay = 0.8
    last_exc: Optional[Exception] = None
    for attempt in range(1, 5):
        try:
            r = await client.post(f"{base_url}/prompt", json=body)
            if r.status_code == 400:
                text = r.text[:300].replace("\n", " ")
                raise RuntimeError(f"comfy_400: {text}")
            r.raise_for_status()
            j = r.json()
            if not isinstance(j, dict):
                raise RuntimeError(f"comfy_prompt_non_object: {type(j)}")
            pid = j.get("prompt_id") or j.get("promptId") or j.get("id")
            if not pid:
                raise RuntimeError("comfy_no_prompt_id")
            print(f"DEBUG comfy prompt_id: {pid}")
            return str(pid)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError, httpx.HTTPStatusError) as e:
            last_exc = e
            if attempt < 4:
                await asyncio.sleep(delay)
                delay *= 1.7
                continue
            break
        except Exception:
            raise
    raise RuntimeError(f"comfy_post_prompt_failed: {last_exc}")

def _node_maps_from_history_obj(history_json: Dict[str, Any], prompt_id: str) -> List[Dict[str, Any]]:
    """
    Comfy history payloads vary. Collect plausible node maps:
    - full object (nodeId->node)
    - entry for given prompt_id
    - entry.outputs
    """
    node_maps: List[Dict[str, Any]] = []
    if isinstance(history_json, dict) and all(isinstance(v, dict) for v in history_json.values()):
        node_maps.append(history_json)
    entry = history_json.get(prompt_id)
    if isinstance(entry, dict) and all(isinstance(v, dict) for v in entry.values()):
        node_maps.append(entry)
    if isinstance(entry, dict):
        outputs = entry.get("outputs")
        if isinstance(outputs, dict) and all(isinstance(v, dict) for v in outputs.values()):
            node_maps.append(outputs)
    seen_ids = set()
    deduped: List[Dict[str, Any]] = []
    for m in node_maps:
        if id(m) in seen_ids:
            continue
        seen_ids.add(id(m))
        deduped.append(m)
    return deduped

def _any_images_in_node_maps(node_maps: Iterable[Dict[str, Any]]) -> bool:
    """Return True if any collected node map contains image descriptors."""
    for node_map in node_maps:
        for _, node in node_map.items():
            if not isinstance(node, dict):
                continue
            outs = node.get("outputs")
            if not isinstance(outs, dict):
                continue
            for outv in outs.values():
                if isinstance(outv, dict) and isinstance(outv.get("images"), list) and outv["images"]:
                    return True
    return False

async def _poll_history_ready(
    client: httpx.AsyncClient,
    base_url: str,
    prompt_id: str,
    max_wait_sec: float = 150.0,
    poll_interval: float = 1.0,
) -> Dict[str, Any]:
    """
    Poll /history/{prompt_id} until at least one image appears or timeout elapses.
    Returns the final history JSON for the prompt.
    """
    deadline = time.time() + max_wait_sec
    last_payload: Optional[Dict[str, Any]] = None
    while time.time() < deadline:
        r = await client.get(f"{base_url}/history/{prompt_id}")
        if r.status_code == 404:
            await asyncio.sleep(poll_interval)
            continue
        r.raise_for_status()
        j = r.json()
        if not isinstance(j, dict):
            await asyncio.sleep(poll_interval)
            continue
        last_payload = j
        maps = _node_maps_from_history_obj(j, prompt_id)
        if maps and _any_images_in_node_maps(maps):
            return j
        await asyncio.sleep(poll_interval)
    # On timeout, dump limited diagnostics and raise
    try:
        keys = list(last_payload.keys()) if isinstance(last_payload, dict) else "n/a"
        print("DEBUG comfy history keys:", keys)
        sample = json.dumps(last_payload or {}, ensure_ascii=False)[:2000]
        print("DEBUG comfy history sample:", sample)
    except Exception:
        pass
    raise TimeoutError("comfy_history_timeout")

def _iter_image_descriptors_from_history(history_obj: Dict[str, Any], prompt_id: str) -> Iterable[Dict[str, Any]]:
    """
    Iterate over image descriptor dicts (with 'images' list) from history object.
    """
    maps = _node_maps_from_history_obj(history_obj, prompt_id)
    for node_map in maps:
        for _, node in node_map.items():
            if not isinstance(node, dict):
                continue
            outs = node.get("outputs")
            if not isinstance(outs, dict):
                continue
            for _, outv in outs.items():
                if isinstance(outv, dict) and isinstance(outv.get("images"), list):
                    yield outv

def _build_view_candidates(folder_type: str, subfolder: str, filename: str) -> List[Tuple[str, str, str]]:
    """
    Build alternative (type, subfolder, filename) triples to try.
    Used by both path and query modes to handle varying history descriptors.
    """
    sub = (subfolder or "").strip().strip("/")
    candidates: List[Tuple[str, str, str]] = []
    candidates.append((folder_type or "output", sub, filename))
    if sub:
        candidates.append((folder_type or "output", "", filename))
    candidates.append(("temp", sub, filename))
    if sub:
        candidates.append(("temp", "", filename))
    candidates.append(("output", "", filename))
    seen = set()
    uniq: List[Tuple[str, str, str]] = []
    for t, s, f in candidates:
        key = (t, s, f)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((t, s, f))
    return uniq

def _is_likely_image(data: bytes) -> bool:
    """
    Fast signature check for PNG/JPEG/WEBP and a minimal length guard.
    Prevents accepting non-image HTML/errors and avoids rejecting tiny but valid files.
    """
    if not data or len(data) < 64:
        return False
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    if data.startswith(b"\xff\xd8\xff"):
        return True
    if data.startswith(b"RIFF") and b"WEBP" in data[:16]:
        return True
    # Fallback minimal size guard
    return len(data) >= 256

async def _download_via_view_path(
    client: httpx.AsyncClient,
    base_url: str,
    folder_type: str,
    subfolder: str,
    filename: str,
) -> bytes:
    """
    Download via legacy path style: /view/{type}/{subfolder}/{filename}
    Try several candidates; URL-encode segments.
    """
    filename = filename.strip().lstrip("/")
    candidates = _build_view_candidates(folder_type, subfolder, filename)
    last_exc: Optional[Exception] = None
    for idx, (t, s, f) in enumerate(candidates, start=1):
        seg_t = quote((t or "").strip(), safe="")
        seg_s = quote((s or "").strip(), safe="") if s else ""
        seg_f = quote((f or "").strip(), safe="")
        path = f"/view/{seg_t}"
        if seg_s:
            path += f"/{seg_s}"
        path += f"/{seg_f}"
        url = f"{base_url}{path}"
        delay = 0.2
        for attempt in range(1, 3):
            try:
                print(f"DEBUG comfy GET[path] try#{attempt} cand#{idx}: {url}")
                r = await client.get(url)
                print(f"DEBUG comfy GET[path] status={r.status_code} bytes={len(r.content) if r.content else 0}")
                r.raise_for_status()
                data = r.content or b""
                if not _is_likely_image(data):
                    raise RuntimeError("image_download_not_recognized")
                return data
            except Exception as e:
                last_exc = e
                if attempt < 2:
                    await asyncio.sleep(delay)
                    delay *= 2.0
                else:
                    break
    if last_exc:
        raise last_exc
    raise RuntimeError("image_download_failed_unknown_path")

async def _download_via_view_query(
    client: httpx.AsyncClient,
    base_url: str,
    folder_type: str,
    subfolder: str,
    filename: str,
) -> bytes:
    """
    Download via query style: /api/view?filename=...&type=...&subfolder=...
    Try several (type, subfolder, filename) candidates; parameters are URL-encoded.
    """
    filename = (filename or "").strip().lstrip("/")
    candidates = _build_view_candidates(folder_type, subfolder, filename)
    last_exc: Optional[Exception] = None
    for idx, (t, s, f) in enumerate(candidates, start=1):
        params = {
            "filename": f,
            "type": (t or "output"),
            "subfolder": (s or ""),
        }
        qs = urlencode(params, safe="/")
        url = f"{base_url}/api/view?{qs}"
        delay = 0.2
        for attempt in range(1, 3):
            try:
                print(f"DEBUG comfy GET[query] try#{attempt} cand#{idx}: {url}")
                r = await client.get(url)
                print(f"DEBUG comfy GET[query] status={r.status_code} bytes={len(r.content) if r.content else 0}")
                r.raise_for_status()
                data = r.content or b""
                if not _is_likely_image(data):
                    raise RuntimeError("image_download_not_recognized")
                return data
            except Exception as e:
                last_exc = e
                if attempt < 2:
                    await asyncio.sleep(delay)
                    delay *= 2.0
                else:
                    break
    if last_exc:
        raise last_exc
    raise RuntimeError("image_download_failed_unknown_query")

def _fs_fallback_read(folder_type: str, subfolder: str, filename: str) -> Optional[bytes]:
    """
    Filesystem fallback: try reading from APP_COMFY_OUTPUT_DIR mirroring Comfy's output structure.
    """
    if _COMFY_OUTPUT_DIR is None:
        return None
    base = _COMFY_OUTPUT_DIR
    sub = (subfolder or "").strip().strip("/")
    if sub:
        base = base / sub
    f = base / filename
    try:
        if f.is_file():
            return f.read_bytes()
    except Exception:
        return None
    return None

async def _download_images(
    client: httpx.AsyncClient,
    base_url: str,
    history_obj: Dict[str, Any],
    prompt_id: str,
    out_dir: Path,
    view_mode: str,
) -> List[Path]:
    """
    Download images from history using either 'path' or 'query' mode, with FS fallback.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    found_any = False

    for outv in _iter_image_descriptors_from_history(history_obj, prompt_id):
        images = outv.get("images", [])
        if not isinstance(images, list) or not images:
            continue
        found_any = True
        for item in images:
            if not isinstance(item, dict):
                continue
            filename = (item.get("filename") or "").strip()
            subfolder = item.get("subfolder", "")
            folder_type = item.get("type", "output")
            if not filename:
                continue

            delay = 0.8
            data: Optional[bytes] = None
            for attempt in range(1, 3):
                try:
                    if view_mode == "query":
                        data = await _download_via_view_query(client, base_url, folder_type, subfolder, filename)
                    else:
                        data = await _download_via_view_path(client, base_url, folder_type, subfolder, filename)
                    break
                except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError, httpx.HTTPStatusError):
                    if attempt < 2:
                        await asyncio.sleep(delay)
                        delay *= 1.7
                        continue
                except Exception:
                    # Fall through to FS fallback
                    pass

            if data is None:
                data = _fs_fallback_read(folder_type, subfolder, filename)

            if data is None:
                print(f"DEBUG comfy: skip missing image filename='{filename}' type='{folder_type}' subfolder='{subfolder}'")
                continue

            suffix = Path(filename).suffix or ".png"
            target = out_dir / f"img_{uuid.uuid4().hex}{suffix}"
            target.write_bytes(data)
            saved.append(target)

    if not found_any:
        print("DEBUG comfy: no images found in history object (iter yielded none).")

    return saved

# -----------------------------
# Public entry points
# -----------------------------

async def stage_reference_on_remote_and_patch_prompt(
    prompt_dict: Dict[str, Any],
    reference_local_path: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8188,
    ref_image_node_id: Optional[str] = None,
    ipadapter_node_id: Optional[str] = None,
    reference_strength: Optional[float] = None,
    ref_image_key: Optional[str] = None,
    ref_weight_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Reference staging and workflow patching:
      1) On localhost: stage the file into APP_COMFY_INPUT_DIR and use basename.
      2) On remote host: upload via HTTP to a custom endpoint (if available).
      3) Patch the prompt dict (LoadImage basename, optional IP-Adapter weight).
    """
    conn = ComfyConnection(host=host, port=port)
    base = conn.base
    payload = _ensure_api_prompt_dict(prompt_dict)

    filename_only: Optional[str] = None
    async with httpx.AsyncClient(limits=_limits(), timeout=_timeout(60)) as client:
        if host in {"127.0.0.1", "localhost"}:
            filename_only = _stage_reference_into_local_input(reference_local_path, _COMFY_INPUT_DIR)
            if not filename_only:
                # Basename fallback; Comfy may fail if file truly missing
                filename_only = reference_local_path.name
                print(f"DEBUG comfy: local stage missing, using basename fallback: {filename_only}")
        else:
            filename_only = await _upload_reference_to_remote_comfy(client, base, reference_local_path)

    if not filename_only:
        # Ultimate fallback: basename only
        filename_only = reference_local_path.name
        print(f"DEBUG comfy: remote upload missing, using basename fallback: {filename_only}")

    return override_reference_inplace(
        payload,
        reference_filename=filename_only,
        ref_image_node_id=ref_image_node_id,
        ipadapter_node_id=ipadapter_node_id,
        reference_strength=reference_strength,
        ref_image_key=ref_image_key,
        ref_weight_key=ref_weight_key,
    )

async def generate_from_prompt_dict(
    prompt_dict: Dict[str, Any],
    out_dir: Path,
    *,
    positive_text: Optional[str] = None,
    negative_text: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    steps: Optional[int] = None,
    cfg: Optional[float] = None,
    sampler_name: Optional[str] = None,
    scheduler: Optional[str] = None,
    denoise: Optional[float] = None,
    seed: Optional[int] = None,
    host: str = "127.0.0.1",
    port: int = 8188,
    max_wait_sec: float = 150.0,
    poll_interval: float = 1.0,
) -> List[Path]:
    """
    Submit a ComfyUI API prompt dict, poll history, and download images to out_dir.
    Selects view mode automatically (path for local, query for remote) or via env override.
    """
    payload = override_prompt_inplace(
        body=prompt_dict,
        positive_text=positive_text,
        negative_text=negative_text,
        width=width,
        height=height,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler=scheduler,
        denoise=denoise,
        seed=seed,
    )

    conn = ComfyConnection(host=host, port=port)
    base = conn.base
    view_mode = _select_view_mode(conn.host)
    print(f"[COMFY VIEW MODE] {view_mode} (host={conn.host})")

    async with httpx.AsyncClient(limits=_limits(), timeout=_timeout(max_wait_sec + 30)) as client:
        prompt_id = await _post_prompt(client, base, payload)
        history_obj = await _poll_history_ready(client, base, prompt_id, max_wait_sec=max_wait_sec, poll_interval=poll_interval)
        images = await _download_images(client, base, history_obj, prompt_id, out_dir=Path(out_dir), view_mode=view_mode)
    return images
