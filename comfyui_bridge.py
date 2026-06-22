# slAIDshow : comfui_bridge.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComfyUI bridge with LAN/remote support.

Features:
- Accepts ComfyUI running on localhost, LAN IP, or remote/VPN host.
- Validates remote usage against env policy (APP_ALLOW_REMOTE_BACKENDS and APP_ALLOWED_SUBNETS).
- Submits prompt dicts to /prompt, polls /history/{id}, downloads images via /api/view or /view.
- Works with URL reference mode and file-based reference mode (helpers included).
- Backward-compatible: generate_from_prompt_dict(..., prompt=...) legacy kw accepted.

Environment of interest (see your .env):
- APP_COMFY_HOST, APP_COMFY_PORT
- APP_ALLOW_REMOTE_BACKENDS=1 to permit non-local hosts
- APP_ALLOWED_SUBNETS=192.168.188.0/24 to restrict remotes
- APP_COMFY_OUTPUT_DIR for FS fallback reads
- APP_COMFY_INPUT_DIR for local staging of reference images
- APP_REF_TTL_SEC + app.build_signed_url() when using URL-reference workflows
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import socket
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from urllib.parse import urlencode, quote

import httpx
from pydantic import BaseModel, Field

# =========================
# Env helpers and policy
# =========================

def _env_str(k: str, d: str = "") -> str:
    return (os.getenv(k, d) or "").strip()

def _env_bool01(k: str, d: int = 0) -> bool:
    v = (os.getenv(k, str(d)) or "").strip().lower()
    return v in {"1", "true", "yes", "on"}

def _resolve_host_to_ips(host: str) -> List[str]:
    """Resolve hostname to IPs; fall back to host itself on failure."""
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
        ips = []
        for _, _, _, _, sockaddr in infos:
            ip = sockaddr[0]
            if ip not in ips:
                ips.append(ip)
        return ips or [host]
    except Exception:
        return [host]

def _is_in_allowed_subnets(ip: str, subnets_str: str) -> bool:
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
    - Always allow localhost.
    - Remote hosts allowed only if APP_ALLOW_REMOTE_BACKENDS=1.
    - If APP_ALLOWED_SUBNETS is set, ensure host IP is in one of those subnets.
    - If no subnets set, allow any remote when APP_ALLOW_REMOTE_BACKENDS=1.
    """
    if host in {"127.0.0.1", "localhost"}:
        return
    allow_remote = _env_bool01("APP_ALLOW_REMOTE_BACKENDS", 0)
    if not allow_remote:
        raise AssertionError("Remote backends disabled; set APP_ALLOW_REMOTE_BACKENDS=1 to permit.")
    subnets = _env_str("APP_ALLOWED_SUBNETS", "")
    if not subnets:
        return
    for ip in _resolve_host_to_ips(host):
        try:
            ipaddress.ip_address(ip)
        except ValueError:
            continue
        if _is_in_allowed_subnets(ip, subnets):
            return
    raise AssertionError(f"Host {host} not within allowed subnets: {subnets}")

_COMFY_OUTPUT_DIR: Optional[Path] = Path(_env_str("APP_COMFY_OUTPUT_DIR", "")).resolve() if _env_str("APP_COMFY_OUTPUT_DIR", "") else None
_COMFY_INPUT_DIR: Path = Path(_env_str("APP_COMFY_INPUT_DIR", "./ComfyUI/input")).resolve()

# Optional remote upload endpoint if you have one on the Comfy host
_COMFY_UPLOAD_ENDPOINT: str = _env_str("APP_COMFY_UPLOAD_ENDPOINT", "/upload/image")

# Workflow node ids/keys (env overrides)
_POS_NODE_ID: str = _env_str("APP_COMFY_NODE_POS", "2")
_NEG_NODE_ID: str = _env_str("APP_COMFY_NODE_NEG", "3")
_LATENT_NODE_ID: str = _env_str("APP_COMFY_NODE_LATENT", "4")
_KSAMPLER_NODE_ID: str = _env_str("APP_COMFY_NODE_KSAMPLER", "5")

_COMFY_NODE_REF_IMAGE: str = _env_str("APP_COMFY_NODE_REF_IMAGE", "")
_COMFY_REF_IMAGE_KEY: str = _env_str("APP_COMFY_KEY_REF_IMAGE_PATH", "image")

_COMFY_NODE_IPADAPTER: str = _env_str("APP_COMFY_NODE_IPADAPTER", "")
_COMFY_REF_WEIGHT_KEY: str = _env_str("APP_COMFY_KEY_REF_WEIGHT", "weight")

_COMFY_NODE_REF_URL: str = _env_str("APP_COMFY_NODE_REF_URL", "")
_COMFY_KEY_REF_URL: str = _env_str("APP_COMFY_KEY_REF_URL", "url")

_APP_REF_TTL_SEC: Optional[int] = int(_env_str("APP_REF_TTL_SEC", "0") or "0") or None

# =========================
# Connection / httpx
# =========================

class ComfyConnection(BaseModel):
    host: str = Field(default=_env_str("APP_COMFY_HOST", "127.0.0.1"))
    port: int = Field(default=int(_env_str("APP_COMFY_PORT", "8188") or 8188))

    @property
    def base(self) -> str:
        _assert_image_backend_host_policy(self.host)
        return f"http://{self.host}:{self.port}"

def _limits() -> httpx.Limits:
    return httpx.Limits(max_keepalive_connections=8, max_connections=16, keepalive_expiry=30.0)

def _timeout(total: float = 150.0) -> httpx.Timeout:
    total = max(30.0, min(total, 600.0))
    return httpx.Timeout(connect=6.0, read=total, write=15.0, pool=8.0)

def _clamp_dim(v: Optional[int]) -> Optional[int]:
    if v is None:
        return None
    x = max(64, min(2048, int(v)))
    return x - (x % 8)

def _select_view_mode(host: str) -> str:
    """Prefer 'path' for localhost, 'query' for remote; allow override via APP_COMFY_FORCE_VIEW_MODE."""
    override = _env_str("APP_COMFY_FORCE_VIEW_MODE", "auto").lower()
    if override in {"path", "query"}:
        return override
    return "path" if host in {"127.0.0.1", "localhost"} else "query"

# =========================
# Prompt manipulation
# =========================

def _ensure_api_prompt_dict(body: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize payload to {'prompt': {...}} for Comfy /prompt API."""
    if isinstance(body, dict) and "prompt" in body and isinstance(body["prompt"], dict):
        return body
    if isinstance(body, dict) and body:
        if all(isinstance(k, str) and isinstance(v, dict) and "class_type" in v for k, v in body.items()):
            return {"prompt": body}
    raise RuntimeError("Invalid prompt format; expected a 'prompt' map or {'prompt': {...}}")

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
    """Override text/latent/sampler params in-place if keys exist in the workflow."""
    payload = _ensure_api_prompt_dict(body)
    prompt: Dict[str, Any] = payload["prompt"]

    if positive_text is not None:
        n = _get_node(prompt, node_id_positive)
        if n:
            _set_input_if_present(n, "text", positive_text)
    if negative_text is not None:
        n = _get_node(prompt, node_id_negative)
        if n:
            _set_input_if_present(n, "text", negative_text)

    w = _clamp_dim(width) if width is not None else None
    h = _clamp_dim(height) if height is not None else None
    if w is not None or h is not None:
        n = _get_node(prompt, node_id_latent)
        if n:
            if w is not None:
                _set_input_if_present(n, "width", int(w))
            if h is not None:
                _set_input_if_present(n, "height", int(h))

    n = _get_node(prompt, node_id_ksampler)
    if n:
        if steps is not None:
            _set_input_if_present(n, "steps", int(steps))
        if cfg is not None:
            _set_input_if_present(n, "cfg", float(cfg))
        if sampler_name is not None:
            _set_input_if_present(n, "sampler_name", sampler_name)
        if scheduler is not None:
            _set_input_if_present(n, "scheduler", scheduler)
        if denoise is not None:
            _set_input_if_present(n, "denoise", float(denoise))
        if seed is not None:
            _set_input_if_present(n, "seed", int(seed))

    return payload

# =========================
# Reference helpers
# =========================

def _stage_reference_into_local_input(local_path: Path, input_dir: Path = _COMFY_INPUT_DIR) -> Optional[str]:
    """Copy reference image to ComfyUI/input and return basename for LoadImage node."""
    try:
        if not local_path.exists() or not local_path.is_file():
            return None
        input_dir.mkdir(parents=True, exist_ok=True)
        target = input_dir / local_path.name
        data = local_path.read_bytes()
        if len(data) < 16:
            return None
        # Write/overwrite if needed
        if (not target.exists()) or (local_path.stat().st_mtime > target.stat().st_mtime):
            target.write_bytes(data)
        return local_path.name
    except Exception as e:
        print(f"DEBUG local reference staging failed: {e}")
        return None

async def _upload_reference_to_remote_comfy(
    client: httpx.AsyncClient,
    base_url: str,
    local_path: Path,
    *,
    upload_endpoint: str = _COMFY_UPLOAD_ENDPOINT,
) -> Optional[str]:
    """Upload reference via custom remote endpoint; return basename for LoadImage node."""
    try:
        if not local_path.exists() or not local_path.is_file():
            return None
        url = f"{base_url}{upload_endpoint}"
        files = {"file": (local_path.name, local_path.open("rb"), "application/octet-stream")}
        r = await client.post(url, files=files)
        if 200 <= r.status_code < 300:
            try:
                j = r.json()
                return j.get("filename") or j.get("name") or local_path.name
            except Exception:
                return local_path.name
        print(f"DEBUG remote reference upload failed: {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"DEBUG remote reference upload exception: {e}")
    return None

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
    """Patch LoadImage and optional IP-Adapter nodes with filename and weight."""
    payload = _ensure_api_prompt_dict(body)
    prompt: Dict[str, Any] = payload["prompt"]

    rid_ref = (ref_image_node_id or _COMFY_NODE_REF_IMAGE or "").strip()
    rid_ip = (ipadapter_node_id or _COMFY_NODE_IPADAPTER or "").strip()
    key_img = (ref_image_key or _COMFY_REF_IMAGE_KEY or "image")
    key_w = (ref_weight_key or _COMFY_REF_WEIGHT_KEY or "weight")

    if rid_ref:
        n = _get_node(prompt, rid_ref)
        if n and isinstance(n.get("inputs"), dict):
            old_val = n["inputs"].get(key_img)
            # Some workflows store {"image": "..."} instead of string
            if isinstance(old_val, dict):
                ov = dict(old_val)
                ov["image"] = reference_filename
                n["inputs"][key_img] = ov
            else:
                n["inputs"][key_img] = reference_filename

    if rid_ip and reference_strength is not None:
        n = _get_node(prompt, rid_ip)
        if n and isinstance(n.get("inputs"), dict):
            try:
                n["inputs"][key_w] = float(reference_strength)
            except Exception:
                n["inputs"][key_w] = 0.6

    return payload

def _build_signed_url_for_basename(basename: str, ttl_sec: Optional[int] = None) -> str:
    """Delegate to app.build_signed_url to create time-limited URL for URL-mode workflows."""
    from app import build_signed_url  # lazy import; raises if absent
    return build_signed_url(basename, ttl_sec) if ttl_sec is not None else build_signed_url(basename)

def inject_reference_url_inplace(
    body: Dict[str, Any],
    *,
    url: str,
    ref_url_node_id: Optional[str] = None,
    ref_url_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Inject signed URL into ImageFromURL node. Falls back to first node with matching key."""
    payload = _ensure_api_prompt_dict(body)
    prompt: Dict[str, Any] = payload["prompt"]

    rid = (ref_url_node_id or _COMFY_NODE_REF_URL or "").strip()
    key = (ref_url_key or _COMFY_KEY_REF_URL or "url")

    if rid:
        n = _get_node(prompt, rid)
        if n and isinstance(n.get("inputs"), dict) and key in n["inputs"]:
            n["inputs"][key] = url
            return payload

    for node in prompt.values():
        if isinstance(node, dict) and isinstance(node.get("inputs"), dict) and key in node["inputs"]:
            node["inputs"][key] = url
            return payload

    return payload

def stage_reference_url_and_patch_prompt(
    prompt_dict: Dict[str, Any],
    reference_local_path: Path,
    *,
    ref_url_node_id: Optional[str] = None,
    ref_url_key: Optional[str] = None,
    ttl_sec: Optional[int] = _APP_REF_TTL_SEC,
) -> Dict[str, Any]:
    """Create signed URL for local reference basename and inject into prompt (URL-mode)."""
    if not reference_local_path.exists() or not reference_local_path.is_file():
        raise FileNotFoundError(f"reference not found: {reference_local_path}")
    url = _build_signed_url_for_basename(reference_local_path.name, ttl_sec)
    return inject_reference_url_inplace(prompt_dict, url=url, ref_url_node_id=ref_url_node_id, ref_url_key=ref_url_key)

# =========================
# HTTP to ComfyUI
# =========================

async def _post_prompt(client: httpx.AsyncClient, base_url: str, body: Dict[str, Any]) -> str:
    """POST /prompt with retry/backoff and extract prompt_id."""
    delay = 0.8
    last_exc: Optional[Exception] = None
    for attempt in range(1, 5):
        try:
            r = await client.post(f"{base_url}/prompt", json=body)
            if r.status_code == 400:
                raise RuntimeError(f"comfy_400: {r.text[:400]}")
            r.raise_for_status()
            j = r.json()
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
    raise RuntimeError(f"comfy_post_prompt_failed: {last_exc or 'All connection attempts failed'}")

def _node_maps_from_history_obj(history_json: Dict[str, Any], prompt_id: str) -> List[Dict[str, Any]]:
    """Collect plausible node maps from history payload variants."""
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
    # Dedup
    seen = set()
    out: List[Dict[str, Any]] = []
    for m in node_maps:
        if id(m) in seen:
            continue
        seen.add(id(m))
        out.append(m)
    return out

def _any_images_in_node_maps(node_maps: Iterable[Dict[str, Any]]) -> bool:
    """Return True if at least one image record is present."""
    for node_map in node_maps:
        for node in node_map.values():
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
    max_wait_sec: float = float(_env_str("APP_COMFY_TIMEOUT_SEC", "240") or 240),
    poll_interval: float = 1.0,
) -> Dict[str, Any]:
    """Poll /history/{prompt_id} until images appear or timeout."""
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
    # Debug aid
    try:
        keys = list(last_payload.keys()) if isinstance(last_payload, dict) else []
        print("DEBUG comfy history keys:", keys)
        sample = json.dumps(last_payload or {}, ensure_ascii=False)[:2000]
        print("DEBUG comfy history sample:", sample)
    except Exception:
        pass
    raise TimeoutError("comfy_history_timeout")

def _build_view_candidates(folder_type: str, subfolder: str, filename: str) -> List[Tuple[str, str, str]]:
    """Build alternative triples to increase chances of successful download."""
    sub = (subfolder or "").strip().strip("/")
    cand: List[Tuple[str, str, str]] = []
    cand.append((folder_type or "output", sub, filename))
    if sub:
        cand.append((folder_type or "output", "", filename))
    cand.append(("temp", sub, filename))
    if sub:
        cand.append(("temp", "", filename))
    cand.append(("output", "", filename))
    # dedup
    seen = set()
    out: List[Tuple[str, str, str]] = []
    for t, s, f in cand:
        key = (t, s, f)
        if key in seen:
            continue
        seen.add(key)
        out.append((t, s, f))
    return out

def _is_likely_image(data: bytes) -> bool:
    """Quick signature + minimal-size check to avoid accepting HTML/errors as images."""
    if not data or len(data) < 64:
        return False
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    if data.startswith(b"\xff\xd8\xff"):
        return True
    if data.startswith(b"RIFF") and b"WEBP" in data[:16]:
        return True
    return len(data) >= 256

async def _download_via_view_path(
    client: httpx.AsyncClient,
    base_url: str,
    folder_type: str,
    subfolder: str,
    filename: str,
) -> bytes:
    """Download via /view/{type}/{subfolder}/{filename} with a few candidate permutations."""
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
        delay = 0.25
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
                    delay *= 1.8
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
    """Download via /api/view?filename=...&type=...&subfolder=... with permutations."""
    filename = (filename or "").strip().lstrip("/")
    candidates = _build_view_candidates(folder_type, subfolder, filename)
    last_exc: Optional[Exception] = None
    for idx, (t, s, f) in enumerate(candidates, start=1):
        params = {"filename": f, "type": (t or "output"), "subfolder": (s or "")}
        url = f"{base_url}/api/view?{urlencode(params, safe='/')}"
        delay = 0.25
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
                    delay *= 1.8
                else:
                    break
    if last_exc:
        raise last_exc
    raise RuntimeError("image_download_failed_unknown_query")

def _fs_fallback_read(folder_type: str, subfolder: str, filename: str) -> Optional[bytes]:
    """FS fallback read from APP_COMFY_OUTPUT_DIR when HTTP view fails."""
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
        pass
    return None

async def _download_images(
    client: httpx.AsyncClient,
    base_url: str,
    history_obj: Dict[str, Any],
    prompt_id: str,
    out_dir: Path,
    view_mode: str,
) -> List[Path]:
    """Iterate history, download all image outputs, and save under out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    def _iter_outputs() -> Iterable[Dict[str, Any]]:
        maps = _node_maps_from_history_obj(history_obj, prompt_id)
        for node_map in maps:
            for node in node_map.values():
                if not isinstance(node, dict):
                    continue
                outs = node.get("outputs")
                if not isinstance(outs, dict):
                    continue
                for outv in outs.values():
                    if isinstance(outv, dict) and isinstance(outv.get("images"), list):
                        yield outv

    for outv in _iter_outputs():
        for item in outv.get("images", []):
            if not isinstance(item, dict):
                continue
            filename = (item.get("filename") or "").strip()
            subfolder = item.get("subfolder", "")
            folder_type = item.get("type", "output")
            if not filename:
                continue

            data: Optional[bytes] = None
            try:
                if view_mode == "query":
                    data = await _download_via_view_query(client, base_url, folder_type, subfolder, filename)
                else:
                    data = await _download_via_view_path(client, base_url, folder_type, subfolder, filename)
            except Exception:
                data = _fs_fallback_read(folder_type, subfolder, filename)

            if not data:
                print(f"DEBUG comfy: could not fetch image '{filename}' ({folder_type}/{subfolder})")
                continue

            suffix = Path(filename).suffix or ".png"
            target = out_dir / f"img_{uuid.uuid4().hex}{suffix}"
            target.write_bytes(data)
            saved.append(target)

    if not saved:
        print("DEBUG comfy: no images saved from history.")
    return saved

# =========================
# Public API
# =========================

async def stage_reference_on_remote_and_patch_prompt(
    prompt_dict: Dict[str, Any],
    reference_local_path: Path,
    *,
    host: str = _env_str("APP_COMFY_HOST", "127.0.0.1"),
    port: int = int(_env_str("APP_COMFY_PORT", "8188") or 8188),
    ref_image_node_id: Optional[str] = None,
    ipadapter_node_id: Optional[str] = None,
    reference_strength: Optional[float] = None,
    ref_image_key: Optional[str] = None,
    ref_weight_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Stage reference for LoadImage nodes:
    - Localhost: copy to APP_COMFY_INPUT_DIR.
    - Remote: try HTTP upload endpoint (if available).
    - Patch prompt with basename and optional weight.
    """
    conn = ComfyConnection(host=host, port=port)
    base = conn.base
    payload = _ensure_api_prompt_dict(prompt_dict)

    filename_only: Optional[str] = None
    async with httpx.AsyncClient(limits=_limits(), timeout=_timeout(60)) as client:
        if host in {"127.0.0.1", "localhost"}:
            filename_only = _stage_reference_into_local_input(reference_local_path, _COMFY_INPUT_DIR)
            if not filename_only:
                filename_only = reference_local_path.name
        else:
            filename_only = await _upload_reference_to_remote_comfy(client, base, reference_local_path)
            if not filename_only:
                filename_only = reference_local_path.name  # last-resort fallback

    return override_reference_inplace(
        payload,
        reference_filename=filename_only,
        ref_image_node_id=ref_image_node_id,
        ipadapter_node_id=ipadapter_node_id,
        reference_strength=reference_strength,
        ref_image_key=ref_image_key,
        ref_weight_key=ref_weight_key,
    )

def stage_reference_url_and_patch_prompt_sync(
    prompt_dict: Dict[str, Any],
    reference_local_path: Path,
    *,
    ref_url_node_id: Optional[str] = None,
    ref_url_key: Optional[str] = None,
    ttl_sec: Optional[int] = _APP_REF_TTL_SEC,
) -> Dict[str, Any]:
    """Sync helper for URL-mode reference injection (wraps app.build_signed_url)."""
    return stage_reference_url_and_patch_prompt(
        prompt_dict=prompt_dict,
        reference_local_path=reference_local_path,
        ref_url_node_id=ref_url_node_id,
        ref_url_key=ref_url_key,
        ttl_sec=ttl_sec,
    )

async def generate_from_prompt_dict(
    prompt_dict: Optional[Dict[str, Any]] = None,
    out_dir: Path = Path(_env_str("APP_OUTPUT_DIR", "./outputs/images")),
    *,
    # legacy kw 'prompt' supported via kwargs
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
    host: str = _env_str("APP_COMFY_HOST", "127.0.0.1"),
    port: int = int(_env_str("APP_COMFY_PORT", "8188") or 8188),
    max_wait_sec: float = float(_env_str("APP_COMFY_TIMEOUT_SEC", "240") or 240),
    poll_interval: float = 1.0,
    **kwargs: Any,
) -> List[Path]:
    """
    Submit prompt to ComfyUI and save resulting images:
    - Normalizes body, applies overrides (text, dims, sampler).
    - POST /prompt, poll /history/{id} until images exist.
    - Download images to out_dir; return list of saved file paths.

    Backward-compat: also accepts legacy keyword 'prompt' instead of 'prompt_dict'.
    """
    if prompt_dict is None:
        legacy = kwargs.pop("prompt", None)
        if legacy is None:
            raise TypeError("generate_from_prompt_dict() requires 'prompt_dict' (or legacy 'prompt').")
        prompt_dict = legacy

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

# Legacy alias
async def generate_from_prompt(
    prompt: Dict[str, Any],
    out_dir: Path,
    **kwargs: Any,
) -> List[Path]:
    return await generate_from_prompt_dict(prompt_dict=prompt, out_dir=out_dir, **kwargs)


# =========================
# Optional: Minimal LocalComfyBackend stub
# =========================

class LocalComfyBackend:
    """
    Minimal backend adapter used by the app. It provides:
      - generate(prompt_dict, ...) -> returns list of Paths
      - _copy_latest_from_comfy(...) legacy method retained for compatibility,
        but now returns already-downloaded images from the bridge.
    """

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, out_dir: Optional[Path] = None) -> None:
        self.host = host or _env_str("APP_COMFY_HOST", "127.0.0.1")
        self.port = int(port or int(_env_str("APP_COMFY_PORT", "8188") or 8188))
        self.out_dir = out_dir or Path(_env_str("APP_OUTPUT_DIR", "./outputs/images")).resolve()

    async def generate(
        self,
        prompt_dict: Dict[str, Any],
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
        max_wait_sec: Optional[float] = None,
    ) -> List[Path]:
        paths = await generate_from_prompt_dict(
            prompt_dict=prompt_dict,
            out_dir=self.out_dir,
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
            host=self.host,
            port=self.port,
            max_wait_sec=float(max_wait_sec or float(_env_str("APP_COMFY_TIMEOUT_SEC", "240") or 240)),
        )
        return paths

    async def _copy_latest_from_comfy(self) -> List[Path]:
        """
        Legacy name kept for compatibility with callers expecting this method.
        In the refactored bridge, images are already downloaded into out_dir.
        We implement this by returning the most recent files from out_dir.
        """
        if not self.out_dir.exists():
            return []
        # Collect newest images (PNG/JPG/WEBP)
        candidates = sorted(
            [p for p in self.out_dir.glob("**/*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        # Return up to the last 4 recent files to be safe
        return candidates[:4]

