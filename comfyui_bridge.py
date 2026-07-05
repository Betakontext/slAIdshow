# comfyui_bridge.py
# --------------------------------------------------------------------------------------
# Production-grade ComfyUI bridge with robust remote (tunnel) handling + bridge artifacts.
#
# This version adds:
# - Deterministic remote artifact sync helper: sync_bridge_artifacts(host, port, ...)
#   so the UI can fetch tunnel_url.json/txt immediately after switching to remote.
# - Keeps existing behavior: pre/post artifact copy around image generation in remote/query mode.
# - Uses the SAME download mechanism as images (ComfyUI /api/view or /view), with retries.
# - Adds FS fallback for artifacts when APP_COMFY_OUTPUT_DIR is set.
# - Tests common subfolders for artifacts ("", "tunnels", "bridge") in addition to folder_type.
#
# Public API:
#   - override_prompt_inplace(...)
#   - generate_from_prompt_dict(..., copy_bridge_artifacts: bool = True)
#   - sync_bridge_artifacts(host: str, port: int, *, view_mode: Optional[str] = None) -> Dict[str, int]
#
# Env expectations for remote:
#   APP_ALLOW_REMOTE_BACKENDS=1
#   APP_COMFY_SCHEME=https
#   APP_COMFY_FORCE_VIEW_MODE=query
#   (host is non-local, port likely 443)
# --------------------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable
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
    Privacy/safety policy:
    - Always allow loopback.
    - Remote only if APP_ALLOW_REMOTE_BACKENDS=1 and, if provided, APP_ALLOWED_SUBNETS allows the IP.
      Hostnames (e.g., Cloudflared hosts) bypass subnet check because we cannot pre-resolve safely here.
    """
    if host in {"127.0.0.1", "localhost"}:
        return
    allow_remote = _env_bool01("APP_ALLOW_REMOTE_BACKENDS", 0)
    if not allow_remote:
        raise AssertionError(f"Only localhost allowed, got {host}")
    subnets = _env_str("APP_ALLOWED_SUBNETS", "")
    if not subnets:
        return
    # If host is an IP, enforce subnet allowlist; if hostname, skip (cannot reliably map CNAMEs here).
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return
    if not _is_in_allowed_subnets(host, subnets):
        raise AssertionError(f"Remote host {host} not in allowed subnets ({subnets})")

# Optional filesystem fallback when /view is unavailable or unsuitable (mounted output dir).
_COMFY_OUTPUT_DIR: Optional[Path] = Path(_env_str("APP_COMFY_OUTPUT_DIR", "")).resolve() if _env_str("APP_COMFY_OUTPUT_DIR", "") else None

# Local bridge dir for discovery artifacts exposed under /static/bridge/ by the server
_LOCAL_BRIDGE_DIR: Path = Path("outputs/images/bridge").resolve()

# -----------------------------
# Connection / helpers
# -----------------------------

class ComfyConnection(BaseModel):
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8188)
    scheme: Optional[str] = Field(default=None)  # 'http' or 'https'; if None, auto by port or env

    @property
    def base(self) -> str:
        """
        Build base URL with proper scheme.
        - Prefer explicit APP_COMFY_SCHEME if provided.
        - Else use provided scheme.
        - Else auto-select https when port == 443, otherwise http.
        """
        _assert_image_backend_host_policy(self.host)
        env_scheme = (_env_str("APP_COMFY_SCHEME") or "").lower()
        scheme = (self.scheme or env_scheme or "").strip()
        if scheme not in {"http", "https"}:
            scheme = "https" if int(self.port) == 443 else "http"
        base = f"{scheme}://{self.host}:{self.port}"
        print(f"[COMFY BASE] {base}")
        return base

def _limits() -> httpx.Limits:
    # Keep connections warm but conservative
    return httpx.Limits(max_keepalive_connections=8, max_connections=16, keepalive_expiry=30.0)

def _timeout(total: float = 150.0) -> httpx.Timeout:
    # Boundaries to prevent runaway timeouts; adds headroom for long generations
    total = max(30.0, min(total, 300.0))
    connect = float(_env_str("APP_COMFY_CONNECT_TIMEOUT_SEC", "5") or "5")
    connect = max(2.0, min(connect, 30.0))
    return httpx.Timeout(connect=connect, read=total, write=12.0, pool=8.0)

def _clamp_dim(v: Optional[int]) -> Optional[int]:
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
    node_id_positive: str = "2",
    node_id_negative: str = "3",
    node_id_latent: str = "4",
    node_id_ksampler: str = "5",
) -> Dict[str, Any]:
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
# HTTP calls
# -----------------------------

async def _post_prompt(
    client: httpx.AsyncClient,
    base_url: str,
    body: Dict[str, Any],
) -> str:
    """
    POST /prompt with retries. Classify HTTP 400 as prompt/workflow error (reachable),
    not as 'unavailable'.
    """
    delay = 0.8
    last_exc: Optional[Exception] = None
    for attempt in range(1, 5):
        try:
            url = f"{base_url}/prompt"
            print(f"DEBUG comfy POST {url} attempt#{attempt}")
            r = await client.post(url, json=body)
            if r.status_code == 400:
                text = r.text[:500].replace("\n", " ")
                print(f"DEBUG comfy POST /prompt 400 body: {text}")
                raise RuntimeError(f"comfy_prompt_invalid_400: {text}")
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
            print(f"DEBUG comfy POST /prompt attempt#{attempt} failed: {repr(e)}")
            if attempt < 4:
                await asyncio.sleep(delay)
                delay *= 1.7
                continue
            break
        except Exception as e:
            print(f"DEBUG comfy POST /prompt unexpected error: {repr(e)}")
            raise
    raise RuntimeError(f"comfy_post_prompt_failed: {last_exc}")

def _node_maps_from_history_obj(history_json: Dict[str, Any], prompt_id: str) -> List[Dict[str, Any]]:
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
    deadline = time.time() + max_wait_sec
    last_payload: Optional[Dict[str, Any]] = None
    while time.time() < deadline:
        url = f"{base_url}/history/{prompt_id}"
        r = await client.get(url)
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
    try:
        keys = list(last_payload.keys()) if isinstance(last_payload, dict) else "n/a"
        print("DEBUG comfy history keys:", keys)
        sample = json.dumps(last_payload or {}, ensure_ascii=False)[:2000]
        print("DEBUG comfy history sample:", sample)
    except Exception:
        pass
    raise TimeoutError(f"comfy_history_timeout (last_payload_keys={keys if isinstance(keys, list) else keys})")

def _iter_image_descriptors_from_history(history_obj: Dict[str, Any], prompt_id: str) -> Iterable[Dict[str, Any]]:
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
    Used by both path and query modes.
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
        uniq.append(key)
    return uniq

async def _download_via_view_path(
    client: httpx.AsyncClient,
    base_url: str,
    folder_type: str,
    subfolder: str,
    filename: str,
) -> bytes:
    """
    Download via legacy path style: /view/{type}/{subfolder}/{filename}
    Tries multiple candidates; URL-encodes each segment.
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
                data = r.content
                if not data or len(data) < 64:
                    raise RuntimeError("download_too_small")
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
    raise RuntimeError("download_failed_unknown_path")

async def _download_via_view_query(
    client: httpx.AsyncClient,
    base_url: str,
    folder_type: str,
    subfolder: str,
    filename: str,
) -> bytes:
    """
    Download via query style: /api/view?filename=...&type=...&subfolder=...
    Tries multiple (type, subfolder, filename) candidates.
    IMPORTANT: Do NOT include 'subfolder' when it is empty, as some tunnels/proxies mishandle empty params.
    """
    filename = (filename or "").strip().lstrip("/")
    candidates = _build_view_candidates(folder_type, subfolder, filename)
    last_exc: Optional[Exception] = None

    headers = {
        "Accept": "*/*",
        "Cache-Control": "no-cache",
        "Connection": "close",
        "Accept-Encoding": "identity",
    }

    for idx, (t, s, f) in enumerate(candidates, start=1):
        params = {"filename": f, "type": (t or "output")}
        s_clean = (s or "").strip().strip("/")
        if s_clean:
            params["subfolder"] = s_clean

        qs = urlencode(params, quote_via=quote, safe="")
        url = f"{base_url}/api/view?{qs}"

        delays = [0.2, 0.4, 0.8, 1.2]
        for attempt, delay in enumerate(delays, start=1):
            try:
                print(f"DEBUG comfy GET[query] try#{attempt} cand#{idx}: {url}")
                r = await client.get(url, headers=headers)
                print(f"DEBUG comfy GET[query] status={r.status_code} bytes={len(r.content) if r.content else 0}")
                r.raise_for_status()
                data = r.content
                if not data or len(data) < 64:
                    raise RuntimeError(f"download_too_small (url={url})")
                return data
            except Exception as e:
                last_exc = e
                print(f"DEBUG comfy GET[query] error: {repr(e)} (url={url})")
                if attempt < len(delays):
                    await asyncio.sleep(delay)
                    continue
                else:
                    break
    if last_exc:
        raise last_exc
    raise RuntimeError("download_failed_unknown_query")

def _fs_fallback_read(folder_type: str, subfolder: str, filename: str) -> Optional[bytes]:
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

# -----------------------------
# Artifact copy (bridge)
# -----------------------------

async def _copy_bridge_artifacts(
    client: httpx.AsyncClient,
    base_url: str,
    view_mode: str,
    out_dir_bridge: Path,
) -> Dict[str, int]:
    """
    Attempt to fetch tunnel_url.{json,txt} from the remote using the same view mechanism
    and write them into outputs/images/bridge locally.
    Non-fatal on failure; logs and returns summary counts.

    Improvements:
    - Also tries common artifact subfolders: "", "tunnels", "bridge"
    - Accepts smaller thresholds: JSON/TXT >= 3 bytes
    - Uses FS fallback when APP_COMFY_OUTPUT_DIR is set
    """
    out_dir_bridge.mkdir(parents=True, exist_ok=True)

    async def _try_fetch_one(filename: str) -> Optional[bytes]:
        # Try candidates across folder types and common subfolders
        subfolders = ["", "tunnels", "bridge"]
        for folder_type in ("output", "temp"):
            for sub in subfolders:
                try:
                    if view_mode == "query":
                        return await _download_via_view_query(client, base_url, folder_type, sub, filename)
                    else:
                        return await _download_via_view_path(client, base_url, folder_type, sub, filename)
                except Exception as e:
                    print(f"DEBUG bridge artifact fetch failed for {filename} type={folder_type} sub='{sub}': {repr(e)}")
                    continue
        # FS fallback attempts with same subfolders
        for sub in ["", "tunnels", "bridge"]:
            data = _fs_fallback_read("output", sub, filename) or _fs_fallback_read("temp", sub, filename)
            if data:
                return data
        return None

    saved = 0
    errors = 0
    for fname in ("tunnel_url.json", "tunnel_url.txt"):
        try:
            data = await _try_fetch_one(fname)
            if data and len(data) >= 3:
                target = out_dir_bridge / fname
                target.write_bytes(data)
                saved += 1
                print(f"[BRIDGE] saved {fname} bytes={len(data)} -> {target}")
            else:
                print(f"[BRIDGE] not found or too small: {fname}")
                errors += 1
        except Exception as e:
            errors += 1
            print(f"[BRIDGE] error saving {fname}: {repr(e)}")
    return {"saved": saved, "errors": errors}

# -----------------------------
# Image downloads
# -----------------------------

async def _download_images(
    client: httpx.AsyncClient,
    base_url: str,
    history_obj: Dict[str, Any],
    prompt_id: str,
    out_dir: Path,
    view_mode: str,
) -> List[Path]:
    """
    Download images from history using either 'path' or 'query' mode.
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

            print(f"[HISTORY] chosen file='{filename}' type='{folder_type}' subfolder='{subfolder if subfolder else '(empty)'}'")

            delay = 0.8
            data: Optional[bytes] = None
            for attempt in range(1, 3):
                try:
                    if view_mode == "query":
                        data = await _download_via_view_query(client, base_url, folder_type, subfolder, filename)
                    else:
                        data = await _download_via_view_path(client, base_url, folder_type, subfolder, filename)
                    break
                except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError, httpx.HTTPStatusError) as e:
                    print(f"DEBUG comfy download attempt#{attempt} failed: {repr(e)}")
                    if attempt < 2:
                        await asyncio.sleep(delay)
                        delay *= 1.7
                        continue
                except Exception as e:
                    print(f"DEBUG comfy download unexpected error (will try FS fallback): {repr(e)}")

            if data is None:
                data = _fs_fallback_read(folder_type, subfolder, filename)

            if data is None:
                print(f"DEBUG comfy: skip missing image filename='{filename}' type='{folder_type}' subfolder='{subfolder}'")
                continue

            suffix = Path(filename).suffix or ".png"
            target = out_dir / f"img_{uuid.uuid4().hex}{suffix}"
            target.write_bytes(data)
            print(f"DEBUG comfy saved image: {target.name} bytes={len(data)}")
            saved.append(target)

    if not found_any:
        print("DEBUG comfy: no images found in history object (iter yielded none).")

    return saved

# -----------------------------
# Public entry points
# -----------------------------

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
    copy_bridge_artifacts: bool = True,
) -> List[Path]:
    """
    Submit a ComfyUI API prompt dict, poll history, and download images to out_dir.
    Additionally (when copy_bridge_artifacts=True), copy discovery artifacts
    (tunnel_url.json/.txt) from the remote into outputs/images/bridge/ for the Web UI.

    Selects view mode automatically (path for local, query for remote) or via env override.

    Caller responsibilities:
    - Local/LAN: host=127.0.0.1 or LAN IP, port 8188; APP_COMFY_SCHEME unset or 'http'; APP_COMFY_FORCE_VIEW_MODE auto/path.
    - Remote (Cloudflared): host=<tunnel-host>, port=443; set APP_COMFY_SCHEME=https and APP_COMFY_FORCE_VIEW_MODE=query.
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

    conn = ComfyConnection(
        host=host,
        port=port,
        scheme=(_env_str("APP_COMFY_SCHEME") or None)
    )
    base = conn.base
    view_mode = _select_view_mode(conn.host)
    print(f"[COMFY VIEW MODE] {view_mode} (host={conn.host})")
    print(f"[COMFY ENDPOINTS] prompt={base}/prompt history={base}/history/{{prompt_id}} view={'/api/view?...' if view_mode=='query' else '/view/...'}")

    async with httpx.AsyncClient(limits=_limits(), timeout=_timeout(max_wait_sec + 30)) as client:
        # Best-effort: copy bridge artifacts up-front (in case already published)
        if copy_bridge_artifacts and view_mode == "query" and host not in {"127.0.0.1", "localhost"}:
            try:
                await _copy_bridge_artifacts(client, base, view_mode, _LOCAL_BRIDGE_DIR)
            except Exception as e:
                print(f"[BRIDGE] pre-copy error: {repr(e)}")

        prompt_id = await _post_prompt(client, base, payload)
        history_obj = await _poll_history_ready(client, base, prompt_id, max_wait_sec=max_wait_sec, poll_interval=poll_interval)

        # Try again after generation (common case: artifacts appear during/after runs)
        if copy_bridge_artifacts and view_mode == "query" and host not in {"127.0.0.1", "localhost"}:
            try:
                await _copy_bridge_artifacts(client, base, view_mode, _LOCAL_BRIDGE_DIR)
            except Exception as e:
                print(f"[BRIDGE] post-copy error: {repr(e)}")

        images = await _download_images(client, base, history_obj, prompt_id, out_dir=Path(out_dir), view_mode=view_mode)
    return images

async def sync_bridge_artifacts(
    *,
    host: str,
    port: int,
    view_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Explicitly fetch tunnel_url.json and tunnel_url.txt from the active ComfyUI host and
    save them into outputs/images/bridge/, so the Web UI (/static/bridge/...) can read them.

    Usage:
      - Call right after switching to remote mode in the UI, on app start, and on reload.
      - Returns a JSON summary: {"saved": N, "errors": M, "host": "...", "base": "...", "view_mode": "query|path"}.

    Notes:
      - Enforces host policy (APP_ALLOW_REMOTE_BACKENDS=1 for non-local).
      - Uses APP_COMFY_SCHEME and APP_COMFY_FORCE_VIEW_MODE to determine base URL and view mode when not provided.
    """
    conn = ComfyConnection(
        host=host,
        port=port,
        scheme=(_env_str("APP_COMFY_SCHEME") or None),
    )
    base = conn.base
    vm = view_mode or _select_view_mode(conn.host)
    print(f"[BRIDGE SYNC] view_mode={vm} host={conn.host} base={base}")

    async with httpx.AsyncClient(limits=_limits(), timeout=_timeout(30.0)) as client:
        result = await _copy_bridge_artifacts(client, base, vm, _LOCAL_BRIDGE_DIR)
        result.update({"host": host, "base": base, "view_mode": vm})
        return result
