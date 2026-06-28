# comfyui_bridge.py
# Comments strictly in English

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field, ValidationError
from pydantic import RootModel  # Pydantic v2 RootModel


# ========= Debug / ENV helpers =========

def _debug() -> bool:
    v = (os.getenv("APP_COMFY_BRIDGE_DEBUG", os.getenv("APP_IMAGE_BACKEND_DEBUG", "0")) or "").strip().lower()
    return v in {"1", "true", "yes", "on"}


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


def _now() -> float:
    return time.time()


def _httpx_limits() -> httpx.Limits:
    # Keep connections warm but bounded
    return httpx.Limits(max_keepalive_connections=10, max_connections=20, keepalive_expiry=30.0)


def _timeout_default() -> httpx.Timeout:
    # Conservative per-request timeout; overall generation timeout handled by polling budget
    return httpx.Timeout(connect=5.0, read=15.0, write=10.0, pool=5.0)


# ========= Security: host policy =========

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
    """Enforce privacy policy for ComfyUI host usage."""
    if host in {"127.0.0.1", "localhost"}:
        return
    allow_remote = _env_bool01("APP_ALLOW_REMOTE_BACKENDS", 0)
    if not allow_remote:
        raise AssertionError(f"Only localhost allowed, got {host}")
    subnets = _env_str("APP_COMFY_REMOTE_WHITELIST", "")
    if not subnets:
        # If no whitelist provided, allow but still restrict to the explicit host given
        return
    try:
        # If host is an IP we can check against subnets; hostnames are allowed but not validated here
        ipaddress.ip_address(host)
    except ValueError:
        return
    if not _is_in_allowed_subnets(host, subnets):
        raise AssertionError(f"Remote host {host} not in allowed subnets ({subnets})")


# ========= Pydantic models for selected responses =========

class _PromptSubmitResponse(BaseModel):
    prompt_id: str = Field(alias="prompt_id")


class _ImageInfo(BaseModel):
    filename: str
    subfolder: str
    type: str


# Keep RootModel for potential future validation reuse, but we do not rely on it for history parsing anymore
class _HistoryNodeOutput(BaseModel):
    images: List[_ImageInfo] = Field(default_factory=list)


class _HistoryPrompt(BaseModel):
    outputs: Dict[str, _HistoryNodeOutput] = Field(default_factory=dict)


class _HistoryEntry(BaseModel):
    prompt: _HistoryPrompt


class _PromptHistoryResponse(RootModel[Dict[str, _HistoryEntry]]):
    def entry(self, pid: str) -> Optional[_HistoryEntry]:
        return self.root.get(pid)


# ========= Internal helpers =========

def _ensure_api_prompt_dict(prompt_dict: Dict[str, Any] | Any) -> Dict[str, Any]:
    """Normalize to Comfy /prompt POST body: {'prompt': {...}}."""
    if isinstance(prompt_dict, dict) and "prompt" in prompt_dict and isinstance(prompt_dict["prompt"], dict):
        return {"prompt": prompt_dict["prompt"]}
    if isinstance(prompt_dict, dict):
        return {"prompt": prompt_dict}
    raise TypeError("prompt_dict must be a dict or {'prompt': {...}}")


def _sanitize_filename(name: str) -> str:
    # Keep simple safe characters to avoid path traversal
    name = name.replace("\\", "/").split("/")[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _choose_view_mode(host: str) -> str:
    """Decide between 'path' (/view/...) and 'query' (/api/view?...)."""
    override = (_env_str("APP_COMFY_FORCE_VIEW_MODE", "") or "").strip().lower()
    if override in {"path", "query"}:
        return override
    return "path" if host in {"127.0.0.1", "localhost"} else "query"


def _image_min_bytes() -> int:
    # Accept very small artifacts by default unless overridden; avoids false negatives on small PNGs/WebPs.
    return max(128, _env_int("APP_COMFY_MIN_IMAGE_BYTES", 512))


def _max_images_to_collect() -> int:
    # Collect up to N images from a generation; typical is 1
    return max(1, _env_int("APP_COMFY_MAX_IMAGES", 4))


# ========= HTTP with retries =========

async def _retrying_post_json(client: httpx.AsyncClient, url: str, payload: Dict[str, Any], *, max_attempts: int = 4, base_delay: float = 0.6) -> httpx.Response:
    last_exc: Optional[Exception] = None
    delay = float(base_delay)
    for attempt in range(1, max_attempts + 1):
        try:
            r = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
            # Treat 4xx (except 408/429) as terminal
            if r.status_code in (400, 401, 403, 404, 405):
                r.raise_for_status()
            if r.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError(f"transient {r.status_code}", request=r.request, response=r)
            r.raise_for_status()
            return r
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError, httpx.HTTPStatusError) as e:
            last_exc = e
            status = getattr(e, "response", None).status_code if getattr(e, "response", None) else None
            retryable = (status in (429, 500, 502, 503, 504)) or isinstance(e, (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError))
            if attempt >= max_attempts or not retryable:
                break
            await asyncio.sleep(delay)
            delay *= 1.8
    raise RuntimeError(f"post_failed after {max_attempts} attempts: {last_exc}")


async def _retrying_get(client: httpx.AsyncClient, url: str, *, max_attempts: int = 4, base_delay: float = 0.6) -> httpx.Response:
    last_exc: Optional[Exception] = None
    delay = float(base_delay)
    for attempt in range(1, max_attempts + 1):
        try:
            r = await client.get(url)
            if r.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError(f"transient {r.status_code}", request=r.request, response=r)
            r.raise_for_status()
            return r
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError, httpx.HTTPStatusError) as e:
            last_exc = e
            status = getattr(e, "response", None).status_code if getattr(e, "response", None) else None
            retryable = (status in (429, 500, 502, 503, 504)) or isinstance(e, (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError))
            if attempt >= max_attempts or not retryable:
                break
            await asyncio.sleep(delay)
            delay *= 1.8
    raise RuntimeError(f"get_failed after {max_attempts} attempts: {last_exc}")


# ========= Core Comfy flow =========

async def _post_prompt(host: str, port: int, payload: Dict[str, Any]) -> str:
    """POST /prompt and return prompt_id."""
    url = f"http://{host}:{port}/prompt"
    async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_default(), follow_redirects=False) as client:
        r = await _retrying_post_json(client, url, payload)
        try:
            parsed = _PromptSubmitResponse.model_validate(r.json())
            return parsed.prompt_id
        except ValidationError as e:
            raise RuntimeError(f"invalid_prompt_submit_response: {e}")


async def _poll_history_for_images(host: str, port: int, prompt_id: str, *, max_wait_sec: float) -> List[_ImageInfo]:
    """
    Poll /history/{id} until images are available or timeout exceeded.

    Compatibility strategy:
    - Prefer 'outputs' at top-level under the prompt_id entry (most ComfyUI builds).
    - Fallback to 'prompt.outputs' for builds that nest outputs inside the 'prompt' field.
    """
    t0 = _now()
    url = f"http://{host}:{port}/history/{prompt_id}"
    delay = 0.5
    async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_default(), follow_redirects=False) as client:
        while True:
            if _now() - t0 > max_wait_sec:
                raise TimeoutError(f"history_poll_timeout after {max_wait_sec:.1f}s")
            try:
                r = await _retrying_get(client, url)
                data = r.json()
                entry = data.get(prompt_id, {})
                # Try preferred shape: outputs at top-level
                outputs = entry.get("outputs")
                # Fallback: outputs nested under entry['prompt']['outputs']
                if outputs is None and isinstance(entry.get("prompt"), dict):
                    prm = entry["prompt"]
                    if isinstance(prm, dict):
                        # Some builds use prompt as a tuple/list; we only care about 'outputs' dict if present
                        if isinstance(prm.get("outputs"), dict):
                            outputs = prm.get("outputs")

                images: List[_ImageInfo] = []
                if isinstance(outputs, dict):
                    for node_id, node_out in outputs.items():
                        # Each node_out should have {"images": [ {filename, subfolder, type}, ... ]}
                        imgs = None
                        if isinstance(node_out, dict):
                            imgs = node_out.get("images")
                        if not isinstance(imgs, list):
                            continue
                        for im in imgs:
                            if not isinstance(im, dict):
                                continue
                            # Validate and normalize via Pydantic to enforce required keys
                            try:
                                info = _ImageInfo.model_validate(im)
                                images.append(info)
                            except ValidationError:
                                continue

                if images:
                    if _debug():
                        names = [f"{x.subfolder}/{x.filename}".strip("/") for x in images]
                        print(f"[COMFY][history] images_ready count={len(images)} nodes={list(outputs.keys()) if isinstance(outputs, dict) else 'n/a'} files={names}")
                    # Enforce max collect cap
                    return images[:_max_images_to_collect()]

                if _debug():
                    # Show lightweight hints for debugging schema
                    keys = list(entry.keys()) if isinstance(entry, dict) else type(entry).__name__
                    print(f"[COMFY][history] no_images_yet keys={keys} retry_in={delay:.2f}s")

            except (httpx.HTTPError, KeyError, ValueError, json.JSONDecodeError) as e:
                if _debug():
                    print(f"[COMFY][history] transient: {type(e).__name__}: {e}")

            await asyncio.sleep(delay)
            delay = min(2.0, delay * 1.2)


# ========= Image download =========

async def _download_image_path_mode(host: str, port: int, info: _ImageInfo, out_dir: Path) -> Optional[Path]:
    """
    Try path mode: /view/{type}/{subfolder}/{filename}
    This commonly works for localhost setups where Comfy serves static files.
    """
    base = f"http://{host}:{port}"
    fname = _sanitize_filename(info.filename)
    subf = "/".join([_sanitize_filename(p) for p in (info.subfolder or "").strip("/").split("/") if p])
    t = _sanitize_filename(info.type or "output")
    url = f"{base}/view/{t}/{subf}/{fname}" if subf else f"{base}/view/{t}/{fname}"
    async with httpx.AsyncClient(limits=_httpx_limits(), timeout=httpx.Timeout(30.0), follow_redirects=False) as client:
        try:
            r = await _retrying_get(client, url)
            content = r.content
            if not content or len(content) < _image_min_bytes():
                if _debug():
                    print(f"[COMFY][dl:path] content_too_small len={len(content) if content else 0} url={url}")
                return None
            suffix = Path(fname).suffix.lower() or ".png"
            target = out_dir / f"img_{uuid.uuid4().hex}{suffix}"
            target.write_bytes(content)
            if _debug():
                print(f"[COMFY][dl:path] saved -> {target}")
            return target
        except Exception as e:
            if _debug():
                print(f"[COMFY][dl:path] {e}")
            return None


async def _download_image_query_mode(host: str, port: int, info: _ImageInfo, out_dir: Path) -> Optional[Path]:
    """
    Try query mode: /api/view?type=&subfolder=&filename=
    This is recommended for remote deployments where static path serving may be disabled.
    """
    base = f"http://{host}:{port}"
    params = {
        "filename": info.filename,
        "subfolder": info.subfolder,
        "type": info.type,
    }
    url = f"{base}/api/view"
    async with httpx.AsyncClient(limits=_httpx_limits(), timeout=httpx.Timeout(30.0), follow_redirects=False) as client:
        try:
            r = await client.get(url, params=params)
            if r.status_code in (429, 500, 502, 503, 504):
                # Light retry for transient server busy
                qp = f"filename={params['filename']}&subfolder={params['subfolder']}&type={params['type']}"
                r = await _retrying_get(client, url=f"{url}?{qp}")
            r.raise_for_status()
            content = r.content
            if not content or len(content) < _image_min_bytes():
                if _debug():
                    print(f"[COMFY][dl:query] content_too_small len={len(content) if content else 0} url={url} params={params}")
                return None
            suffix = Path(params["filename"]).suffix.lower() or ".png"
            target = out_dir / f"img_{uuid.uuid4().hex}{suffix}"
            target.write_bytes(content)
            if _debug():
                print(f"[COMFY][dl:query] saved -> {target}")
            return target
        except Exception as e:
            if _debug():
                print(f"[COMFY][dl:query] {e} params={params}")
            return None


async def _download_images(host: str, port: int, images: List[_ImageInfo], out_dir: Path) -> List[Path]:
    """Attempt to download images using the preferred view mode; fallback to alternate mode if needed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = _choose_view_mode(host)
    results: List[Path] = []

    async def _try_one(info: _ImageInfo) -> Optional[Path]:
        # Prefer selected mode; if it fails, fallback to the other
        if mode == "path":
            p = await _download_image_path_mode(host, port, info, out_dir)
            if p is not None:
                return p
            return await _download_image_query_mode(host, port, info, out_dir)
        else:
            p = await _download_image_query_mode(host, port, info, out_dir)
            if p is not None:
                return p
            return await _download_image_path_mode(host, port, info, out_dir)

    for info in images:
        p = await _try_one(info)
        if p is not None:
            results.append(p)
        if len(results) >= _max_images_to_collect():
            break
    return results


# ========= Public API =========

async def generate_from_prompt_dict(
    *,
    prompt_dict: Dict[str, Any],
    out_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8188,
    max_wait_sec: float | int | None = None,
) -> List[Path]:
    """
    Main entrypoint consumed by image_backend.LocalComfyBackend.
    - Normalizes the workflow prompt payload
    - Sends to ComfyUI /prompt
    - Polls /history/{id}
    - Downloads resulting images and returns their paths
    """
    # Security: enforce host policy
    _assert_image_backend_host_policy(host)

    # Resolve timeout budget
    budget = float(max_wait_sec if (max_wait_sec is not None) else _env_float("APP_COMFY_TIMEOUT_SEC", 180.0))
    payload = _ensure_api_prompt_dict(prompt_dict)

    if _debug():
        print(f"[COMFY][submit] host={host}:{port} budget={budget:.1f}s prompt_nodes={len(payload.get('prompt', {}))}")

    # POST prompt
    prompt_id = await _post_prompt(host, port, payload)

    # Poll history
    images_info = await _poll_history_for_images(host, port, prompt_id, max_wait_sec=budget)

    # Download images
    paths = await _download_images(host, port, images_info, Path(out_dir).resolve())

    if not paths:
        raise RuntimeError("no_images_downloaded")
    return paths


# ========= Optional URL reference injection helper =========

def stage_reference_url_and_patch_prompt_sync(
    *,
    prompt_dict: Dict[str, Any],
    reference_local_path: Path,
) -> Dict[str, Any]:
    """
    Optional helper used by image_backend for APP_COMFY_REF_MODE=url.
    - Builds a signed URL for the given local path by calling app.build_signed_url(name, ttl).
    - Patches the provided prompt payload to insert the URL into a URL-capable node.
    Contract:
      - Input prompt_dict can be {'prompt': {...}} or raw prompt map; we return the same wrapper shape as input.
    """
    # Keep wrapper shape
    had_wrapper = isinstance(prompt_dict, dict) and "prompt" in prompt_dict and isinstance(prompt_dict["prompt"], dict)
    body = _ensure_api_prompt_dict(prompt_dict)
    prompt_map: Dict[str, Any] = body["prompt"]

    # Try to import app.build_signed_url dynamically
    try:
        import app as _app  # type: ignore
        build_signed_url = getattr(_app, "build_signed_url")
    except Exception as e:
        raise RuntimeError(f"build_signed_url_not_available: {e}")

    # Determine TTL and URL node config from ENV
    ttl = _env_int("APP_REF_TTL_SEC", 900)
    node_id_ref_url = _env_str("APP_COMFY_NODE_REF_URL", "") or None
    node_key_ref_url = _env_str("APP_COMFY_KEY_REF_URL", "url") or "url"

    # Compute signed URL using basename to avoid leaking paths
    if not isinstance(reference_local_path, Path):
        reference_local_path = Path(str(reference_local_path))
    if not reference_local_path.exists() or not reference_local_path.is_file():
        raise FileNotFoundError(reference_local_path)
    signed = build_signed_url(reference_local_path.name, ttl=ttl)  # type: ignore[misc]
    if not isinstance(signed, str) or not signed.startswith(("http://", "https://")):
        raise RuntimeError("signed_url_invalid")

    # Patch specific node if configured
    patched = False
    if node_id_ref_url:
        node = prompt_map.get(node_id_ref_url)
        if isinstance(node, dict):
            inputs = node.get("inputs")
            if isinstance(inputs, dict):
                inputs[node_key_ref_url] = signed
                patched = True

    # Fallback scan: look for inputs with 'url' key
    if not patched:
        for node in prompt_map.values():
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs")
            if isinstance(inputs, dict):
                if "url" in inputs and isinstance(inputs.get("url"), (str, type(None), dict)):
                    if isinstance(inputs.get("url"), dict):
                        ov = dict(inputs["url"])
                        ov["url"] = signed
                        inputs["url"] = ov
                    else:
                        inputs["url"] = signed
                    patched = True
                    break

    if _debug():
        print(f"[COMFY][url] injected={patched} node_id={node_id_ref_url or 'auto'} ttl={ttl}")

    # Return with original wrapper shape
    return {"prompt": prompt_map} if not had_wrapper else {"prompt": prompt_map}


# ========= Optional convenience: legacy alias =========

async def generate_from_prompt(
    *,
    prompt: Dict[str, Any],
    out_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8188,
    max_wait_sec: float | int | None = None,
) -> List[Path]:
    """Alias for compatibility with callers that name the prompt map as 'prompt'."""
    return await generate_from_prompt_dict(
        prompt_dict=prompt,
        out_dir=out_dir,
        host=host,
        port=port,
        max_wait_sec=max_wait_sec,
    )
