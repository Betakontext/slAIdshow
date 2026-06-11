#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable

import httpx
from pydantic import BaseModel, Field


# -----------------------------
# Connection / helpers
# -----------------------------

class ComfyConnection(BaseModel):
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8188)

    @property
    def base(self) -> str:
        # Restrict to localhost for privacy/safety
        if self.host not in {"127.0.0.1", "localhost"}:
            raise AssertionError(f"Only localhost allowed, got {self.host}")
        return f"http://{self.host}:{self.port}"


def _limits() -> httpx.Limits:
    # Reasonable connection pool for a local service
    return httpx.Limits(max_keepalive_connections=8, max_connections=16, keepalive_expiry=30.0)


def _timeout(total: float = 150.0) -> httpx.Timeout:
    # Total read timeout is the dominant factor for image generation
    total = max(30.0, min(total, 300.0))
    return httpx.Timeout(connect=5.0, read=total, write=8.0, pool=8.0)


def _clamp_dim(v: Optional[int]) -> Optional[int]:
    # Keep image dims in a safe range; many nodes require multiples of 8
    if v is None:
        return None
    x = int(v)
    x = max(64, min(2048, x))
    return x - (x % 8)


def _env_str(k: str, d: str = "") -> str:
    return (os.getenv(k, d) or "").strip()


# Optional filesystem fallback when /view is unavailable
_COMFY_OUTPUT_DIR: Optional[Path] = Path(_env_str("APP_COMFY_OUTPUT_DIR", "")).resolve() if _env_str("APP_COMFY_OUTPUT_DIR", "") else None


# -----------------------------
# Prompt manipulation
# -----------------------------

def _ensure_api_prompt_dict(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure the payload has a top-level 'prompt' dict in ComfyUI API format.
    - If body already contains 'prompt' as a dict, return as-is.
    - If body itself is a node mapping (id -> node with 'class_type'), wrap it into {'prompt': body}.
    - Otherwise raise for invalid shape.
    """
    if isinstance(body, dict) and "prompt" in body and isinstance(body["prompt"], dict):
        return body
    if isinstance(body, dict) and body:
        # Heuristic: all keys are strings and all values are dicts with 'class_type'
        if all(isinstance(k, str) and isinstance(v, dict) and "class_type" in v for k, v in body.items()):
            return {"prompt": body}
    raise RuntimeError("invalid_prompt_format: expected a 'prompt' dict or a node mapping")


def _get_node(prompt: Dict[str, Any], node_id: str) -> Optional[Dict[str, Any]]:
    n = prompt.get(node_id)
    return n if isinstance(n, dict) else None


def _set_input_if_present(node: Dict[str, Any], key: str, value: Any) -> None:
    """
    Modify 'inputs' map only if it exists and contains the key.
    This avoids creating invalid node structures.
    """
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
    # Node IDs for your shared workflow (adjust if your workflow differs)
    node_id_positive: str = "2",
    node_id_negative: str = "3",
    node_id_latent: str = "4",
    node_id_ksampler: str = "5",
) -> Dict[str, Any]:
    """
    Apply safe in-place overrides for the given API prompt structure:
    - text on CLIPTextEncode (positive/negative)
    - width/height on EmptyLatentImage
    - steps/cfg/sampler/scheduler/denoise/seed on KSampler
    The function is idempotent and only sets keys that already exist in the node's inputs.
    """
    payload = _ensure_api_prompt_dict(body)
    prompt: Dict[str, Any] = payload["prompt"]

    # Positive/negative prompts
    if positive_text is not None:
        n2 = _get_node(prompt, node_id_positive)
        if n2:
            _set_input_if_present(n2, "text", positive_text)
    if negative_text is not None:
        n3 = _get_node(prompt, node_id_negative)
        if n3:
            _set_input_if_present(n3, "text", negative_text)

    # Latent size (EmptyLatentImage)
    w = _clamp_dim(width) if width is not None else None
    h = _clamp_dim(height) if height is not None else None
    if w is not None or h is not None:
        n4 = _get_node(prompt, node_id_latent)
        if n4:
            if w is not None:
                _set_input_if_present(n4, "width", int(w))
            if h is not None:
                _set_input_if_present(n4, "height", int(h))

    # Sampler params
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
    Post to /prompt with retry on transient failures.
    Provides short diagnostic on HTTP 400 (bad request), including first 300 chars of response.
    """
    delay = 0.8
    last_exc: Optional[Exception] = None
    for attempt in range(1, 5):
        try:
            r = await client.post(f"{base_url}/prompt", json=body)
            if r.status_code == 400:
                # Include short server message for easier debugging
                text = r.text[:300].replace("\n", " ")
                raise RuntimeError(f"comfy_400: {text}")
            r.raise_for_status()
            j = r.json()
            if not isinstance(j, dict):
                raise RuntimeError(f"comfy_prompt_non_object: {type(j)}")
            pid = j.get("prompt_id") or j.get("promptId") or j.get("id")
            if not pid:
                raise RuntimeError("comfy_no_prompt_id")
            return str(pid)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError, httpx.HTTPStatusError) as e:
            last_exc = e
            if attempt < 4:
                await asyncio.sleep(delay)
                delay *= 1.7
                continue
            break
        except Exception:
            # Do not retry on schema errors/400 diagnostics
            raise
    raise RuntimeError(f"comfy_post_prompt_failed: {last_exc}")


def _node_maps_from_history_obj(history_json: Dict[str, Any], prompt_id: str) -> List[Dict[str, Any]]:
    """
    Normalize different history response shapes into a list of node maps:
    - Some builds return { "<prompt_id>": { ... node map or entry ... } }
    - Others may return the node map directly.
    We collect plausible node maps (dicts whose values are dicts).
    """
    node_maps: List[Dict[str, Any]] = []

    # Direct map possibility
    if isinstance(history_json, dict) and all(isinstance(v, dict) for v in history_json.values()):
        node_maps.append(history_json)

    # Wrapped by prompt_id
    entry = history_json.get(prompt_id)
    if isinstance(entry, dict) and all(isinstance(v, dict) for v in entry.values()):
        node_maps.append(entry)

    # Some responses may nest an "outputs" layer within the entry
    if isinstance(entry, dict):
        outputs = entry.get("outputs")
        if isinstance(outputs, dict) and all(isinstance(v, dict) for v in outputs.values()):
            node_maps.append(outputs)

    # De-duplicate while preserving order
    seen_ids = set()
    deduped: List[Dict[str, Any]] = []
    for m in node_maps:
        if id(m) in seen_ids:
            continue
        seen_ids.add(id(m))
        deduped.append(m)
    return deduped


def _any_images_in_node_maps(node_maps: Iterable[Dict[str, Any]]) -> bool:
    """
    Return True if any node map contains an output with an 'images' list.
    """
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
    Poll /history/{prompt_id} until at least one output contains a non-empty 'images' list.
    Returns the history JSON object.
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

    raise TimeoutError(f"comfy_history_timeout (last_payload_keys={list(last_payload.keys()) if isinstance(last_payload, dict) else 'n/a'})")


def _iter_image_descriptors_from_history(history_obj: Dict[str, Any], prompt_id: str) -> Iterable[Dict[str, Any]]:
    """
    Yield each 'output' dict that contains an 'images' list from the normalized node maps.
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


async def _download_via_view(
    client: httpx.AsyncClient,
    base_url: str,
    folder_type: str,
    subfolder: str,
    filename: str,
) -> bytes:
    """
    Download a single image via ComfyUI /view endpoint.
    Uses path style: /view/{type}/{subfolder?}/{filename}
    """
    # Build /view path defensively
    path = f"/view/{folder_type}"
    subfolder = (subfolder or "").strip().strip("/")
    if subfolder:
        path += f"/{subfolder}"
    path += f"/{filename}"
    url = f"{base_url}{path}"
    r = await client.get(url)
    r.raise_for_status()
    return r.content


def _fs_fallback_read(folder_type: str, subfolder: str, filename: str) -> Optional[bytes]:
    """
    Optional filesystem fallback using APP_COMFY_OUTPUT_DIR when /view is unavailable.
    Only applies for type='output' by default ComfyUI behavior.
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
) -> List[Path]:
    """
    Scan history for image outputs and download them via /view,
    with a filesystem fallback when configured.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    for outv in _iter_image_descriptors_from_history(history_obj, prompt_id):
        images = outv.get("images", [])
        if not isinstance(images, list) or not images:
            continue
        for item in images:
            if not isinstance(item, dict):
                continue
            filename = item.get("filename")
            subfolder = item.get("subfolder", "")
            folder_type = item.get("type", "output")
            if not filename or not isinstance(filename, str):
                continue

            # Try to download with small retry loop
            delay = 0.8
            data: Optional[bytes] = None
            for attempt in range(1, 5):
                try:
                    # Prefer /view
                    data = await _download_via_view(client, base_url, folder_type, subfolder, filename)
                    if not data or len(data) < 256:  # sanity check
                        raise RuntimeError("image_download_too_small")
                    break
                except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError, httpx.HTTPStatusError):
                    if attempt < 4:
                        await asyncio.sleep(delay)
                        delay *= 1.7
                        continue
                    # last attempt failed -> try fs fallback
                except Exception:
                    # fall through to fs fallback below
                    pass

            if data is None:
                data = _fs_fallback_read(folder_type, subfolder, filename)

            if data is None:
                # Could not obtain this image; continue with others
                continue

            suffix = Path(filename).suffix or ".png"
            target = out_dir / f"img_{uuid.uuid4().hex}{suffix}"
            target.write_bytes(data)
            saved.append(target)

    return saved


# -----------------------------
# Public entry point
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
) -> List[Path]:
    """
    Submit a ready-to-use ComfyUI API prompt dict (mapping node_id -> node).
    Applies safe in-place overrides for text/size/sampler params, then posts to /prompt and downloads images.
    Returns a list of saved Paths in out_dir.
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

    async with httpx.AsyncClient(limits=_limits(), timeout=_timeout(max_wait_sec + 30)) as client:
        prompt_id = await _post_prompt(client, base, payload)
        history_obj = await _poll_history_ready(client, base, prompt_id, max_wait_sec=max_wait_sec, poll_interval=poll_interval)
        images = await _download_images(client, base, history_obj, prompt_id, out_dir=Path(out_dir))
    return images
