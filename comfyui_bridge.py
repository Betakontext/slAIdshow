from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        if self.host not in {"127.0.0.1", "localhost"}:
            raise AssertionError(f"Only localhost allowed, got {self.host}")
        return f"http://{self.host}:{self.port}"

def _limits() -> httpx.Limits:
    # Conservative connection pool for local service
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

# -----------------------------
# Prompt manipulation
# -----------------------------

def _ensure_api_prompt_dict(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure the payload has a top-level 'prompt' dict in ComfyUI API format.
    - If body already is the prompt dict (mapping node_id -> node), wrap it.
    - If body has 'prompt' as dict, return as is.
    """
    if isinstance(body, dict) and "prompt" in body and isinstance(body["prompt"], dict):
        return body
    # Heuristic: if the root is a mapping with numeric-like keys and each value has 'class_type'
    if all(
        isinstance(k, str) and isinstance(v, dict) and "class_type" in v
        for k, v in body.items()
    ):
        return {"prompt": body}
    raise RuntimeError("invalid_prompt_format: expected a 'prompt' dict or a node mapping")

def _get_node(prompt: Dict[str, Any], node_id: str) -> Optional[Dict[str, Any]]:
    n = prompt.get(node_id)
    return n if isinstance(n, dict) else None

def _set_input(node: Dict[str, Any], key: str, value: Any) -> None:
    # Modify 'inputs' map only if it exists; do not invent structure
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
            _set_input(n2, "text", positive_text)
    if negative_text is not None:
        n3 = _get_node(prompt, node_id_negative)
        if n3:
            _set_input(n3, "text", negative_text)

    # Latent size (EmptyLatentImage)
    w = _clamp_dim(width) if width is not None else None
    h = _clamp_dim(height) if height is not None else None
    if w is not None or h is not None:
        n4 = _get_node(prompt, node_id_latent)
        if n4:
            if w is not None:
                _set_input(n4, "width", int(w))
            if h is not None:
                _set_input(n4, "height", int(h))

    # Sampler params
    n5 = _get_node(prompt, node_id_ksampler)
    if n5:
        if steps is not None:
            _set_input(n5, "steps", int(steps))
        if cfg is not None:
            _set_input(n5, "cfg", float(cfg))
        if sampler_name is not None:
            _set_input(n5, "sampler_name", sampler_name)
        if scheduler is not None:
            _set_input(n5, "scheduler", scheduler)
        if denoise is not None:
            _set_input(n5, "denoise", float(denoise))
        if seed is not None:
            _set_input(n5, "seed", int(seed))

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
        except Exception as e:
            # Do not retry on schema errors/400 diagnostics
            raise
    raise RuntimeError(f"comfy_post_prompt_failed: {last_exc}")

async def _poll_history_ready(
    client: httpx.AsyncClient,
    base_url: str,
    prompt_id: str,
    max_wait_sec: float = 150.0,
    poll_interval: float = 1.0,
) -> Dict[str, Any]:
    """
    Poll /history/{prompt_id} until outputs are present or timeout is reached.
    Returns the history JSON object.
    """
    deadline = time.time() + max_wait_sec
    while time.time() < deadline:
        r = await client.get(f"{base_url}/history/{prompt_id}")
        if r.status_code == 404:
            await asyncio.sleep(poll_interval)
            continue
        r.raise_for_status()
        j = r.json()
        # Some ComfyUI builds return {prompt_id: {...}}, others return {...} directly
        if isinstance(j, dict):
            if prompt_id in j:
                node_map = j[prompt_id]
            else:
                node_map = j
            # Heuristic: if any node has 'images' in its outputs, we are done
            for _, node in (node_map or {}).items():
                outs = (node or {}).get("outputs") or {}
                if any(isinstance(v, dict) and "images" in v for v in outs.values()):
                    return j
        await asyncio.sleep(poll_interval)
    raise TimeoutError("comfy_history_timeout")

async def _download_images(
    client: httpx.AsyncClient,
    base_url: str,
    history_obj: Dict[str, Any],
    out_dir: Path,
) -> List[Path]:
    """
    Scan history for image outputs and download them via /view/{type}/{subfolder}/{filename}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    def iter_image_descriptors() -> List[Dict[str, Any]]:
        # Flatten all outputs that contain 'images'
        result: List[Dict[str, Any]] = []
        node_maps = [history_obj]
        # Some builds wrap by {prompt_id: node_map}
        for key, v in list(history_obj.items()):
            if isinstance(v, dict) and all(isinstance(x, dict) for x in v.values()):
                node_maps.append(v)
        for node_map in node_maps:
            for _, node in node_map.items():
                outs = (node or {}).get("outputs") or {}
                for _, outv in outs.items():
                    if isinstance(outv, dict) and "images" in outv and isinstance(outv["images"], list):
                        result.append(outv)
        return result

    for outv in iter_image_descriptors():
        for item in outv.get("images", []):
            if not isinstance(item, dict):
                continue
            filename = item.get("filename")
            subfolder = item.get("subfolder", "")
            folder_type = item.get("type", "output")
            if not filename:
                continue
            # Build /view URL
            view_path = f"/view/{folder_type}"
            if subfolder:
                # Ensure no accidental double slashes
                subfolder = str(subfolder).strip("/")
                view_path += f"/{subfolder}"
            view_path += f"/{filename}"
            url = f"{base_url}{view_path}"

            # Download with small retry
            delay = 0.8
            for attempt in range(1, 5):
                try:
                    r = await client.get(url)
                    r.raise_for_status()
                    data = r.content
                    if not data or len(data) < 1024:
                        raise RuntimeError("image_download_too_small")
                    suffix = Path(filename).suffix or ".png"
                    target = out_dir / f"img_{uuid.uuid4().hex}{suffix}"
                    target.write_bytes(data)
                    saved.append(target)
                    break
                except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError, httpx.HTTPStatusError):
                    if attempt < 4:
                        await asyncio.sleep(delay)
                        delay *= 1.7
                        continue
                    raise
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
        images = await _download_images(client, base, history_obj, out_dir=Path(out_dir))
    return images
