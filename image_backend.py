from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Any

import httpx
from pydantic import BaseModel, Field, ValidationError

from comfyui_bridge import generate_from_prompt_dict

# --- Debug flag (optional minimal logging) ---
def _debug() -> bool:
    return (os.getenv("APP_IMAGE_BACKEND_DEBUG", "0") or "").strip().lower() in {"1", "true", "yes", "on"}

def _env_str(k: str, d: str) -> str:
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

def _httpx_limits() -> httpx.Limits:
    return httpx.Limits(max_keepalive_connections=10, max_connections=20, keepalive_expiry=30.0)

def _timeout_short() -> httpx.Timeout:
    return httpx.Timeout(connect=3.0, read=6.0, write=4.0, pool=4.0)

def _timeout_long(total: float) -> httpx.Timeout:
    total = max(10.0, min(total, 240.0))
    return httpx.Timeout(connect=8.0, read=total, write=8.0, pool=8.0)

def _clamp_dim(v: Optional[int]) -> Optional[int]:
    if v is None:
        return None
    x = max(64, min(2048, int(v)))
    return x - (x % 8)

def _now() -> float:
    return time.time()

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
        return
    if not _is_in_allowed_subnets(host, subnets):
        raise AssertionError(f"Remote host {host} not in allowed subnets ({subnets})")

def _clamp8(v: int) -> int:
    v = max(64, min(4096, int(v)))
    return v - (v % 8)

def _env_opt_int(name: str) -> Optional[int]:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None

def _resolve_size_for_backend(backend_name: str, req_w: Optional[int], req_h: Optional[int]) -> tuple[Optional[int], Optional[int]]:
    if isinstance(req_w, int) and req_w > 0 and isinstance(req_h, int) and req_h > 0:
        return _clamp8(req_w), _clamp8(req_h)
    gw = _env_opt_int("APP_IMAGE_WIDTH")
    gh = _env_opt_int("APP_IMAGE_HEIGHT")
    if gw and gh:
        return _clamp8(gw), _clamp8(gh)
    b = (backend_name or "").strip().lower()
    if b == "comfyui":
        cw = _env_opt_int("APP_COMFY_WIDTH")
        ch = _env_opt_int("APP_COMFY_HEIGHT")
        if cw and ch:
            return _clamp8(cw), _clamp8(ch)
    elif b == "pollinations":
        pw = _env_opt_int("POLLINATIONS_WIDTH")
        ph = _env_opt_int("POLLINATIONS_HEIGHT")
        if pw and ph:
            return _clamp8(pw), _clamp8(ph)
    return None, None

@dataclass
class StyleRuntime:
    reference_path: Optional[Path] = None
    reference_strength: float = 0.6
    @property
    def has_reference(self) -> bool:
        return self.reference_path is not None and self.reference_path.exists() and self.reference_path.is_file()

class ImageBackend:
    async def generate(self, prompt: str, width: int | None = None, height: int | None = None, negative_prompt: str | None = None) -> Path:
        raise NotImplementedError

class _PollinationsV1Datum(BaseModel):
    b64_json: str
    revised_prompt: Optional[str] = None

class _PollinationsV1Response(BaseModel):
    created: int
    data: list[_PollinationsV1Datum]

class PollinationsConfig(BaseModel):
    api_base: str = Field(default_factory=lambda: _env_str("POLLINATIONS_API_BASE", "https://gen.pollinations.ai").rstrip("/"))
    secret: str = Field(default_factory=lambda: _env_str("POLLINATIONS_SECRET", ""))
    model: Optional[str] = Field(default_factory=lambda: _env_str("POLLINATIONS_MODEL", "") or None)
    width: int = Field(default_factory=lambda: _env_int("POLLINATIONS_WIDTH", 1440))
    height: int = Field(default_factory=lambda: _env_int("POLLINATIONS_HEIGHT", 900))
    nologo: bool = Field(default_factory=lambda: _env_bool01("POLLINATIONS_NOLOGO", 1))
    seed_raw: Optional[str] = Field(default_factory=lambda: os.getenv("POLLINATIONS_SEED"))
    use_v1: bool = Field(default_factory=lambda: _env_bool01("POLLINATIONS_USE_V1", 1))
    size_override: str = Field(default_factory=lambda: _env_str("POLLINATIONS_SIZE", ""))
    allow_cloud: bool = Field(default_factory=lambda: _env_bool01("ALLOW_CLOUD_IMAGE_BACKEND", 0))
    @property
    def seed(self) -> Optional[int]:
        if self.seed_raw is None:
            return None
        try:
            return int(self.seed_raw)
        except Exception:
            return None
    def require_cloud_enabled(self) -> None:
        if not self.allow_cloud:
            raise RuntimeError("Cloud image backend not allowed (set ALLOW_CLOUD_IMAGE_BACKEND=1 to enable)")
        if not self.secret:
            raise RuntimeError("POLLINATIONS_SECRET missing in environment")

def _size_from_wh(width: int, height: int) -> str:
    if width > 0 and height > 0:
        return f"{width}x{height}"
    return "1024x1024"

def _build_pollinations_image_url(api_base: str, prompt: str, model: Optional[str],
                                  width: Optional[int], height: Optional[int],
                                  nologo: bool, seed: Optional[int]) -> str:
    from urllib.parse import quote, urlencode
    base = (api_base or "").rstrip("/")
    encoded_prompt = quote(prompt, safe="")
    url = f"{base}/image/{encoded_prompt}"
    params: dict[str, str] = {}
    if model:
        params["model"] = model
    if width and width > 0:
        params["width"] = str(width)
    if height and height > 0:
        params["height"] = str(height)
    if nologo:
        params["nologo"] = "true"
    if seed is not None:
        params["seed"] = str(seed)
    if params:
        url = f"{url}?{urlencode(params)}"
    return url

class PollinationsBackend(ImageBackend):
    def __init__(self, out_dir: Path, cfg: Optional[PollinationsConfig] = None) -> None:
        self.out_dir = Path(out_dir).resolve()
        self.cfg = cfg or PollinationsConfig()
        self.out_dir.mkdir(parents=True, exist_ok=True)

    async def _fetch_v1(self, prompt: str, width: int | None, height: int | None) -> Path:
        self.cfg.require_cloud_enabled()
        url = f"{self.cfg.api_base}/v1/images/generations"
        headers = {"Authorization": f"Bearer {self.cfg.secret}", "Content-Type": "application/json"}
        w = width if (width and width > 0) else self.cfg.width
        h = height if (height and height > 0) else self.cfg.height
        payload = {"model": self.cfg.model or "flux", "prompt": prompt, "size": (self.cfg.size_override or _size_from_wh(w, h))}
        timeout = _timeout_long(120.0)
        delay = 1.0
        last_exc: Optional[Exception] = None
        async with httpx.AsyncClient(timeout=timeout, limits=_httpx_limits()) as client:
            for attempt in range(1, 6):
                try:
                    r = await client.post(url, headers=headers, json=payload)
                    r.raise_for_status()
                    parsed = _PollinationsV1Response.model_validate(r.json())
                    if not parsed.data:
                        raise RuntimeError("pollinations_v1_empty_data")
                    from base64 import b64decode
                    raw = b64decode(parsed.data[0].b64_json, validate=True)
                    target = self.out_dir / f"img_{uuid.uuid4().hex}.jpg"
                    target.write_bytes(raw)
                    if target.stat().st_size < 1024:
                        raise RuntimeError("pollinations_v1_too_small")
                    return target
                except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError, httpx.HTTPStatusError, ValidationError) as e:
                    last_exc = e
                    if attempt < 5:
                        await asyncio.sleep(delay)
                        delay *= 1.7
                    continue
        raise RuntimeError(f"pollinations_v1_all_attempts_failed: {last_exc}")

    async def _fetch_get(self, prompt: str, width: int | None, height: int | None) -> Path:
        self.cfg.require_cloud_enabled()
        w = width if (width and width > 0) else self.cfg.width
        h = height if (height and height > 0) else self.cfg.height
        url = _build_pollinations_image_url(self.cfg.api_base, prompt, self.cfg.model, w, h, self.cfg.nologo, self.cfg.seed)
        params: dict[str, str] = {}
        if self.cfg.secret:
            params["key"] = self.cfg.secret
        timeout = _timeout_long(120.0)
        delay = 1.0
        last_exc: Optional[Exception] = None
        async with httpx.AsyncClient(timeout=timeout, limits=_httpx_limits(), follow_redirects=True) as client:
            for attempt in range(1, 5):
                try:
                    r = await client.get(url, params=params)
                    r.raise_for_status()
                    content = r.content
                    if not content or len(content) < 1024:
                        raise RuntimeError("pollinations_get_too_small")
                    target = self.out_dir / f"img_{uuid.uuid4().hex}.jpg"
                    target.write_bytes(content)
                    return target
                except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError, httpx.HTTPStatusError) as e:
                    last_exc = e
                    if attempt < 4:
                        await asyncio.sleep(delay)
                        delay *= 1.7
                    continue
        raise RuntimeError(f"pollinations_get_all_attempts_failed: {last_exc}")

    async def generate(self, prompt: str, width: int | None = None, height: int | None = None, negative_prompt: str | None = None) -> Path:
        if not self.cfg.allow_cloud:
            raise RuntimeError("Cloud image backend disabled (ALLOW_CLOUD_IMAGE_BACKEND=0)")
        full_prompt = (prompt or "").strip()
        if negative_prompt:
            n = negative_prompt.strip()
            if n:
                full_prompt = f"{full_prompt}\n-- negative: {n}"
        rw, rh = _resolve_size_for_backend("pollinations", width, height)
        eff_w = rw if (rw and rw > 0) else (width if (width and width > 0) else self.cfg.width)
        eff_h = rh if (rh and rh > 0) else (height if (height and height > 0) else self.cfg.height)
        if self.cfg.use_v1:
            try:
                return await self._fetch_v1(full_prompt, eff_w, eff_h)
            except Exception:
                return await self._fetch_get(full_prompt, eff_w, eff_h)
        else:
            return await self._fetch_get(full_prompt, eff_w, eff_h)

class ComfyConfig(BaseModel):
    host: str = Field(default_factory=lambda: _env_str("APP_COMFY_HOST", "127.0.0.1"))
    port: int = Field(default_factory=lambda: _env_int("APP_COMFY_PORT", 8188))
    workflow_path: Path = Field(default_factory=lambda: Path(_env_str("APP_COMFY_WORKFLOW", "./workflows/text2img_SD15-FP16.json")).resolve())
    width: int = Field(default_factory=lambda: _env_int("APP_COMFY_WIDTH", int(_env_str("APP_IMAGE_WIDTH", "512") or "512")))
    height: int = Field(default_factory=lambda: _env_int("APP_COMFY_HEIGHT", int(_env_str("APP_IMAGE_HEIGHT", "512") or "512")))
    steps: int = Field(default_factory=lambda: _env_int("APP_COMFY_STEPS", 20))
    cfg: float = Field(default_factory=lambda: _env_float("APP_COMFY_CFG", 6.5))
    sampler: str = Field(default_factory=lambda: _env_str("APP_COMFY_SAMPLER", "euler"))
    timeout_sec: float = Field(default_factory=lambda: _env_float("APP_COMFY_TIMEOUT_SEC", 180.0))
    disabled: bool = Field(default_factory=lambda: _env_bool01("APP_DISABLE_COMFYUI", 1))
    comfy_output_dir: Optional[Path] = Field(default_factory=lambda: (Path(_env_str("APP_COMFY_OUTPUT_DIR", "")).resolve() if _env_str("APP_COMFY_OUTPUT_DIR", "") else None))
    negative: str = Field(default_factory=lambda: _env_str("APP_COMFY_NEGATIVE", "text, watermark, logo, low quality, blurry, bad anatomy"))

    node_id_positive: str = Field(default_factory=lambda: _env_str("APP_COMFY_NODE_POS", "2"))
    node_id_negative: str = Field(default_factory=lambda: _env_str("APP_COMFY_NODE_NEG", "3"))
    node_id_latent: str = Field(default_factory=lambda: _env_str("APP_COMFY_NODE_LATENT", "4"))

    node_id_ref_image: Optional[str] = Field(default_factory=lambda: (_env_str("APP_COMFY_NODE_REF_IMAGE", "") or None))
    node_id_ipadapter: Optional[str] = Field(default_factory=lambda: (_env_str("APP_COMFY_NODE_IPADAPTER", "") or None))
    node_key_ref_image_path: str = Field(default_factory=lambda: _env_str("APP_COMFY_KEY_REF_IMAGE_PATH", "image"))
    node_key_ref_weight: str = Field(default_factory=lambda: _env_str("APP_COMFY_KEY_REF_WEIGHT", "weight"))

    def assert_local(self) -> None:
        _assert_image_backend_host_policy(self.host)

class LocalComfyBackend(ImageBackend):
    def __init__(self, out_dir: Path, cfg: Optional[ComfyConfig] = None, style: Optional[StyleRuntime] = None) -> None:
        self.out_dir = Path(out_dir).resolve()
        self.cfg = cfg or ComfyConfig()
        self.style = style or StyleRuntime()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.assert_local()
        self._samplers_cache: Optional[Set[str]] = None

    def set_style_runtime(self, style: Optional[StyleRuntime]) -> None:
        self.style = style or StyleRuntime()

    async def _available(self) -> bool:
        if self.cfg.disabled:
            return False
        try:
            async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_short()) as c:
                r = await c.get(f"http://{self.cfg.host}:{self.cfg.port}/history")
                r.raise_for_status()
                return True
        except Exception:
            return False

    async def _fetch_valid_samplers(self) -> Set[str]:
        if self._samplers_cache is not None:
            return self._samplers_cache
        url = f"http://{self.cfg.host}:{self.cfg.port}/object_info/KSampler"
        try:
            async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_short()) as c:
                r = await c.get(url)
                r.raise_for_status()
                j = r.json()
                choices = j.get("input", {}).get("sampler_name", {}).get("choices", [])
                if isinstance(choices, list):
                    self._samplers_cache = {str(x) for x in choices}
                else:
                    self._samplers_cache = set()
        except Exception:
            self._samplers_cache = {
                "euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_sde",
                "dpmpp_2m_karras", "heun", "dpm_fast", "uni_pc"
            }
        return self._samplers_cache

    def _normalize_sampler(self, name: str) -> str:
        n = (name or "").strip().lower()
        if n in {"euler a", "euler_a", "euler-ancestral"}:
            return "euler_ancestral"
        return n

    def _load_prompt_file(self) -> dict:
        data = json.loads(self.cfg.workflow_path.read_text(encoding="utf-8"))
        if "prompt" in data and isinstance(data["prompt"], dict):
            return data["prompt"]
        return data

    def _override_text_nodes(self, prompt_dict: dict, positive: str, negative: str) -> None:
        pos_set = False
        neg_set = False
        node_pos = prompt_dict.get(self.cfg.node_id_positive)
        if isinstance(node_pos, dict) and node_pos.get("class_type") == "CLIPTextEncode":
            inputs = node_pos.get("inputs")
            if isinstance(inputs, dict) and "text" in inputs:
                inputs["text"] = positive
                pos_set = True
        node_neg = prompt_dict.get(self.cfg.node_id_negative)
        if isinstance(node_neg, dict) and node_neg.get("class_type") == "CLIPTextEncode":
            inputs = node_neg.get("inputs")
            if isinstance(inputs, dict) and "text" in inputs:
                inputs["text"] = negative
                neg_set = True
        if not (pos_set and neg_set):
            clip_nodes = []
            for node in prompt_dict.values():
                if isinstance(node, dict) and node.get("class_type") == "CLIPTextEncode":
                    clip_nodes.append(node)
            if clip_nodes and not pos_set:
                inputs = clip_nodes[0].get("inputs", {})
                if isinstance(inputs, dict) and "text" in inputs:
                    inputs["text"] = positive
                    pos_set = True
            if len(clip_nodes) > 1 and not neg_set:
                inputs = clip_nodes[1].get("inputs", {})
                if isinstance(inputs, dict) and "text" in inputs:
                    inputs["text"] = negative
                    neg_set = True

    def _override_dimensions_in_prompt(self, prompt_dict: dict, width: int, height: int) -> None:
        node_latent = prompt_dict.get(self.cfg.node_id_latent)
        if isinstance(node_latent, dict):
            cls = str(node_latent.get("class_type") or node_latent.get("class", "")).strip()
            if cls in {"EmptyLatentImage", "EmptyLatentImageBatch", "LatentImage", "CreateLatentImage"}:
                inputs = node_latent.get("inputs")
                if isinstance(inputs, dict):
                    if "width" in inputs:
                        inputs["width"] = width
                    if "height" in inputs:
                        inputs["height"] = height
                    return
        for node in prompt_dict.values():
            if not isinstance(node, dict):
                continue
            cls = str(node.get("class_type") or node.get("class", "")).strip()
            if cls in {"EmptyLatentImage", "EmptyLatentImageBatch", "LatentImage", "CreateLatentImage"}:
                inputs = node.get("inputs")
                if isinstance(inputs, dict):
                    if "width" in inputs:
                        inputs["width"] = width
                    if "height" in inputs:
                        inputs["height"] = height
            if cls.startswith("KSampler"):
                inputs = node.get("inputs")
                if isinstance(inputs, dict):
                    if "width" in inputs:
                        inputs["width"] = width
                    if "height" in inputs:
                        inputs["height"] = height

    def _inject_style_reference(self, prompt_dict: dict) -> None:
        if not (self.style and self.style.has_reference):
            return
        ref_path = str(self.style.reference_path.as_posix())
        weight = float(max(0.0, min(1.0, self.style.reference_strength)))
        if _debug():
            print(f"[COMFY][style] apply ref={ref_path} w={weight}")
        if self.cfg.node_id_ref_image:
            node = prompt_dict.get(self.cfg.node_id_ref_image)
            if isinstance(node, dict):
                inputs = node.get("inputs")
                if isinstance(inputs, dict):
                    key = self.cfg.node_key_ref_image_path or "image"
                    if key in inputs:
                        inputs[key] = ref_path
        if self.cfg.node_id_ipadapter:
            node = prompt_dict.get(self.cfg.node_id_ipadapter)
            if isinstance(node, dict):
                inputs = node.get("inputs")
                if isinstance(inputs, dict):
                    k = self.cfg.node_key_ref_weight or "weight"
                    if k in inputs:
                        inputs[k] = weight

    def _copy_latest_from_comfy(self, since_ts: float) -> Optional[Path]:
        src_dir = self.cfg.comfy_output_dir
        if not src_dir or not src_dir.exists():
            return None
        candidates = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            candidates.extend(src_dir.rglob(ext))
        if not candidates:
            return None
        recent = [p for p in candidates if p.is_file() and p.stat().st_mtime >= since_ts - 0.8]
        if not recent:
            return None
        latest = max(recent, key=lambda p: p.stat().st_mtime)
        target = self.out_dir / f"img_{uuid.uuid4().hex}{latest.suffix.lower()}"
        try:
            shutil.copy2(latest, target)
            if target.stat().st_size < 1024:
                return None
            return target
        except Exception:
            return None

    async def generate(self, prompt: str, width: int | None = None, height: int | None = None, negative_prompt: str | None = None) -> Path:
        if self.cfg.disabled:
            raise RuntimeError("ComfyUI disabled (APP_DISABLE_COMFYUI=1)")
        if not self.cfg.workflow_path.exists():
            raise RuntimeError(f"workflow_not_found: {self.cfg.workflow_path}")
        ok = await self._available()
        if not ok:
            raise RuntimeError("comfy_unavailable")

        rw, rh = _resolve_size_for_backend("comfyui", width, height)
        w_eff = rw if (rw and rw > 0) else (width if (width and width > 0) else self.cfg.width)
        h_eff = rh if (rh and rh > 0) else (height if (height and height > 0) else self.cfg.height)
        w = _clamp_dim(w_eff)
        h = _clamp_dim(h_eff)

        valid_samplers = await self._fetch_valid_samplers()
        sampler = self._normalize_sampler(self.cfg.sampler)
        if sampler not in valid_samplers:
            sampler = "euler" if "euler" in valid_samplers else (next(iter(valid_samplers)) if valid_samplers else "euler")

        prompt_dict = self._load_prompt_file()
        self._override_dimensions_in_prompt(prompt_dict, width=w, height=h)

        neg_eff = (negative_prompt or self.cfg.negative or "").strip()
        self._override_text_nodes(prompt_dict, positive=(prompt or "").strip(), negative=neg_eff)

        self._inject_style_reference(prompt_dict)

        started_at = _now()

        images = await generate_from_prompt_dict(
            prompt_dict=prompt_dict,
            out_dir=self.out_dir,
            positive_text=None,
            negative_text=None,
            width=None,
            height=None,
            steps=self.cfg.steps,
            cfg=self.cfg.cfg,
            sampler_name=sampler,
            scheduler=None,
            denoise=None,
            seed=None,
            host=self.cfg.host,
            port=self.cfg.port,
            max_wait_sec=float(self.cfg.timeout_sec),
            poll_interval=1.0,
        )

        if images:
            return images[0]

        copied = self._copy_latest_from_comfy(since_ts=started_at)
        if copied:
            return copied

        raise RuntimeError("comfy_no_images")

class BackendEnv(BaseModel):
    image_backend: str = Field(default_factory=lambda: _env_str("IMAGE_BACKEND", "comfyui").lower())
    allow_cloud: bool = Field(default_factory=lambda: _env_bool01("ALLOW_CLOUD_IMAGE_BACKEND", 0))
    output_dir: Path = Field(default_factory=lambda: Path(_env_str("APP_OUTPUT_DIR", "./outputs/images")).resolve())

def build_image_backend(style: Optional[StyleRuntime] = None) -> ImageBackend:
    env = BackendEnv()
    out_dir = env.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if env.image_backend == "comfyui":
        cfg = ComfyConfig()
        return LocalComfyBackend(out_dir=out_dir, cfg=cfg, style=style)
    elif env.image_backend == "pollinations":
        cfg = PollinationsConfig()
        cfg.allow_cloud = env.allow_cloud
        return PollinationsBackend(out_dir=out_dir, cfg=cfg)
    else:
        raise RuntimeError(f"Unsupported IMAGE_BACKEND={env.image_backend}")
