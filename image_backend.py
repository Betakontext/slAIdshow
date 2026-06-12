from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional, Set, Any

import httpx
from pydantic import BaseModel, Field, ValidationError

# Important:
# The bridge must NOT hard-override width/height (e.g., force 512x512).
# It must send the provided prompt_dict as-is to ComfyUI (/prompt)
# and fetch results via /history. We will not pass width/height to it.
from comfyui_bridge import generate_from_prompt_dict


# ---------------------------
# Helper & Safety Utilities
# ---------------------------

def _env_str(k: str, d: str) -> str:
    """Read an environment variable as string with default, trimming whitespace."""
    return (os.getenv(k, d) or "").strip()

def _env_int(k: str, d: int) -> int:
    """Read an environment variable as int with default and safe fallback."""
    try:
        return int(os.getenv(k, str(d)))
    except Exception:
        return d

def _env_float(k: str, d: float) -> float:
    """Read an environment variable as float with default and safe fallback."""
    try:
        return float(os.getenv(k, str(d)))
    except Exception:
        return d

def _env_bool01(k: str, d: int = 0) -> bool:
    """Interpret common truthy strings (1, true, yes, on) as boolean True."""
    v = (os.getenv(k, str(d)) or "").strip().lower()
    return v in {"1", "true", "yes", "on"}

def _httpx_limits() -> httpx.Limits:
    """Shared connection pool limits for async HTTP clients."""
    return httpx.Limits(max_keepalive_connections=10, max_connections=20, keepalive_expiry=30.0)

def _timeout_short() -> httpx.Timeout:
    """Short timeout profile for metadata and probes."""
    return httpx.Timeout(connect=3.0, read=6.0, write=4.0, pool=4.0)

def _timeout_long(total: float) -> httpx.Timeout:
    """Long timeout profile for generation workflows, safely clamped."""
    total = max(10.0, min(total, 240.0))
    return httpx.Timeout(connect=8.0, read=total, write=8.0, pool=8.0)

def _assert_local_host(host: str) -> None:
    """Enforce localhost-only connectivity for privacy compliance."""
    if host not in {"127.0.0.1", "localhost"}:
        raise AssertionError(f"Only localhost allowed, got {host}")

def _clamp_dim(v: Optional[int]) -> Optional[int]:
    """Clamp dimensions to [64, 2048] and align to multiples of 8."""
    if v is None:
        return None
    x = max(64, min(2048, int(v)))
    return x - (x % 8)

def _now() -> float:
    """Current timestamp used to filter fresh files in output directories."""
    return time.time()


# ---------------------------
# Abstract Interface
# ---------------------------

class ImageBackend:
    """Abstract image backend interface."""
    async def generate(self, prompt: str, width: int | None = None, height: int | None = None) -> Path:
        raise NotImplementedError


# ---------------------------
# Optional Cloud Backend (Pollinations)
# ---------------------------

class _PollinationsV1Datum(BaseModel):
    b64_json: str
    revised_prompt: Optional[str] = None

class _PollinationsV1Response(BaseModel):
    created: int
    data: list[_PollinationsV1Datum]

class PollinationsConfig(BaseModel):
    """Config for the optional Pollinations cloud backend (disabled by default)."""
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
        """Convert seed string to int if provided, otherwise None."""
        if self.seed_raw is None:
            return None
        try:
            return int(self.seed_raw)
        except Exception:
            return None

    def require_cloud_enabled(self) -> None:
        """Ensure cloud usage is explicitly enabled and properly configured."""
        if not self.allow_cloud:
            raise RuntimeError("Cloud image backend not allowed (set ALLOW_CLOUD_IMAGE_BACKEND=1 to enable)")
        if not self.secret:
            raise RuntimeError("POLLINATIONS_SECRET missing in environment")

def _size_from_wh(width: int, height: int) -> str:
    """Return an API-friendly WxH string."""
    if width > 0 and height > 0:
        return f"{width}x{height}"
    return "1024x1024"

def _build_pollinations_image_url(api_base: str, prompt: str, model: Optional[str],
                                  width: Optional[int], height: Optional[int],
                                  nologo: bool, seed: Optional[int]) -> str:
    """Build URL for Pollinations GET-based generation endpoint."""
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
    """Cloud image generation (optional). Disabled by default for privacy reasons."""
    def __init__(self, out_dir: Path, cfg: Optional[PollinationsConfig] = None) -> None:
        self.out_dir = Path(out_dir).resolve()
        self.cfg = cfg or PollinationsConfig()
        self.out_dir.mkdir(parents=True, exist_ok=True)

    async def _fetch_v1(self, prompt: str, width: int | None, height: int | None) -> Path:
        """Use POST /v1/images/generations which returns base64-encoded images."""
        self.cfg.require_cloud_enabled()
        url = f"{self.cfg.api_base}/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {self.cfg.secret}",
            "Content-Type": "application/json",
        }
        w = width if (width and width > 0) else self.cfg.width
        h = height if (height and height > 0) else self.cfg.height
        payload = {
            "model": self.cfg.model or "flux",
            "prompt": prompt,
            "size": (self.cfg.size_override or _size_from_wh(w, h)),
        }
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
        """Use simple GET endpoint that streams a single image."""
        self.cfg.require_cloud_enabled()
        w = width if (width and width > 0) else self.cfg.width
        h = height if (height and height > 0) else self.cfg.height
        url = _build_pollinations_image_url(
            self.cfg.api_base, prompt, self.cfg.model, w, h, self.cfg.nologo, self.cfg.seed
        )
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

    async def generate(self, prompt: str, width: int | None = None, height: int | None = None) -> Path:
        """Public generate method with v1 try-first fallback to GET."""
        if not self.cfg.allow_cloud:
            raise RuntimeError("Cloud image backend disabled (ALLOW_CLOUD_IMAGE_BACKEND=0)")
        if self.cfg.use_v1:
            try:
                return await self._fetch_v1(prompt, width, height)
            except Exception:
                return await self._fetch_get(prompt, width, height)
        else:
            return await self._fetch_get(prompt, width, height)


# ---------------------------
# Local ComfyUI Backend
# ---------------------------

class ComfyConfig(BaseModel):
    """Runtime configuration for the local ComfyUI backend."""
    host: str = Field(default_factory=lambda: _env_str("APP_COMFY_HOST", "127.0.0.1"))
    port: int = Field(default_factory=lambda: _env_int("APP_COMFY_PORT", 8188))
    workflow_path: Path = Field(default_factory=lambda: Path(_env_str("APP_COMFY_WORKFLOW", "./workflows/text2img_any45.json")).resolve())
    width: int = Field(default_factory=lambda: _env_int("APP_COMFY_WIDTH", int(_env_str("APP_IMAGE_WIDTH", "512") or "512")))
    height: int = Field(default_factory=lambda: _env_int("APP_COMFY_HEIGHT", int(_env_str("APP_IMAGE_HEIGHT", "512") or "512")))
    steps: int = Field(default_factory=lambda: _env_int("APP_COMFY_STEPS", 20))
    cfg: float = Field(default_factory=lambda: _env_float("APP_COMFY_CFG", 6.5))
    sampler: str = Field(default_factory=lambda: _env_str("APP_COMFY_SAMPLER", "euler"))
    timeout_sec: float = Field(default_factory=lambda: _env_float("APP_COMFY_TIMEOUT_SEC", 180.0))
    disabled: bool = Field(default_factory=lambda: _env_bool01("APP_DISABLE_COMFYUI", 1))
    comfy_output_dir: Optional[Path] = Field(default_factory=lambda: (Path(_env_str("APP_COMFY_OUTPUT_DIR", "")).resolve() if _env_str("APP_COMFY_OUTPUT_DIR", "") else None))
    negative: str = Field(default_factory=lambda: _env_str("APP_COMFY_NEGATIVE", "text, watermark, logo, low quality, blurry, bad anatomy"))

    # Explicit node IDs for robust overrides (adjust via .env if your workflow differs)
    node_id_positive: str = Field(default_factory=lambda: _env_str("APP_COMFY_NODE_POS", "2"))
    node_id_negative: str = Field(default_factory=lambda: _env_str("APP_COMFY_NODE_NEG", "3"))
    node_id_latent: str = Field(default_factory=lambda: _env_str("APP_COMFY_NODE_LATENT", "4"))

    def assert_local(self) -> None:
        """Verify localhost-only policy."""
        _assert_local_host(self.host)

class LocalComfyBackend(ImageBackend):
    """
    Local ComfyUI backend:
    - Loads workflow JSON (prompt dict)
    - Enforces width/height on the latent node
    - Writes positive/negative prompts into CLIPTextEncode nodes
    - Sends prompt_dict as-is through the bridge and downloads results
    """
    def __init__(self, out_dir: Path, cfg: Optional[ComfyConfig] = None) -> None:
        self.out_dir = Path(out_dir).resolve()
        self.cfg = cfg or ComfyConfig()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.assert_local()
        self._samplers_cache: Optional[Set[str]] = None

    async def _available(self) -> bool:
        """Check if ComfyUI is reachable by calling /history."""
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
        """Query KSampler object info for available sampler_name choices."""
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
            # Static fallback to keep running even if object_info fails
            self._samplers_cache = {
                "euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_sde",
                "dpmpp_2m_karras", "heun", "dpm_fast", "uni_pc"
            }
        return self._samplers_cache

    def _normalize_sampler(self, name: str) -> str:
        """Normalize a few common alias names for samplers."""
        n = (name or "").strip().lower()
        if n in {"euler a", "euler_a", "euler-ancestral"}:
            return "euler_ancestral"
        return n

    def _load_prompt_file(self) -> dict:
        """Load the workflow JSON and return its 'prompt' dict (or the full dict)."""
        data = json.loads(self.cfg.workflow_path.read_text(encoding="utf-8"))
        if "prompt" in data and isinstance(data["prompt"], dict):
            return data["prompt"]
        return data

    def _override_text_nodes(self, prompt_dict: dict, positive: str, negative: str) -> None:
        """
        Robustly set texts in CLIPTextEncode nodes.
        Strategy:
        1) Target explicit node IDs (configured in env, defaults: pos='2', neg='3').
        2) Fallback to the first two CLIPTextEncode nodes if IDs are missing.
        """
        pos_set = False
        neg_set = False

        # Try configured positive node
        node_pos = prompt_dict.get(self.cfg.node_id_positive)
        if isinstance(node_pos, dict) and node_pos.get("class_type") == "CLIPTextEncode":
            inputs = node_pos.get("inputs")
            if isinstance(inputs, dict) and "text" in inputs:
                inputs["text"] = positive
                pos_set = True

        # Try configured negative node
        node_neg = prompt_dict.get(self.cfg.node_id_negative)
        if isinstance(node_neg, dict) and node_neg.get("class_type") == "CLIPTextEncode":
            inputs = node_neg.get("inputs")
            if isinstance(inputs, dict) and "text" in inputs:
                inputs["text"] = negative
                neg_set = True

        # Fallback: first two CLIPTextEncode nodes
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
        """
        Enforce width/height on the active latent source node.
        Strategy:
        1) Primary: target configured latent node ID (default '4').
        2) Fallback: update all known latent source classes to minimize mismatch.
        Note: We only modify the in-memory prompt_dict; the bridge sends it as-is.
        """
        # 1) Try explicit latent node ID
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
                    return  # Success: explicit target found and set

        # 2) Fallback: apply to all potential latent sources
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
            # Rarely workflows put width/height into KSampler; patch if present
            if cls.startswith("KSampler"):
                inputs = node.get("inputs")
                if isinstance(inputs, dict):
                    if "width" in inputs:
                        inputs["width"] = width
                    if "height" in inputs:
                        inputs["height"] = height

    def _copy_latest_from_comfy(self, since_ts: float) -> Optional[Path]:
        """
        Conservative fallback in case the bridge returns no files:
        - Only consider images created/modified after 'since_ts'
        - Copy the most recent candidate into our output directory
        """
        src_dir = self.cfg.comfy_output_dir
        if not src_dir or not src_dir.exists():
            return None
        candidates = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            candidates.extend(src_dir.rglob(ext))
        if not candidates:
            return None
        # Keep only fresh files (with a small slack)
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

    async def generate(self, prompt: str, width: int | None = None, height: int | None = None) -> Path:
        """
        Generate an image via ComfyUI:
        - Validate availability
        - Load workflow prompt dict
        - Override latent dimensions and text nodes
        - Submit prompt_dict via bridge (no extra width/height)
        - Return first resulting image path or use safe fallback
        """
        if self.cfg.disabled:
            raise RuntimeError("ComfyUI disabled (APP_DISABLE_COMFYUI=1)")
        if not self.cfg.workflow_path.exists():
            raise RuntimeError(f"workflow_not_found: {self.cfg.workflow_path}")
        ok = await self._available()
        if not ok:
            raise RuntimeError("comfy_unavailable")

        # Clamp UI-requested dimensions or use configured defaults
        w = _clamp_dim(width if (width and width > 0) else self.cfg.width)
        h = _clamp_dim(height if (height and height > 0) else self.cfg.height)

        # Validate/normalize sampler against ComfyUI choices (kept for future extension)
        valid_samplers = await self._fetch_valid_samplers()
        sampler = self._normalize_sampler(self.cfg.sampler)
        if sampler not in valid_samplers:
            sampler = "euler" if "euler" in valid_samplers else (next(iter(valid_samplers)) if valid_samplers else "euler")

        # Load and override prompt dict (dimensions and text)
        prompt_dict = self._load_prompt_file()
        self._override_dimensions_in_prompt(prompt_dict, width=w, height=h)
        self._override_text_nodes(prompt_dict, positive=(prompt or "").strip(), negative=(self.cfg.negative or "").strip())

        started_at = _now()

        # Critical: Do NOT pass width/height to the bridge; the dict overrides are authoritative.
        images = await generate_from_prompt_dict(
            prompt_dict=prompt_dict,
            out_dir=self.out_dir,
            positive_text=None,    # already applied in dict
            negative_text=None,    # already applied in dict
            width=None,            # avoid override collisions
            height=None,           # avoid override collisions
            steps=self.cfg.steps,  # if your bridge overwrites sampler/steps, set to None here
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

        # Fallback: copy the latest fresh file from ComfyUI's output dir (if configured)
        copied = self._copy_latest_from_comfy(since_ts=started_at)
        if copied:
            return copied

        raise RuntimeError("comfy_no_images")


# ---------------------------
# Factory
# ---------------------------

class BackendEnv(BaseModel):
    """Environment switches for backend selection and output directory."""
    image_backend: str = Field(default_factory=lambda: _env_str("IMAGE_BACKEND", "comfyui").lower())
    allow_cloud: bool = Field(default_factory=lambda: _env_bool01("ALLOW_CLOUD_IMAGE_BACKEND", 0))
    output_dir: Path = Field(default_factory=lambda: Path(_env_str("APP_OUTPUT_DIR", "./outputs/images")).resolve())

def build_image_backend() -> ImageBackend:
    """
    Backend factory:
    - comfyui (default): local privacy-preserving pipeline
    - pollinations: optional cloud backend (requires explicit env opt-in)
    """
    env = BackendEnv()
    out_dir = env.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if env.image_backend == "comfyui":
        cfg = ComfyConfig()
        return LocalComfyBackend(out_dir=out_dir, cfg=cfg)
    elif env.image_backend == "pollinations":
        cfg = PollinationsConfig()
        cfg.allow_cloud = env.allow_cloud
        return PollinationsBackend(out_dir=out_dir, cfg=cfg)
    else:
        raise RuntimeError(f"Unsupported IMAGE_BACKEND={env.image_backend}")
