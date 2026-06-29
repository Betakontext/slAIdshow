# image_backend.py
# -*- coding: utf-8 -*-
# Production-ready image backend adapters for ComfyUI (local/LAN/remote) and Pollinations (cloud)
#
# This release keeps Pollinations fully decoupled and working as before, while adding
# an opt-in, safe auto-discovery for ComfyUI that:
#   - Prefers the configured APP_COMFY_HOST first (default 127.0.0.1)
#   - Optionally tries APP_COMFY_FALLBACK_HOSTS and AUTO_DISCOVERY_SUBNETS if local is down
#   - Enforces APP_ALLOW_REMOTE_BACKENDS and APP_ALLOWED_SUBNETS before connecting
#   - Caches the first discovered host for the process lifetime (no repeated scans)
#
# IMPORTANT:
# - Pollinations path never triggers ComfyUI discovery and is not affected by it.
# - ComfyUI discovery only runs if IMAGE_BACKEND=comfyui and APP_DISABLE_COMFYUI=0.
# - 0.0.0.0 is a bind address only; clients must use a real IP like 192.168.188.24.
#
# Minimal .env for LAN ComfyUI usage:
#   IMAGE_BACKEND=comfyui
#   APP_COMFY_HOST=127.0.0.1
#   APP_COMFY_PORT=8188
#   APP_ALLOW_REMOTE_BACKENDS=1
#   APP_ALLOWED_SUBNETS=192.168.188.0/24
#   APP_COMFY_FALLBACK_HOSTS=192.168.188.24
#   AUTO_DISCOVERY_ENABLE=1
#   AUTO_DISCOVERY_SUBNETS=192.168.188.0/24
#
# To use Pollinations (cloud):
#   IMAGE_BACKEND=pollinations
#   ALLOW_CLOUD_IMAGE_BACKEND=1
#   POLLINATIONS_SECRET=sk_...
#   (Discovery code is not touched in this path.)

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional, Set, Tuple, List, Dict

import httpx
from pydantic import BaseModel, Field, ValidationError

from comfyui_bridge import generate_from_prompt_dict


# ---------------------------
# Env helpers & HTTP config
# ---------------------------

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
    # General quick calls (health/object_info)
    return httpx.Timeout(connect=2.5, read=5.0, write=4.0, pool=4.0)

def _timeout_probe() -> httpx.Timeout:
    # Very short probes for discovery
    return httpx.Timeout(connect=1.2, read=2.0, write=1.5, pool=1.5)

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

def _is_loopback(host: str) -> bool:
    return host in {"127.0.0.1", "localhost"}

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
    Image backend host policy:
    - Always allow loopback.
    - Remote only if APP_ALLOW_REMOTE_BACKENDS=1.
    - If APP_ALLOWED_SUBNETS is set and host is an IP, it must match.
    """
    if _is_loopback(host):
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


# ---------------------------
# Unified size resolution
# ---------------------------

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
    # 1) Request overrides
    if isinstance(req_w, int) and req_w > 0 and isinstance(req_h, int) and req_h > 0:
        return _clamp8(req_w), _clamp8(req_h)

    # 2) Global
    gw = _env_opt_int("APP_IMAGE_WIDTH")
    gh = _env_opt_int("APP_IMAGE_HEIGHT")
    if gw and gh:
        return _clamp8(gw), _clamp8(gh)

    # 3) Backend-specific
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


# ---------------------------
# Abstract Interface
# ---------------------------

class ImageBackend:
    async def generate(self, prompt: str, width: int | None = None, height: int | None = None, negative: str | None = None) -> Path:
        raise NotImplementedError


# ============================================================
# Pollinations (Cloud) with Keyword-Style Negative Injection
# ============================================================

INJECTION_MODE = "append"        # 'append' or 'prepend'
INJECTION_SEPARATOR = " "
MAX_NEG_TERMS = 24
MAX_CONSTRAINT_CHARS = 240
DEFAULT_COLOR_BIAS = True

def _is_german_text(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    if any(ch in lower for ch in ("ä", "ö", "ü", "ß")):
        return True
    german_markers = {
        " der ", " die ", " das ", " und ", " mit ", " ohne ", " kein ", " keine ", " einem ",
        " einer ", " im ", " am ", " zum ", " zur ", " vom ", " für ", " nicht ", " auch ",
        " wie ", " aber ", " weil ", " wenn ", " dann ", " dort ", " hier ", " auf ", " aus ",
        " über ", " unter ", " hinter ", " vor ", " zwischen ", " werden ", " wurde ",
    }
    padded = f" {lower} "
    matches = sum(1 for token in german_markers if token in padded)
    return matches >= 2

def _normalize_negative_terms(neg_text: str) -> List[str]:
    if not neg_text:
        return []
    parts = re.split(r"[,\n;]+", neg_text)
    cleaned: List[str] = []
    seen = set()
    for p in parts:
        term = p.strip().strip('"').strip("'")
        if not term:
            continue
        key = term.lower()
        if key not in seen:
            cleaned.append(term)
            seen.add(key)
    if len(cleaned) > MAX_NEG_TERMS:
        cleaned = cleaned[:MAX_NEG_TERMS]
    return cleaned

def _term_in_prompt(term: str, prompt: str) -> bool:
    t = term.lower().strip()
    if not t:
        return False
    p = prompt.lower()
    if " " in t:
        return t in p
    return re.search(rf"(^|[^a-z0-9]){re.escape(t)}([^a-z0-9]|$)", p) is not None

RED_PRODUCE_EN = [
    "red vegetables", "red fruits",
    "tomatoes", "tomato",
    "strawberries", "strawberry",
    "red peppers", "bell peppers", "pepper", "chili peppers", "chili",
    "red apples", "cherries", "pomegranates", "watermelon",
    "raspberries", "red currants",
]
RED_COLOR_TOKENS_EN = [
    "warm red tones", "high saturation reds", "crimson", "scarlet", "vermilion",
]

def _expand_semantic_negatives(neg_terms: List[str]) -> List[str]:
    if not neg_terms:
        return []
    base = [t.strip().lower() for t in neg_terms if t.strip()]
    out: List[str] = []

    text_blob = " ".join(base)
    de_rot = any(w in text_blob for w in ["rot", "rotes", "rote", "roten"])
    de_gemuese = "gemüse" in text_blob or "gemuese" in text_blob
    de_obst = "obst" in text_blob

    en_red = "red" in text_blob
    en_veg = "vegetable" in text_blob or "vegetables" in text_blob
    en_fruits = "fruit" in text_blob or "fruits" in text_blob

    if (de_rot and (de_gemuese or de_obst)) or (en_red and (en_veg or en_fruits)):
        out.extend(RED_PRODUCE_EN)

    for t in base:
        if t not in out:
            out.append(t)

    seen = set()
    compact: List[str] = []
    for t in out:
        key = t.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        compact.append(t.strip())

    if len(compact) > MAX_NEG_TERMS:
        compact = compact[:MAX_NEG_TERMS]
    return compact

def _build_keyword_constraints(neg_terms: List[str], add_cool_bias: bool = DEFAULT_COLOR_BIAS) -> str:
    if not neg_terms:
        return ""
    parts: List[str] = []
    neg_kw = ", ".join(neg_terms)
    if neg_kw:
        parts.append(f"negative: {neg_kw}")
    if add_cool_bias:
        parts.append("cool color palette")
        parts.append("avoid warm reds")
        parts.append("blue/green accents")
        parts.append("low saturation")
    sentence = "; ".join(parts) + ";"
    if len(sentence) > MAX_CONSTRAINT_CHARS:
        while len(sentence) > MAX_CONSTRAINT_CHARS and parts:
            parts.pop()
            sentence = "; ".join(parts) + (";" if parts else "")
    return sentence

def _inject_negatives_into_prompt_keyword(prompt: str, negative_prompt: str) -> tuple[str, str]:
    raw_terms = _normalize_negative_terms(negative_prompt)
    if not raw_terms:
        return prompt, ""
    expanded = _expand_semantic_negatives(raw_terms)
    filtered = [t for t in expanded if not _term_in_prompt(t, prompt)]
    if not filtered:
        return prompt, ""
    constraints = _build_keyword_constraints(filtered, add_cool_bias=DEFAULT_COLOR_BIAS)
    if not constraints:
        return prompt, ""
    sep = "" if prompt.endswith((" ", "\n")) else INJECTION_SEPARATOR
    if INJECTION_MODE == "prepend":
        return (constraints + INJECTION_SEPARATOR + prompt).strip(), constraints
    return (prompt + sep + constraints).strip(), constraints


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
    if nlogo:
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

    def _resolve_negative(self, negative: Optional[str]) -> str:
        if negative and negative.strip():
            return negative.strip()
        env_pol = _env_str("POLLINATIONS_NEGATIVE", "")
        if env_pol:
            return env_pol
        env_global = _env_str("APP_GLOBAL_NEGATIVE", "")
        return env_global

    def _finalize_prompt_for_pollinations(self, base_prompt: str, request_negative: Optional[str]) -> tuple[str, str, str]:
        neg_combined = self._resolve_negative(request_negative).strip()
        constraints_preview = ""
        final_prompt = base_prompt

        if neg_combined:
            final_prompt, constraints_preview = _inject_negatives_into_prompt_keyword(base_prompt, neg_combined)

        neg_field = ""
        if constraints_preview:
            m = re.search(r"negative:\s*([^;]+)", constraints_preview, flags=re.IGNORECASE)
            if m:
                neg_field = m.group(1).strip()

        try:
            lang = "de" if _is_german_text(base_prompt) or _is_german_text(neg_combined) else "en"
            print(f"[POLL PREP] lang={lang} style=keywords_en force_inline=False constraints='{(constraints_preview or '')[:140]}'")
            print(f"[POLL PROMPT] used_prompt='{final_prompt[:220]}'")
        except Exception:
            pass

        return final_prompt, neg_field, constraints_preview

    async def _fetch_v1(self, prompt: str, negative: str | None, width: int | None, height: int | None) -> Path:
        self.cfg.require_cloud_enabled()
        url = f"{self.cfg.api_base}/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {self.cfg.secret}",
            "Content-Type": "application/json",
        }
        w = width if (width and width > 0) else self.cfg.width
        h = height if (height and height > 0) else self.cfg.height

        injected_prompt, neg_field, _constraints = self._finalize_prompt_for_pollinations(prompt, negative)

        payload: Dict[str, object] = {
            "model": self.cfg.model or "flux",
            "prompt": injected_prompt,
            "size": (self.cfg.size_override or _size_from_wh(w, h)),
        }
        if neg_field:
            payload["negative_prompt"] = neg_field

        print(f"[POLLINATIONS V1] url={url} model={payload.get('model')} size={payload.get('size')} "
              f"has_negative_field={'negative_prompt' in payload} inline_neg=1")
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
                    print(f"[POLLINATIONS V1] attempt={attempt} error={type(e).__name__}: {e}")
                    if attempt < 5:
                        await asyncio.sleep(delay)
                        delay *= 1.7
                    continue
        raise RuntimeError(f"pollinations_v1_all_attempts_failed: {last_exc}")

    async def _fetch_get(self, prompt: str, negative: str | None, width: int | None, height: int | None) -> Path:
        self.cfg.require_cloud_enabled()
        w = width if (width and width > 0) else self.cfg.width
        h = height if (height and height > 0) else self.cfg.height

        injected_prompt, _neg_field, _constraints = self._finalize_prompt_for_pollinations(prompt, negative)

        url = _build_pollinations_image_url(
            self.cfg.api_base, injected_prompt, self.cfg.model, w, h, self.cfg.nologo, self.cfg.seed
        )
        params: dict[str, str] = {}
        if self.cfg.secret:
            params["key"] = self.cfg.secret

        print(f"[POLLINATIONS GET] url_base={self.cfg.api_base} model={self.cfg.model or 'flux'} "
              f"size={w}x{h} inline_neg=1 nologo={self.cfg.nologo}")
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
                    print(f"[POLLINATIONS GET] attempt={attempt} error={type(e).__name__}: {e}")
                    if attempt < 4:
                        await asyncio.sleep(delay)
                        delay *= 1.7
                    continue
        raise RuntimeError(f"pollinations_get_all_attempts_failed: {last_exc}")

    async def generate(self, prompt: str, width: int | None = None, height: int | None = None, negative: str | None = None) -> Path:
        if not self.cfg.allow_cloud:
            raise RuntimeError("Cloud image backend disabled (ALLOW_CLOUD_IMAGE_BACKEND=0)")

        rw, rh = _resolve_size_for_backend("pollinations", width, height)
        eff_w = rw if (rw and rw > 0) else (width if (width and width > 0) else self.cfg.width)
        eff_h = rh if (rh and rh > 0) else (height if (height and height > 0) else self.cfg.height)

        if self.cfg.use_v1:
            try:
                return await self._fetch_v1(prompt, negative, eff_w, eff_h)
            except Exception:
                return await self._fetch_get(prompt, negative, eff_w, eff_h)
        else:
            return await self._fetch_get(prompt, negative, eff_w, eff_h)


# ============================================================
# ComfyUI (Local/LAN) with optional Auto-Discovery
# ============================================================

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
    node_id_ksampler: str = Field(default_factory=lambda: _env_str("APP_COMFY_NODE_KSAMPLER", "5"))

    # Discovery (opt-in, decoupled from Pollinations)
    auto_discovery_enable: bool = Field(default_factory=lambda: _env_bool01("AUTO_DISCOVERY_ENABLE", 1))
    fallback_hosts_raw: str = Field(default_factory=lambda: _env_str("APP_COMFY_FALLBACK_HOSTS", ""))
    discovery_subnets_raw: str = Field(default_factory=lambda: _env_str("AUTO_DISCOVERY_SUBNETS", ""))
    discovery_max_parallel: int = Field(default_factory=lambda: _env_int("AUTO_DISCOVERY_MAX_PARALLEL", 64))

    def assert_policy(self, host: str) -> None:
        _assert_image_backend_host_policy(host)

    @property
    def fallback_hosts(self) -> List[str]:
        items = [x.strip() for x in self.fallback_hosts_raw.replace(";", ",").split(",") if x.strip()]
        # Deduplicate while preserving order
        out = []
        seen = set()
        for h in items:
            if h not in seen:
                seen.add(h)
                out.append(h)
        return out

    @property
    def discovery_subnets(self) -> List[str]:
        items = [x.strip() for x in self.discovery_subnets_raw.replace(",", " ").split() if x.strip()]
        out = []
        for cidr in items:
            try:
                ipaddress.ip_network(cidr, strict=False)
                out.append(cidr)
            except Exception:
                print(f"[DISCOVERY] ignoring invalid subnet: {cidr}")
        return out


class LocalComfyBackend(ImageBackend):
    def __init__(self, out_dir: Path, cfg: Optional[ComfyConfig] = None) -> None:
        self.out_dir = Path(out_dir).resolve()
        self.cfg = cfg or ComfyConfig()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._samplers_cache: Optional[Set[str]] = None
        self._active_host: Optional[str] = None
        self._discovery_done: bool = False

    async def _probe_history(self, host: str, port: int, timeout: httpx.Timeout) -> bool:
        try:
            async with httpx.AsyncClient(limits=_httpx_limits(), timeout=timeout) as c:
                r = await c.get(f"http://{host}:{port}/history")
                r.raise_for_status()
                return True
        except Exception:
            return False

    async def _ensure_host(self) -> None:
        if self._discovery_done and self._active_host:
            return

        # 1) Try configured host first (works for local 127.0.0.1 and also if explicitly set to LAN IP)
        host = (self.cfg.host or "127.0.0.1").strip()
        port = int(self.cfg.port)
        try:
            self.cfg.assert_policy(host)
        except AssertionError as e:
            print(f"[DISCOVERY] configured host rejected by policy: {e}")
        else:
            if await self._probe_history(host, port, _timeout_probe()):
                self._active_host = host
                self._discovery_done = True
                print(f"[DISCOVERY] using configured ComfyUI host {host}:{port}")
                return

        # 2) Optionally try fallback hosts and subnets (only if remote backends are allowed)
        allow_remote = _env_bool01("APP_ALLOW_REMOTE_BACKENDS", 0)
        if not allow_remote or not self.cfg.auto_discovery_enable:
            # Fallback to configured host; errors will be raised later if unavailable
            self._active_host = host
            self._discovery_done = True
            print("[DISCOVERY] auto-discovery disabled or remote not allowed; keeping configured host")
            return

        # 2a) Sequential quick check of fallback hosts
        for fb in self.cfg.fallback_hosts:
            try:
                self.cfg.assert_policy(fb)
            except AssertionError as e:
                print(f"[DISCOVERY] skip fallback {fb}: {e}")
                continue
            if await self._probe_history(fb, port, _timeout_probe()):
                self._active_host = fb
                self._discovery_done = True
                print(f"[DISCOVERY] selected fallback ComfyUI host {fb}:{port}")
                return

        # 2b) Parallel scan of fallback hosts (if many provided)
        if self.cfg.fallback_hosts and len(self.cfg.fallback_hosts) > 1:
            sem = asyncio.Semaphore(self.cfg.discovery_max_parallel)

            async def _task(h: str):
                async with sem:
                    try:
                        self.cfg.assert_policy(h)
                    except AssertionError:
                        return None
                    ok = await self._probe_history(h, port, _timeout_probe())
                    return h if ok else None

            tasks = [asyncio.create_task(_task(h)) for h in self.cfg.fallback_hosts]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for d in done:
                res = d.result()
                if res:
                    for p in pending:
                        p.cancel()
                    self._active_host = res
                    self._discovery_done = True
                    print(f"[DISCOVERY] selected fallback (parallel) {res}:{port}")
                    return
            for p in pending:
                try:
                    res = await p
                    if res:
                        self._active_host = res
                        self._discovery_done = True
                        print(f"[DISCOVERY] selected fallback (late) {res}:{port}")
                        return
                except asyncio.CancelledError:
                    pass

        # 2c) Optional subnet scans
        for cidr in self.cfg.discovery_subnets:
            try:
                net = ipaddress.ip_network(cidr, strict=False)
            except Exception:
                continue
            # We only scan host addresses, skip loopback/link-local automatically
            ips = [str(ip) for ip in net.hosts()]
            if not ips:
                continue

            sem = asyncio.Semaphore(self.cfg.discovery_max_parallel)

            async def _probe(ip: str):
                async with sem:
                    try:
                        self.cfg.assert_policy(ip)
                    except AssertionError:
                        return None
                    ok = await self._probe_history(ip, port, _timeout_probe())
                    return ip if ok else None

            print(f"[DISCOVERY] scanning subnet {cidr} with up to {self.cfg.discovery_max_parallel} parallel probes")
            tasks = [asyncio.create_task(_probe(ip)) for ip in ips]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for d in done:
                res = d.result()
                if res:
                    for p in pending:
                        p.cancel()
                    self._active_host = res
                    self._discovery_done = True
                    print(f"[DISCOVERY] selected subnet host {res}:{port}")
                    return
            for p in pending:
                try:
                    res = await p
                    if res:
                        self._active_host = res
                        self._discovery_done = True
                        print(f"[DISCOVERY] selected subnet host (late) {res}:{port}")
                        return
                except asyncio.CancelledError:
                    pass

        # 3) As last resort, stick to configured host (may be unreachable; generate() will fail gracefully)
        self._active_host = host
        self._discovery_done = True
        print(f"[DISCOVERY] no reachable ComfyUI found; keeping configured host {host}:{port}")

    async def _available_active(self) -> bool:
        # Ensure discovery (or configured host) is set
        await self._ensure_host()
        host = self._active_host or self.cfg.host
        port = self.cfg.port
        return await self._probe_history(host, port, _timeout_probe())

    async def _fetch_valid_samplers(self, host: str, port: int) -> Set[str]:
        url = f"http://{host}:{port}/object_info/KSampler"
        try:
            async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_short()) as c:
                r = await c.get(url)
                r.raise_for_status()
                j = r.json()
                choices = j.get("input", {}).get("sampler_name", {}).get("choices", [])
                if isinstance(choices, list):
                    return {str(x) for x in choices}
        except Exception:
            pass
        return {
            "euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_sde",
            "dpmpp_2m_karras", "heun", "dpm_fast", "uni_pc"
        }

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

    def _find_clip_nodes(self, prompt_dict: dict) -> list[dict]:
        nodes = []
        for node in prompt_dict.values():
            if isinstance(node, dict) and node.get("class_type") == "CLIPTextEncode":
                nodes.append(node)
        return nodes

    def _override_text_nodes(self, prompt_dict: dict, positive: str, negative: str) -> tuple[bool, bool]:
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

        # Fallback by scanning first two CLIP nodes
        if not (pos_set and neg_set):
            clip_nodes = self._find_clip_nodes(prompt_dict)
            if not clip_nodes:
                return pos_set, neg_set
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

        return pos_set, neg_set

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

    async def generate(self, prompt: str, width: int | None = None, height: int | None = None, negative: str | None = None) -> Path:
        if self.cfg.disabled:
            raise RuntimeError("ComfyUI disabled (APP_DISABLE_COMFYUI=1)")
        if not self.cfg.workflow_path.exists():
            raise RuntimeError(f"workflow_not_found: {self.cfg.workflow_path}")

        await self._ensure_host()
        host = self._active_host or self.cfg.host
        port = int(self.cfg.port)

        # Final availability check before job
        if not await self._probe_history(host, port, _timeout_probe()):
            raise RuntimeError("comfy_unavailable")

        rw, rh = _resolve_size_for_backend("comfyui", width, height)
        w_eff = rw if (rw and rw > 0) else (width if (width and width > 0) else self.cfg.width)
        h_eff = rh if (rh and rh > 0) else (height if (height and height > 0) else self.cfg.height)
        w = _clamp_dim(w_eff)
        h = _clamp_dim(h_eff)

        valid_samplers = await self._fetch_valid_samplers(host, port)
        sampler = self._normalize_sampler(self.cfg.sampler)
        if sampler not in valid_samplers:
            sampler = "euler" if "euler" in valid_samplers else (next(iter(valid_samplers)) if valid_samplers else "euler")

        prompt_dict = self._load_prompt_file()
        clip_nodes_before = self._find_clip_nodes(prompt_dict)
        self._override_dimensions_in_prompt(prompt_dict, width=w, height=h)

        pos_text = (prompt or "").strip()
        neg_text = (negative or self.cfg.negative or "").strip()

        pos_set, neg_set = self._override_text_nodes(prompt_dict, positive=pos_text, negative=neg_text)

        clip_count = len(clip_nodes_before)
        print(f"[COMFY WORKFLOW] host={host} pos_set={pos_set} neg_set={neg_set} clip_nodes={clip_count} "
              f"size={w}x{h} sampler={sampler} pos_sample='{pos_text[:96]}' neg_sample='{neg_text[:96]}'")

        if clip_count == 0:
            raise RuntimeError("workflow_missing_CLIPTextEncode_nodes: no CLIPTextEncode nodes found; cannot set positive/negative text")
        if not neg_set:
            print("[WARN] Negative prompt node not matched by configured IDs; used fallback if possible. "
                  "Verify APP_COMFY_NODE_NEG or workflow structure.")

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
            host=host,
            port=port,
            max_wait_sec=float(self.cfg.timeout_sec),
            poll_interval=1.0,
        )

        if images:
            return images[0]

        copied = self._copy_latest_from_comfy(since_ts=started_at)
        if copied:
            return copied

        raise RuntimeError("comfy_no_images")


# ---------------------------
# Factory (Decoupled selection)
# ---------------------------

class BackendEnv(BaseModel):
    image_backend: str = Field(default_factory=lambda: _env_str("IMAGE_BACKEND", "comfyui").lower())
    allow_cloud: bool = Field(default_factory=lambda: _env_bool01("ALLOW_CLOUD_IMAGE_BACKEND", 0))
    output_dir: Path = Field(default_factory=lambda: Path(_env_str("APP_OUTPUT_DIR", "./outputs/images")).resolve())

def build_image_backend() -> ImageBackend:
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


# ---------------------------
# Optional helper for controllers (post-LLM guard)
# ---------------------------

def inject_negatives_for_final_prompt(prompt: str, negative: str | None) -> str:
    negative = (negative or "").strip()
    if not negative:
        return prompt
    new_prompt, _preview = _inject_negatives_into_prompt_keyword(prompt, negative)
    return new_prompt
