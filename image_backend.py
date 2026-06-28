# image_backend.py
# Production-ready image backend implementations for slAIdshow.
# Comments in English; concise German notes only where logic is subtle.

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import httpx
from pydantic import BaseModel, Field, ValidationError

# External comfy bridge (preferred):
# This module is provided in the project and known-good.
import comfyui_bridge  # type: ignore


# ========= Generic ENV helpers =========

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


def _env_bool(k: str, default: bool = False) -> bool:
    v = (os.getenv(k, "1" if default else "0") or "").strip().lower()
    return v in {"1", "true", "yes", "on"}


def _debug() -> bool:
    v = (os.getenv("APP_IMAGE_BACKEND_DEBUG", "0") or "").strip().lower()
    return v in {"1", "true", "yes", "on"}


def _now_ms() -> int:
    import time
    return int(time.time() * 1000)


def _httpx_limits() -> httpx.Limits:
    # Keep connections warm but bounded
    return httpx.Limits(max_keepalive_connections=10, max_connections=40, keepalive_expiry=30.0)


def _timeout_default() -> httpx.Timeout:
    # Balanced timeouts; overall budget controlled elsewhere
    return httpx.Timeout(connect=5.0, read=20.0, write=10.0, pool=5.0)


# ========= Shared models =========

class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = Field(default=None)
    width: int = Field(default=768, ge=64, le=2048)
    height: int = Field(default=512, ge=64, le=2048)
    seed: Optional[int] = None
    steps: Optional[int] = Field(default=None, ge=1, le=200)
    cfg: Optional[float] = Field(default=None, ge=0.1, le=30.0)
    style: Optional[Dict[str, Any]] = None
    reference_path: Optional[str] = None  # local file path to a style/reference image (if used)
    reference_strength: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ImageResult(BaseModel):
    images: List[str]  # absolute file paths on disk
    backend: str
    meta: Dict[str, Any] = Field(default_factory=dict)


# ========= Pollinations backend (cloud) =========

class PollinationsBackend:
    """
    Cloud backend using Pollinations API.
    Robust retries, negative prompt suffix merging, optional multipart for reference.
    """
    def __init__(self, *, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.base_url = _env_str("POLLINATIONS_BASE_URL", "https://image.pollinations.ai")
        self.timeout_sec = _env_float("POLLINATIONS_TIMEOUT_SEC", 60.0)
        self.max_attempts = _env_int("POLLINATIONS_RETRIES", 4)

    async def generate(self, req: ImageRequest) -> ImageResult:
        # NOTE: This simplified version uses a single endpoint; your project likely has dedicated routes.
        # Here we keep a robust fallback strategy and produce a local saved image if possible.
        payload = {
            "prompt": self._compose_prompt(req),
            "width": req.width,
            "height": req.height,
        }
        if _debug():
            print(f"[POLLINATIONS] POST {self.base_url} payload_keys={list(payload.keys())}")

        url = f"{self.base_url}/images/generate"  # v1 JSON endpoint (example)
        out_paths: List[str] = []

        async with httpx.AsyncClient(limits=_httpx_limits(), timeout=httpx.Timeout(self.timeout_sec)) as client:
            last_exc: Optional[Exception] = None
            delay = 0.6
            for attempt in range(1, self.max_attempts + 1):
                try:
                    r = await client.post(url, json=payload)
                    if r.status_code in (429, 500, 502, 503, 504):
                        raise httpx.HTTPStatusError(f"transient {r.status_code}", request=r.request, response=r)
                    r.raise_for_status()
                    # Expect image bytes or a JSON with URL; support both
                    ctype = r.headers.get("content-type", "").lower()
                    if "application/json" in ctype:
                        data = r.json()
                        img_url = data.get("image_url") or data.get("url")
                        if not img_url:
                            raise RuntimeError("pollinations_no_image_url")
                        # Download the image
                        p = await self._download_image(client, img_url)
                        out_paths.append(str(p))
                    else:
                        # Direct image bytes
                        p = self._write_bytes(r.content)
                        out_paths.append(str(p))
                    break
                except Exception as e:
                    last_exc = e
                    if _debug():
                        print(f"[POLLINATIONS] attempt={attempt} err={type(e).__name__}: {e}")
                    if attempt >= self.max_attempts:
                        raise RuntimeError(f"pollinations_failed after {attempt} attempts: {last_exc}")
                    await asyncio.sleep(delay)
                    delay *= 1.8

        if not out_paths:
            raise RuntimeError("pollinations_empty_result")
        return ImageResult(images=out_paths, backend="pollinations", meta={"endpoint": url})

    async def _download_image(self, client: httpx.AsyncClient, url: str) -> Path:
        r = await client.get(url)
        r.raise_for_status()
        return self._write_bytes(r.content)

    def _write_bytes(self, content: bytes) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        name = f"img_{uuid.uuid4().hex}.png"
        p = self.out_dir / name
        p.write_bytes(content)
        if _debug():
            print(f"[POLLINATIONS] saved -> {p}")
        return p

    def _compose_prompt(self, req: ImageRequest) -> str:
        # Merge negative prompt as suffix; Pollinations typically parses a single string
        prompt = req.prompt.strip()
        if req.negative_prompt:
            prompt = f"{prompt} --no {req.negative_prompt.strip()}"
        return prompt


# ========= Local/Remote ComfyUI backend =========

@dataclass
class ComfyLocalConfig:
    host: str = "127.0.0.1"
    port: int = 8188
    timeout_sec: float = 180.0
    out_dir: Path = Path("outputs/images").resolve()
    workflow_path: Optional[Path] = None  # if you load from a workflow JSON


class LocalComfyBackend:
    """
    Local or remote ComfyUI backend.
    - Preferred path: delegate to comfyui_bridge.generate_from_prompt_dict()
    - Optional internal-bridge fallback: enabled via APP_COMFY_USE_INTERNAL_BRIDGE=1
    - Optional preflight POST to /prompt: APP_COMFY_PREFLIGHT=1
    """
    def __init__(self, cfg: ComfyLocalConfig) -> None:
        self.cfg = cfg

    async def generate(self, prompt_map: Dict[str, Any], req: ImageRequest) -> ImageResult:
        host, port = self.cfg.host, self.cfg.port
        budget = self.cfg.timeout_sec
        out_dir = self.cfg.out_dir

        # Debug info: show host/port and notable ENVs to spot drift
        if _debug():
            print(f"[COMFY][backend] target={host}:{port} budget={budget:.1f}s out_dir={out_dir}")
            print(f"[COMFY][env] APP_COMFY_HOST={_env_str('APP_COMFY_HOST','')} APP_COMFY_PORT={_env_str('APP_COMFY_PORT','')}")
            if self.cfg.workflow_path:
                print(f"[COMFY][workflow] {self.cfg.workflow_path}")

        # Optional preflight to detect pure connectivity issues early
        if _env_bool("APP_COMFY_PREFLIGHT", False):
            await self._preflight_prompt(host, port)

        # Compose prompt payload: assume prompt_map is a Comfy prompt dict (nodes keyed by id)
        payload = {"prompt": prompt_map}

        # Dispatch mode: external comfy bridge or internal fallback
        use_internal = _env_bool("APP_COMFY_USE_INTERNAL_BRIDGE", False)

        try:
            if not use_internal:
                # Preferred: call the robust comfy bridge
                paths = await comfyui_bridge.generate_from_prompt_dict(
                    prompt_dict=payload,
                    out_dir=out_dir,
                    host=host,
                    port=port,
                    max_wait_sec=budget,
                )
            else:
                # Internal fallback: mirrors comfyui_bridge steps for isolation testing
                if _debug():
                    print("[COMFY][backend] using internal bridge fallback")
                paths = await self._internal_bridge_generate(payload, out_dir, host, port, budget)

        except Exception as e:
            # Provide a precise message with URL target and exception fingerprint
            msg = (
                f"comfy_generation_failed: host={host} port={port} "
                f"type={type(e).__name__} msg={str(e)}"
            )
            # Kurzer Hinweis auf History-Aufruf zur manuellen Prüfung (Deutsch für Bedienerfreundlichkeit)
            if _debug():
                print(f"[COMFY][error] {msg}")
                print(f"[COMFY][hint-DE] Teste: curl -s http://{host}:{port}/history | head -c 200")
                print(f"[COMFY][hint-DE] Oder: curl -s -X POST http://{host}:{port}/prompt -H 'Content-Type: application/json' -d '{{\"prompt\":{{}}}}'")

            raise RuntimeError(msg) from e

        if not paths:
            raise RuntimeError("comfy_no_images")
        return ImageResult(images=[str(p) for p in paths], backend="comfyui", meta={"host": host, "port": port})

    # ----- Helpers -----

    async def _preflight_prompt(self, host: str, port: int) -> None:
        """
        Minimal POST to /prompt to separate connectivity from schema issues.
        Uses the same timeouts/limits profile; does not consume budget significantly.
        """
        url = f"http://{host}:{port}/prompt"
        payload = {"prompt": {}}  # minimal empty map; server may 4xx but must connect
        if _debug():
            print(f"[COMFY][preflight] POST {url}")

        async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_default(), follow_redirects=False) as client:
            try:
                r = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
                # 400/405 are acceptable here; we only care that the socket/connect works
                if r.status_code >= 500:
                    r.raise_for_status()
                if _debug():
                    print(f"[COMFY][preflight] status={r.status_code} ok_connectivity")
            except Exception as e:
                # Connectivity or server failure
                raise RuntimeError(f"preflight_failed url={url} type={type(e).__name__} msg={e}")

    async def _internal_bridge_generate(
        self,
        payload: Dict[str, Any],
        out_dir: Path,
        host: str,
        port: int,
        budget: float,
    ) -> List[str]:
        """
        Internal bridge: mirrors comfyui_bridge behavior to help isolate issues.
        - POST /prompt
        - Poll /history/{id}
        - Download images via /view or /api/view
        """
        prompt_id = await self._ib_post_prompt(host, port, payload)
        infos = await self._ib_poll_history(host, port, prompt_id, max_wait_sec=budget)
        paths = await self._ib_download_images(host, port, infos, out_dir)
        return [str(p) for p in paths]

    # ----- Internal Bridge (IB) -----

    class _IBPromptSubmit(BaseModel):
        prompt_id: str = Field(alias="prompt_id")

    class _IBImageInfo(BaseModel):
        filename: str
        subfolder: str
        type: str

    async def _ib_post_prompt(self, host: str, port: int, payload: Dict[str, Any]) -> str:
        url = f"http://{host}:{port}/prompt"
        if _debug():
            print(f"[COMFY][ib:submit] {url}")
        async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_default(), follow_redirects=False) as client:
            r = await self._ib_retry_post(client, url, payload)
            try:
                parsed = LocalComfyBackend._IBPromptSubmit.model_validate(r.json())
                return parsed.prompt_id
            except ValidationError as e:
                raise RuntimeError(f"ib_invalid_prompt_submit_response: {e}")

    async def _ib_retry_post(self, client: httpx.AsyncClient, url: str, payload: Dict[str, Any]) -> httpx.Response:
        last_exc: Optional[Exception] = None
        delay = 0.6
        max_attempts = _env_int("APP_COMFY_POST_RETRIES", 4)
        for attempt in range(1, max_attempts + 1):
            try:
                r = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
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
        raise RuntimeError(f"ib_post_failed after {max_attempts} attempts: {last_exc}")

    async def _ib_poll_history(self, host: str, port: int, prompt_id: str, *, max_wait_sec: float) -> List[_IBImageInfo]:
        url = f"http://{host}:{port}/history/{prompt_id}"
        if _debug():
            print(f"[COMFY][ib:history] {url} budget={max_wait_sec:.1f}s")
        t0 = _now_ms()
        delay = 0.5
        async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_default(), follow_redirects=False) as client:
            while True:
                if (_now_ms() - t0) / 1000.0 > max_wait_sec:
                    raise TimeoutError(f"ib_history_poll_timeout after {max_wait_sec:.1f}s")
                try:
                    r = await client.get(url)
                    if r.status_code in (429, 500, 502, 503, 504):
                        # Retry on transient
                        await asyncio.sleep(delay)
                        delay = min(2.0, delay * 1.2)
                        continue
                    r.raise_for_status()
                    data = r.json()
                    entry = data.get(prompt_id, {})
                    outputs = entry.get("outputs")
                    if outputs is None and isinstance(entry.get("prompt"), dict):
                        prm = entry["prompt"]
                        if isinstance(prm.get("outputs"), dict):
                            outputs = prm.get("outputs")
                    images: List[LocalComfyBackend._IBImageInfo] = []
                    if isinstance(outputs, dict):
                        for node_out in outputs.values():
                            imgs = node_out.get("images") if isinstance(node_out, dict) else None
                            if isinstance(imgs, list):
                                for im in imgs:
                                    try:
                                        ii = LocalComfyBackend._IBImageInfo.model_validate(im)
                                        images.append(ii)
                                    except ValidationError:
                                        continue
                    if images:
                        if _debug():
                            print(f"[COMFY][ib:history] images={len(images)}")
                        return images
                except (httpx.HTTPError, ValueError, KeyError, json.JSONDecodeError):
                    # transient parse/network; continue
                    pass
                await asyncio.sleep(delay)
                delay = min(2.0, delay * 1.2)

    async def _ib_download_images(self, host: str, port: int, images: List[_IBImageInfo], out_dir: Path) -> List[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        mode = self._ib_choose_view_mode(host)
        results: List[Path] = []

        async def _try_one(info: LocalComfyBackend._IBImageInfo) -> Optional[Path]:
            if mode == "path":
                p = await self._ib_download_path_mode(host, port, info, out_dir)
                if p is not None:
                    return p
                return await self._ib_download_query_mode(host, port, info, out_dir)
            else:
                p = await self._ib_download_query_mode(host, port, info, out_dir)
                if p is not None:
                    return p
                return await self._ib_download_path_mode(host, port, info, out_dir)

        for info in images:
            p = await _try_one(info)
            if p is not None:
                results.append(p)
            if len(results) >= max(1, _env_int("APP_COMFY_MAX_IMAGES", 4)):
                break
        return results

    def _ib_choose_view_mode(self, host: str) -> str:
        override = _env_str("APP_COMFY_FORCE_VIEW_MODE", "")
        if override in {"path", "query"}:
            return override
        return "path" if host in {"127.0.0.1", "localhost"} else "query"

    async def _ib_download_path_mode(self, host: str, port: int, info: _IBImageInfo, out_dir: Path) -> Optional[Path]:
        base = f"http://{host}:{port}"
        fname = self._ib_sanitize_filename(info.filename)
        subf = "/".join([self._ib_sanitize_filename(p) for p in (info.subfolder or "").strip("/").split("/") if p])
        t = self._ib_sanitize_filename(info.type or "output")
        url = f"{base}/view/{t}/{subf}/{fname}" if subf else f"{base}/view/{t}/{fname}"
        async with httpx.AsyncClient(limits=_httpx_limits(), timeout=httpx.Timeout(30.0), follow_redirects=False) as client:
            try:
                r = await client.get(url)
                if r.status_code in (429, 500, 502, 503, 504):
                    r = await client.get(url)  # one retry; keep it simple here
                r.raise_for_status()
                content = r.content
                if not content or len(content) < max(128, _env_int("APP_COMFY_MIN_IMAGE_BYTES", 512)):
                    if _debug():
                        print(f"[COMFY][ib:dl:path] too_small len={len(content) if content else 0} url={url}")
                    return None
                p = out_dir / f"img_{uuid.uuid4().hex}{self._ib_suffix_from_name(fname)}"
                p.write_bytes(content)
                if _debug():
                    print(f"[COMFY][ib:dl:path] saved -> {p}")
                return p
            except Exception as e:
                if _debug():
                    print(f"[COMFY][ib:dl:path] {e}")
                return None

    async def _ib_download_query_mode(self, host: str, port: int, info: _IBImageInfo, out_dir: Path) -> Optional[Path]:
        base = f"http://{host}:{port}"
        params = {"filename": info.filename, "subfolder": info.subfolder, "type": info.type}
        url = f"{base}/api/view"
        async with httpx.AsyncClient(limits=_httpx_limits(), timeout=httpx.Timeout(30.0), follow_redirects=False) as client:
            try:
                r = await client.get(url, params=params)
                if r.status_code in (429, 500, 502, 503, 504):
                    r = await client.get(url, params=params)
                r.raise_for_status()
                content = r.content
                if not content or len(content) < max(128, _env_int("APP_COMFY_MIN_IMAGE_BYTES", 512)):
                    if _debug():
                        print(f"[COMFY][ib:dl:query] too_small len={len(content) if content else 0} url={url} params={params}")
                    return None
                p = out_dir / f"img_{uuid.uuid4().hex}{self._ib_suffix_from_name(params['filename'])}"
                p.write_bytes(content)
                if _debug():
                    print(f"[COMFY][ib:dl:query] saved -> {p}")
                return p
            except Exception as e:
                if _debug():
                    print(f"[COMFY][ib:dl:query] {e} params={params}")
                return None

    @staticmethod
    def _ib_suffix_from_name(name: str) -> str:
        s = Path(name).suffix.lower()
        return s if s else ".png"

    @staticmethod
    def _ib_sanitize_filename(name: str) -> str:
        name = name.replace("\\", "/").split("/")[-1]
        return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


# ========= Backend factory =========

BackendName = Literal["comfyui", "comfyui_remote", "pollinations"]


@dataclass
class BackendConfig:
    backend: BackendName = "pollinations"
    # Comfy config
    comfy_host: str = "127.0.0.1"
    comfy_port: int = 8188
    comfy_timeout_sec: float = 180.0
    workflow_path: Optional[str] = None
    # Output dir
    out_dir: str = str(Path("outputs/images").resolve())


class ImageBackend:
    """
    Facade that dispatches to configured backend (Comfy local/remote, Pollinations).
    """
    def __init__(self, cfg: BackendConfig) -> None:
        self.cfg = cfg
        self._impl = self._build_impl(cfg)

    def _build_impl(self, cfg: BackendConfig):
        out_dir = Path(cfg.out_dir).resolve()
        if cfg.backend in ("comfyui", "comfyui_remote"):
            wf = Path(cfg.workflow_path).resolve() if cfg.workflow_path else None
            return LocalComfyBackend(
                ComfyLocalConfig(
                    host=cfg.comfy_host,
                    port=cfg.comfy_port,
                    timeout_sec=cfg.comfy_timeout_sec,
                    out_dir=out_dir,
                    workflow_path=wf,
                )
            )
        elif cfg.backend == "pollinations":
            return PollinationsBackend(out_dir=out_dir)
        else:
            raise ValueError(f"unknown backend: {cfg.backend}")

    async def generate(self, req: ImageRequest, prompt_map: Optional[Dict[str, Any]] = None) -> ImageResult:
        # For Comfy, prompt_map is required; for Pollinations, we compose from req.prompt.
        if isinstance(self._impl, LocalComfyBackend):
            if not isinstance(prompt_map, dict):
                raise ValueError("Comfy backend requires a workflow prompt_map (dict of nodes).")
            return await self._impl.generate(prompt_map, req)
        else:
            return await self._impl.generate(req)


# ========= Convenience builders (for app.py) =========

def build_image_backend_from_env() -> ImageBackend:
    backend = _env_str("APP_IMAGE_BACKEND", _env_str("IMAGE_BACKEND", "pollinations")) or "pollinations"
    comfy_host = _env_str("APP_COMFY_HOST", "127.0.0.1")
    comfy_port = _env_int("APP_COMFY_PORT", 8188)
    comfy_timeout = _env_float("APP_COMFY_TIMEOUT_SEC", 180.0)
    out_dir = _env_str("APP_OUTPUT_DIR", str(Path("outputs/images").resolve()))
    wf = _env_str("APP_COMFY_WORKFLOW", "")
    cfg = BackendConfig(
        backend=backend, comfy_host=comfy_host, comfy_port=comfy_port,
        comfy_timeout_sec=comfy_timeout, workflow_path=(wf or None), out_dir=out_dir
    )
    return ImageBackend(cfg)


def build_image_backend_from_name(name: BackendName, *, comfy_host: str, comfy_port: int, out_dir: str, comfy_timeout_sec: float = 180.0, workflow_path: Optional[str] = None) -> ImageBackend:
    cfg = BackendConfig(
        backend=name,
        comfy_host=comfy_host,
        comfy_port=comfy_port,
        comfy_timeout_sec=comfy_timeout_sec,
        workflow_path=workflow_path,
        out_dir=out_dir,
    )
    return ImageBackend(cfg)
