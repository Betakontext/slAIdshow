#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import httpx
from pydantic import BaseModel, Field, ConfigDict, ValidationError

# =========================
# Env helpers
# =========================

def _env_str(k: str, d: str = "") -> str:
    return (os.getenv(k, d) or "").strip()

def _env_int(k: str, d: int) -> int:
    try:
        return int(os.getenv(k, str(d)))
    except Exception:
        return d

def _env_bool01(k: str, d: int = 0) -> bool:
    v = (os.getenv(k, str(d)) or "").strip().lower()
    return v in {"1", "true", "yes", "on"}

APP_HTTP_USER_AGENT = _env_str("APP_HTTP_USER_AGENT", "slAIDshow/2026 (comfy-client)")
# Comfy endpoints: local default 127.0.0.1:8188, remote via WireGuard allowed explicitly
COMFY_HOST_LOCAL = _env_str("COMFY_HOST_LOCAL", "http://127.0.0.1:8188")
COMFY_HOST_REMOTE = _env_str("COMFY_HOST_REMOTE", "")  # e.g., http://10.8.0.5:8188
COMFY_USE_REMOTE = _env_bool01("COMFY_USE_REMOTE", 0)
STRICT_LOCAL_ONLY = _env_bool01("STRICT_LOCAL_ONLY", 1)

# HTTP tuning
COMFY_TIMEOUT_SEC = float(_env_str("COMFY_TIMEOUT_SEC", "60"))
COMFY_MAX_RETRIES = _env_int("COMFY_MAX_RETRIES", 4)
COMFY_RETRY_BASE = float(_env_str("COMFY_RETRY_BASE", "0.8"))

# =========================
# Pydantic models
# =========================

class ComfyPromptRequest(BaseModel):
    """Minimal shape for ComfyUI /prompt request (workflow graph as JSON)."""
    prompt: Dict[str, Any]

class ComfyPromptResponse(BaseModel):
    """ComfyUI /prompt response shape."""
    model_config = ConfigDict(extra="ignore")
    prompt_id: Optional[str] = Field(default=None)
    number: Optional[int] = Field(default=None)
    node_errors: Optional[Dict[str, Any]] = Field(default=None)

class ComfyHistoryImage(BaseModel):
    """Single image result object from /history/{id}."""
    model_config = ConfigDict(extra="ignore")
    filename: Optional[str] = None
    subfolder: Optional[str] = None
    type: Optional[str] = None

class ComfyHistoryOutput(BaseModel):
    """Outputs structure of a node (we read images array)."""
    model_config = ConfigDict(extra="ignore")
    images: Optional[list[ComfyHistoryImage]] = None

class ComfyHistoryEntry(BaseModel):
    """History payload for a prompt_id."""
    model_config = ConfigDict(extra="ignore")
    outputs: Dict[str, ComfyHistoryOutput] = Field(default_factory=dict)
    status: Optional[Dict[str, Any]] = None

# =========================
# HTTP utils
# =========================

def _limits() -> httpx.Limits:
    return httpx.Limits(max_keepalive_connections=6, max_connections=8, keepalive_expiry=20.0)

def _timeout() -> httpx.Timeout:
    return httpx.Timeout(connect=8.0, read=COMFY_TIMEOUT_SEC, write=30.0, pool=8.0)

def _headers() -> Dict[str, str]:
    return {"User-Agent": APP_HTTP_USER_AGENT}

async def _get_with_retries(client: httpx.AsyncClient, url: str) -> httpx.Response:
    """GET with simple backoff on transient errors."""
    last: Optional[Exception] = None
    delay = COMFY_RETRY_BASE
    for attempt in range(1, COMFY_MAX_RETRIES + 1):
        try:
            return await client.get(url)
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            last = e
            if attempt >= COMFY_MAX_RETRIES:
                break
            await asyncio.sleep(delay)
            delay *= 1.8
    raise RuntimeError(f"comfy_get_failed after {COMFY_MAX_RETRIES} attempts: {last}")

async def _post_json_with_retries(client: httpx.AsyncClient, url: str, json_payload: Dict[str, Any]) -> httpx.Response:
    """POST JSON with simple backoff on transient errors."""
    last: Optional[Exception] = None
    delay = COMFY_RETRY_BASE
    for attempt in range(1, COMFY_MAX_RETRIES + 1):
        try:
            return await client.post(url, json=json_payload)
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            last = e
            if attempt >= COMFY_MAX_RETRIES:
                break
            await asyncio.sleep(delay)
            delay *= 1.8
    raise RuntimeError(f"comfy_post_failed after {COMFY_MAX_RETRIES} attempts: {last}")

# =========================
# Client
# =========================

@dataclass
class ComfyClient:
    """Async client for ComfyUI REST endpoints (/prompt, /history/{id})."""
    base_url: str

    @staticmethod
    def from_env() -> "ComfyClient":
        # Deutsch: Sicherheit – wenn STRICT_LOCAL_ONLY aktiv ist, ignoriere Remote.
        if STRICT_LOCAL_ONLY:
            return ComfyClient(base_url=COMFY_HOST_LOCAL.rstrip("/"))
        if COMFY_USE_REMOTE and COMFY_HOST_REMOTE:
            return ComfyClient(base_url=COMFY_HOST_REMOTE.rstrip("/"))
        return ComfyClient(base_url=COMFY_HOST_LOCAL.rstrip("/"))

    def _endpoint(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.base_url}{path}"

    async def health(self) -> Tuple[bool, str]:
        """Quick healthcheck: GET /system_prompt or / (Comfy has /history UI; most setups answer /)."""
        # Deutsch: Manche Builds haben kein explizites Health-Endpoint; wir prüfen "/" und "/history".
        async with httpx.AsyncClient(limits=_limits(), timeout=_timeout(), follow_redirects=True, headers=_headers()) as client:
            for probe in ("/", "/history"):
                url = self._endpoint(probe)
                try:
                    resp = await _get_with_retries(client, url)
                    if resp.status_code < 500:
                        return True, f"ok via {url} [{resp.status_code}]"
                except Exception as e:
                    # Try next probe
                    last = str(e)
            return False, f"healthcheck failed @ {self.base_url}"

    async def submit_prompt(self, workflow: Dict[str, Any]) -> str:
        """POST /prompt with given workflow graph; returns prompt_id."""
        payload = ComfyPromptRequest(prompt=workflow)
        async with httpx.AsyncClient(limits=_limits(), timeout=_timeout(), follow_redirects=True, headers=_headers()) as client:
            url = self._endpoint("/prompt")
            resp = await _post_json_with_retries(client, url, json_payload=payload.model_dump())
            if resp.status_code >= 400:
                raise RuntimeError(f"comfy_post_prompt_failed: http {resp.status_code}: {resp.text[:256]}")
            try:
                js = resp.json()
            except Exception:
                raise RuntimeError("comfy_post_prompt_failed: invalid_json")
            try:
                parsed = ComfyPromptResponse(**js)
            except ValidationError:
                # Fallback minimal extraction
                pid = js.get("prompt_id") or js.get("id")
                if not pid:
                    raise RuntimeError(f"comfy_post_prompt_failed: malformed_response: {js}")
                return str(pid)
            if not parsed.prompt_id:
                raise RuntimeError(f"comfy_post_prompt_failed: missing prompt_id: {js}")
            if parsed.node_errors:
                # Node-level errors are returned, but still a prompt_id might exist
                raise RuntimeError(f"comfy_post_prompt_node_errors: {json.dumps(parsed.node_errors)[:256]}")
            return str(parsed.prompt_id)

    async def wait_for_images(self, prompt_id: str, *, poll_interval: float = 0.8, timeout_s: float = 120.0) -> Dict[str, list[ComfyHistoryImage]]:
        """Poll /history/{prompt_id} until images appear or timeout. Returns dict[node_id] -> [images]."""
        async with httpx.AsyncClient(limits=_limits(), timeout=_timeout(), follow_redirects=True, headers=_headers()) as client:
            url = self._endpoint(f"/history/{prompt_id}")
            deadline = asyncio.get_event_loop().time() + timeout_s
            last_status = ""
            while True:
                if asyncio.get_event_loop().time() > deadline:
                    raise TimeoutError(f"comfy_history_timeout after {timeout_s}s (last_status={last_status})")
                try:
                    resp = await _get_with_retries(client, url)
                except Exception as e:
                    # Backoff and continue polling
                    await asyncio.sleep(poll_interval)
                    continue
                if resp.status_code == 404:
                    # Not yet ready
                    await asyncio.sleep(poll_interval)
                    continue
                if resp.status_code >= 400:
                    raise RuntimeError(f"comfy_history_failed: http {resp.status_code}: {resp.text[:200]}")
                try:
                    js = resp.json()
                except Exception:
                    await asyncio.sleep(poll_interval)
                    continue
                # Response shape: { "<prompt_id>": { "outputs": {node_id: {...}} , "status": {...}}}
                entry_raw = js.get(prompt_id) if isinstance(js, dict) else None
                if not entry_raw:
                    await asyncio.sleep(poll_interval)
                    continue
                try:
                    entry = ComfyHistoryEntry(**entry_raw)
                except ValidationError:
                    # Minimal path: just try to pull outputs
                    outputs = entry_raw.get("outputs", {}) if isinstance(entry_raw, dict) else {}
                    mapped: Dict[str, list[ComfyHistoryImage]] = {}
                    for nid, node_out in outputs.items():
                        images = []
                        for im in (node_out.get("images") or []):
                            try:
                                images.append(ComfyHistoryImage(**im))
                            except ValidationError:
                                pass
                        if images:
                            mapped[nid] = images
                    if mapped:
                        return mapped
                    await asyncio.sleep(poll_interval)
                    continue
                # Track status for debugging
                if entry.status and isinstance(entry.status, dict):
                    last_status = entry.status.get("status", "") or ""
                # Collect images if present
                mapped: Dict[str, list[ComfyHistoryImage]] = {}
                for nid, node_out in (entry.outputs or {}).items():
                    if node_out.images:
                        mapped[nid] = node_out.images
                if mapped:
                    return mapped
                await asyncio.sleep(poll_interval)

# =========================
# Simple CLI test
# =========================

TEST_WORKFLOW_PATH = _env_str("COMFY_TEST_WORKFLOW", "")

async def main() -> None:
    client = ComfyClient.from_env()
    ok, msg = await client.health()
    print(f"[COMFY] health={ok} msg={msg} base={client.base_url}")
    if not ok:
        # Deutsch: Wenn Remote genutzt werden soll, gib klare Hinweise zu WireGuard/Firewall.
        print("[COMFY] HINWEIS: Prüfe ob ComfyUI läuft, Host/Port stimmen und ggf. WireGuard-IP erreichbar ist.")
        return
    if not TEST_WORKFLOW_PATH:
        print("[COMFY] No TEST_WORKFLOW provided; set COMFY_TEST_WORKFLOW=path/to/workflow.json to run a submit.")
        return
    wf_text = Path(TEST_WORKFLOW_PATH).read_text(encoding="utf-8")
    try:
        wf = json.loads(wf_text)
    except Exception as e:
        print(f"[COMFY] invalid workflow JSON: {e}")
        return
    try:
        pid = await client.submit_prompt(wf)
        print(f"[COMFY] submitted prompt_id={pid}")
        images = await client.wait_for_images(pid, poll_interval=0.8, timeout_s=180.0)
        # Print a short summary of image filenames
        for nid, imgs in images.items():
            files = [im.filename for im in imgs if im.filename]
            print(f"[COMFY] node {nid}: images={files}")
    except Exception as e:
        print(f"[COMFY] error: {e}")

if __name__ == "__main__":
    import sys
    try:
        from pathlib import Path
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
