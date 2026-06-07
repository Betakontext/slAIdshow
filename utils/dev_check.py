#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dev_check.py

Purpose:
- Quick local health check for the developer machine.
- Verifies that Ollama and ComfyUI are reachable on localhost with expected ports.
- Prints human-readable status and (optionally) a JSON summary for CI.

Safety:
- Enforces localhost-only connections via assert_local().
- Uses short timeouts and no external network calls.

Env:
- APP_OLLAMA_HOST (default 127.0.0.1)
- APP_OLLAMA_PORT (default 11434)
- APP_COMFY_HOST  (default 127.0.0.1)
- APP_COMFY_PORT  (default 8188)
- DEV_CHECK_JSON=1 to emit a JSON object in addition to human logs
"""

from __future__ import annotations

import os
import sys
import json
import asyncio
from typing import Dict, Any, Tuple

import httpx

OLLAMA_HOST = os.getenv("APP_OLLAMA_HOST", "127.0.0.1")
OLLAMA_PORT = int(os.getenv("APP_OLLAMA_PORT", "11434"))
COMFY_HOST = os.getenv("APP_COMFY_HOST", "127.0.0.1")
COMFY_PORT = int(os.getenv("APP_COMFY_PORT", "8188"))
WANT_JSON = os.getenv("DEV_CHECK_JSON", "0").strip().lower() in {"1", "true", "yes"}


def assert_local(host: str) -> None:
    """
    Hard safety guard: only allow connections to 127.0.0.1.
    Adjust here if you explicitly support ::1 (IPv6 loopback) later.
    """
    if host != "127.0.0.1":
        raise AssertionError(f"Localhost only (got: {host})")


def default_client() -> httpx.AsyncClient:
    """
    Create a preconfigured AsyncClient with short timeouts.
    Retries are left to the caller (we keep checks snappy).
    """
    timeout = httpx.Timeout(3.0, connect=2.0)
    return httpx.AsyncClient(timeout=timeout, follow_redirects=True)


async def check_ollama() -> Tuple[bool, str, Dict[str, Any]]:
    """
    Check Ollama by querying /api/tags on localhost.
    Returns (ok, message, details)
    """
    assert_local(OLLAMA_HOST)
    url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/tags"
    details: Dict[str, Any] = {"host": OLLAMA_HOST, "port": OLLAMA_PORT, "url": url}
    try:
        async with default_client() as client:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()
            # Newer Ollama returns {"models": [{name:..}, ...]}
            models = [m.get("name") for m in data.get("models", []) if isinstance(m, dict)]
            msg = "Ollama OK. Models: " + (", ".join(models) if models else "none")
            details.update({"ok": True, "models": models})
            return True, msg, details
    except Exception as e:
        msg = f"Ollama NOT reachable: {e}"
        details.update({"ok": False, "error": str(e)})
        return False, msg, details


async def check_comfy() -> Tuple[bool, str, Dict[str, Any]]:
    """
    Check ComfyUI on localhost.
    Strategy:
    - Try GET /system_stats (exists in many ComfyUI builds).
    - Fallback: GET / (root) and consider any 200 a good sign.
    Returns (ok, message, details)
    """
    assert_local(COMFY_HOST)
    base = f"http://{COMFY_HOST}:{COMFY_PORT}"
    details: Dict[str, Any] = {"host": COMFY_HOST, "port": COMFY_PORT, "base": base}
    try:
        async with default_client() as client:
            # Preferred health-ish endpoint
            try:
                r = await client.get(f"{base}/system_stats")
                if r.status_code == 200:
                    info = {}
                    try:
                        info = r.json()
                    except Exception:
                        info = {"raw": r.text[:200]}
                    details.update({"ok": True, "endpoint": "/system_stats", "info": info})
                    return True, "ComfyUI OK (/system_stats)", details
            except Exception:
                # Fallback to root
                pass

            r = await client.get(base)
            if r.status_code == 200:
                details.update({"ok": True, "endpoint": "/", "title_hint": r.text[:80]})
                return True, "ComfyUI OK (root reachable)", details
            # Non-200: treat as failure
            details.update({"ok": False, "status": r.status_code})
            return False, f"ComfyUI unexpected status: {r.status_code}", details
    except Exception as e:
        msg = f"ComfyUI NOT reachable: {e}"
        details.update({"ok": False, "error": str(e)})
        return False, msg, details


async def main() -> None:
    o_ok, o_msg, o_det = await check_ollama()
    c_ok, c_msg, c_det = await check_comfy()

    print(o_msg)
    print(c_msg)

    if WANT_JSON:
        print(json.dumps({
            "ollama": o_det,
            "comfyui": c_det,
            "ok": bool(o_ok and c_ok)
        }, ensure_ascii=False))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
