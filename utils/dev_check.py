#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dev_check.py

Purpose:
- Quick local health check: ensure Ollama and ComfyUI are reachable on localhost.

Safety:
- Enforces localhost-only connections via assert_local().
- Short timeouts.

Env:
- APP_OLLAMA_HOST (default 127.0.0.1)
- APP_OLLAMA_PORT (default 11434)
- APP_COMFY_HOST  (default 127.0.0.1)
- APP_COMFY_PORT  (default 8188)
- DEV_CHECK_JSON=1 for JSON output
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Dict, Tuple
from pathlib import Path

import httpx

# ---------- .env loader and helpers (shared logic) ----------

def _strip_inline_comment(s: str) -> str:
    """Strip inline comments after '#', unless inside quotes."""
    s = s.strip()
    out = []
    in_single = False
    in_double = False
    for ch in s:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        if ch == "#" and not in_single and not in_double:
            break
        out.append(ch)
    return "".join(out).strip()


def _load_dotenv_inline(p: Path) -> None:
    """
    Load .env safely:
    - Supports 'export KEY=VALUE'
    - Strips inline comments after '#'
    - Does not override already-set env vars
    """
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        k, v = line.split("=", 1)
        k = k.strip()
        v = _strip_inline_comment(v).strip().strip("'").strip('"')
        if k and v != "":
            os.environ.setdefault(k, v)


def env_str(k: str, default: str) -> str:
    v = os.getenv(k)
    return v.strip() if v is not None else default


def env_int(k: str, default: int) -> int:
    v = os.getenv(k)
    if v is None:
        return default
    try:
        vv = _strip_inline_comment(v)
        return int(vv)
    except Exception:
        return default


# Load .env from current working directory
_load_dotenv_inline(Path(".env"))

# ---------- Config ----------

OLLAMA_HOST = env_str("APP_OLLAMA_HOST", "127.0.0.1")
OLLAMA_PORT = env_int("APP_OLLAMA_PORT", 11434)
COMFY_HOST = env_str("APP_COMFY_HOST", "127.0.0.1")
COMFY_PORT = env_int("APP_COMFY_PORT", 8188)
WANT_JSON = env_str("DEV_CHECK_JSON", "0").lower() in {"1", "true", "yes"}


# ---------- Networking helpers ----------

def assert_local(host: str) -> None:
    """Only allow 127.0.0.1 for safety."""
    if host != "127.0.0.1":
        raise AssertionError(f"Localhost only (got: {host})")


def default_client() -> httpx.AsyncClient:
    """Preconfigured async client with short timeouts."""
    timeout = httpx.Timeout(3.0, connect=2.0)
    return httpx.AsyncClient(timeout=timeout, follow_redirects=True)


async def check_ollama() -> Tuple[bool, str, Dict[str, Any]]:
    """
    Check Ollama via /api/tags.
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
    Prefer GET /system_stats; fallback to GET /.
    Returns (ok, message, details)
    """
    assert_local(COMFY_HOST)
    base = f"http://{COMFY_HOST}:{COMFY_PORT}"
    details: Dict[str, Any] = {"host": COMFY_HOST, "port": COMFY_PORT, "base": base}
    try:
        async with default_client() as client:
            try:
                r = await client.get(f"{base}/system_stats")
                if r.status_code == 200:
                    try:
                        info = r.json()
                    except Exception:
                        info = {"raw": r.text[:200]}
                    details.update({"ok": True, "endpoint": "/system_stats", "info": info})
                    return True, "ComfyUI OK (/system_stats)", details
            except Exception:
                pass

            r = await client.get(base)
            if r.status_code == 200:
                details.update({"ok": True, "endpoint": "/", "title_hint": r.text[:80]})
                return True, "ComfyUI OK (root reachable)", details

            details.update({"ok": False, "status": r.status_code})
            return False, f"ComfyUI unexpected status: {r.status_code}", details
    except Exception as e:
        msg = f"ComfyUI NOT reachable: {e}"
        details.update({"ok": False, "error": str(e)})
        return False, msg, details


# ---------- Main ----------

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
