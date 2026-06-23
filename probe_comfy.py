#!/usr/bin/env python3
# Comments strictly in English

import asyncio
import json
import os
from typing import Any, Dict, Optional, Tuple

import httpx

# Quick async probe of a ComfyUI host:port, reporting detailed outcome.
# Usage:
#   NO_PROXY=localhost,127.0.0.1,192.168.188.24 python probe_comfy.py 192.168.188.24 8188

def _timeout_short() -> httpx.Timeout:
    # German: kurze Timeouts für schnelle Diagnose
    return httpx.Timeout(connect=2.5, read=4.0, write=3.0, pool=3.0)

def _limits() -> httpx.Limits:
    return httpx.Limits(max_keepalive_connections=4, max_connections=8, keepalive_expiry=15.0)

async def probe(host: str, port: int) -> Tuple[bool, Dict[str, Any]]:
    """Return (ok, info) where info contains error detail or status/latency."""
    base = f"http://{host}:{port}"
    info: Dict[str, Any] = {"host": host, "port": port, "base": base}
    try:
        async with httpx.AsyncClient(limits=_limits(), timeout=_timeout_short(), follow_redirects=True) as c:
            # First: /history (cheap, no body)
            r1 = await c.get(f"{base}/history")
            info["history_status"] = r1.status_code
            if 200 <= r1.status_code < 300:
                # Optional: fetch tags object_info to confirm API shape
                try:
                    r2 = await c.get(f"{base}/object_info/KSampler")
                    info["ksampler_status"] = r2.status_code
                except Exception as e:
                    info["ksampler_error"] = f"{type(e).__name__}: {e}"
                return True, info
            else:
                # Report body snippet for diagnostics
                try:
                    text = (r1.text or "")[:240]
                except Exception:
                    text = ""
                info["history_body"] = text
                return False, info
    except httpx.ConnectError as e:
        info["error"] = f"ConnectError: {e}"
    except httpx.ReadTimeout as e:
        info["error"] = f"ReadTimeout: {e}"
    except httpx.WriteTimeout as e:
        info["error"] = f"WriteTimeout: {e}"
    except httpx.RemoteProtocolError as e:
        info["error"] = f"RemoteProtocolError: {e}"
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return False, info

async def amain() -> int:
    import sys
    if len(sys.argv) < 3:
        print("Usage: python probe_comfy.py <host> <port>", flush=True)
        return 2
    host = sys.argv[1].strip()
    try:
        port = int(sys.argv[2])
    except ValueError:
        print("Invalid port", flush=True)
        return 2

    # German: Optional NO_PROXY setzen, um Proxy-Störungen zu vermeiden
    no_proxy = os.getenv("NO_PROXY", "")
    if host not in (no_proxy or ""):
        os.environ["NO_PROXY"] = (no_proxy + ("," if no_proxy else "") + host)

    ok, info = await probe(host, port)
    print(json.dumps({"ok": ok, "info": info}, ensure_ascii=False, indent=2))
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
