#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import asyncio
import httpx

OLLAMA_HOST = os.getenv("APP_OLLAMA_HOST", "127.0.0.1")
OLLAMA_PORT = int(os.getenv("APP_OLLAMA_PORT", "11434"))
COMFY_HOST = os.getenv("APP_COMFY_HOST", "127.0.0.1")
COMFY_PORT = int(os.getenv("APP_COMFY_PORT", "8188"))

def assert_local(host: str) -> None:
    assert host == "127.0.0.1", f"Nur localhost erlaubt (gefunden: {host})"

async def check_ollama() -> str:
    url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/tags"
    assert_local(OLLAMA_HOST)
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, timeout=3.0)
            r.raise_for_status()
            data = r.json()
            models = [m.get("name") for m in data.get("models", [])]
            return f"Ollama OK. Modelle: {', '.join(models) or 'keine'}"
    except Exception as e:
        return f"Ollama NICHT erreichbar: {e}"

async def check_comfy() -> str:
    url = f"http://{COMFY_HOST}:{COMFY_PORT}/prompt"
    assert_local(COMFY_HOST)
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"http://{COMFY_HOST}:{COMFY_PORT}", timeout=3.0)
            return "ComfyUI OK (Root erreichbar)"
    except Exception as e:
        return f"ComfyUI NICHT erreichbar: {e}"

async def main() -> None:
    o = await check_ollama()
    c = await check_comfy()
    print(o)
    print(c)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
