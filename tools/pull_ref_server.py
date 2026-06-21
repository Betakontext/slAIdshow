# -*- coding: utf-8 -*-
"""
Minimal helper: fetch an image from a given LAN HTTP URL and save it into ComfyUI/input.
Windows-friendly, ASCII-only comments to avoid encoding issues.
Start (PowerShell/CMD in ComfyUI root with venv activated):
  pip install fastapi uvicorn[standard] httpx
  python tools\pull_ref_server.py --host 0.0.0.0 --port 8190
"""

import argparse
from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import uvicorn

# Adjust to your ComfyUI installation root if needed:
# This script expects to be located at <COMFY_ROOT>/tools/pull_ref_server.py
COMFY_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = (COMFY_ROOT / "input").resolve()
INPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ComfyUI PullRef Helper", version="1.0.0")


@app.get("/pull_from_url")
async def pull_from_url(url: str = Query(..., description="LAN-accessible HTTP URL to an image")):
    """
    Download the given URL and save it to ComfyUI/input using the URL path basename.
    Returns the basename and basic metadata.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return JSONResponse({"error": "unsupported scheme"}, status_code=400)

    basename = Path(parsed.path).name
    if not basename:
        return JSONResponse({"error": "invalid url path"}, status_code=400)

    target = INPUT_DIR / basename

    # Download with basic timeouts and redirects enabled
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, read=20.0, connect=10.0), follow_redirects=True) as client:
            r = await client.get(url)
            r.raise_for_status()
            data = r.content
    except Exception as e:
        return JSONResponse({"error": f"download failed: {e}"}, status_code=502)

    # Minimal sanity check: avoid saving empty files
    if not data or len(data) < 256:
        return JSONResponse({"error": "file too small or empty"}, status_code=400)

    # Save into ComfyUI/input
    try:
        target.write_bytes(data)
    except Exception as e:
        return JSONResponse({"error": f"write failed: {e}"}, status_code=500)

    return {"saved": str(target), "basename": basename, "size": len(data)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8190)
    args = parser.parse_args()
    print(f"[pull_ref_server] saving into: {INPUT_DIR}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
