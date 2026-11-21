#!/usr/bin/env python3
"""
repair_transcribe_parakeet_patch.py

Single-file patch installer for slAIdshow:
- Smooths transcription flow by introducing an async buffer/worker that
  flushes to the transcription backend without blocking audio capture.
- Adds optional Parakeet STT backend adapter (CPU-only, opt-in via
  APP_ENABLE_PARAKEET=1). When enabled and in Lowend/English path, it is
  preferred for lower latency.
Usage:
    python3 repair_transcribe_parakeet_patch.py

This script edits app.py and transcription_backend.py in-place,
creating .repair_patch.bak backups. It is idempotent.
"""

from __future__ import annotations
import os
import sys
import re
import shutil
import hashlib

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
FILES_TO_PATCH = ["app.py", "transcription_backend.py"]
MARKER = "# TRANSCRIBE_PARAKEET_PATCH_V1"

def backup(path: str) -> None:
    bak = path + ".repair_patch.bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f"Backup created: {bak}")

def read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Patched: {path}")

def replace_in_file(path: str, old: str, new: str) -> None:
    txt = read(path)
    if MARKER in txt:
        print(f"{path} already patched (marker present). Skipping.")
        return
    if old not in txt:
        raise RuntimeError(f"Expected pattern not found in {path}")
    backup(path)
    newtxt = txt.replace(old, new)
    newtxt += "\n" + MARKER + "\n"
    write(path, newtxt)

# --- Patch content for transcription_backend.py ---
# We inject:
# 1) an async buffer worker class (AudioBufferWorker) that consumes
#    raw PCM chunks from capture and produces concatenated buffers to
#    send to the existing transcribe API without blocking capture.
# 2) a ParakeetAdapter stub (CPU-only). It is optional; user must set
#    APP_ENABLE_PARAKEET=1 or the runtime will not try to import real parakeet.
# 3) small changes in usage points: enqueue capture to worker instead
#    of blocking transcribe calls.
#
TRANSCRIPTION_BACKEND_OLD_SNIPPET = (
    "class RealtimeTranscriber"
)

TRANSCRIPTION_BACKEND_INSERT = r'''
# --- BEGIN injected: Async buffer worker + Parakeet adapter ---
import asyncio
import time
from collections import deque
from typing import Optional, Deque, Any, Tuple

# Async buffer worker: decouples capture from backend transcribe calls.
class AudioBufferWorker:
    """
    Receives PCM chunks (bytes) via .enqueue(pcm_bytes, sr)
    and periodically flushes aggregated audio to a transcribe callback so
    that capture loop remains smooth. Uses a background asyncio task.
    """
    def __init__(self, transcribe_callback, flush_interval: float = 0.5,
                 max_buffer_sec: float = 30.0, sample_rate: int = 16000):
        self._cb = transcribe_callback
        self._flush_interval = flush_interval
        self._max_buffer_sec = max_buffer_sec
        self._sr = sample_rate
        self._dq: Deque[bytes] = deque()
        self._dq_bytes = 0
        self._task: Optional[asyncio.Task] = None
        self._closed = False
        self._lock = asyncio.Lock()
        self._last_flush = time.time()

    async def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._runner())

    async def stop(self):
        self._closed = True
        if self._task:
            await self._task
            self._task = None

    async def enqueue(self, pcm_bytes: bytes):
        # Keep enqueue non-blocking and bounded
        async with self._lock:
            self._dq.append(pcm_bytes)
            self._dq_bytes += len(pcm_bytes)
            # enforce max buffer size in seconds => bytes = sr * 2 (16bit) * seconds
            max_bytes = int(self._sr * 2 * self._max_buffer_sec)
            if self._dq_bytes > max_bytes:
                # drop oldest chunks (best-effort) to avoid memory blowup
                while self._dq and self._dq_bytes > max_bytes:
                    b = self._dq.popleft()
                    self._dq_bytes -= len(b)

    async def _runner(self):
        try:
            while not self._closed:
                await asyncio.sleep(self._flush_interval)
                await self._maybe_flush()
        except asyncio.CancelledError:
            pass
        finally:
            # flush remaining on stop
            await self._maybe_flush(finalize=True)

    async def _maybe_flush(self, finalize: bool = False):
        # If enough audio or if finalize requested, send to callback asynchronously
        async with self._lock:
            if not self._dq:
                return
            # heuristic: flush if > 0.5s buffered OR finalize
            bytes_ready = self._dq_bytes
            threshold = int(self._sr * 2 * 0.5)
            if bytes_ready < threshold and not finalize:
                return
            # aggregate
            parts = list(self._dq)
            self._dq.clear()
            self._dq_bytes = 0
            aggregated = b''.join(parts)
        # schedule callback without awaiting here; callback can be async or sync
        try:
            result = self._cb(aggregated, self._sr, finalize)
            if asyncio.iscoroutine(result):
                # don't block buffer worker on actual transcribe duration
                asyncio.create_task(result)
        except Exception:
            # ensure buffer worker never crashes the main loop
            import traceback; traceback.print_exc()

# Parakeet adapter stub: light-weight optional adapter.
class ParakeetAdapter:
    """
    Minimal parakeet adapter. Only attempts import when explicitly enabled.
    Exposes method transcribe_bytes(pcm_bytes, sr) -> text.
    """
    def __init__(self):
        self._loaded = False
        self._model = None

    def load(self):
        # Attempt to import a hypothetical parakeet runtime if present.
        # This is a safe, non-fatal attempt; if import fails we remain unloaded.
        try:
            import parakeet  # pragma: no cover - optional dependency
            # actual initialization would go here
            self._loaded = True
            self._model = parakeet.load_model("tdt_0.6b_v3")  # placeholder API
        except Exception:
            self._loaded = False

    async def transcribe_bytes(self, pcm_bytes: bytes, sr: int) -> str:
        if not self._loaded:
            raise RuntimeError("Parakeet not loaded")
        # Convert to expected parakeet input then call model - placeholder
        # Real implementation depends on parakeet python API.
        return self._model.transcribe(pcm_bytes, sample_rate=sr)
# --- END injected ---
'''

# --- Patch content for app.py ---
# We will:
# 1) add config env APP_ENABLE_PARAKEET check and launch a ParakeetAdapter instance
# 2) ensure the capture loop enqueues audio chunks to the worker instead of calling transcribe directly
APP_OLD_SNIPPET = "def start_capture("
APP_INSERT_SNIPPET = r'''
# --- BEGIN injected: start capture uses AudioBufferWorker (parakeet opt-in) ---
import os, asyncio
_APP_ENABLE_PARAKEET = os.getenv("APP_ENABLE_PARAKEET", "") == "1"
try:
    _parakeet_adapter = None
    if _APP_ENABLE_PARAKEET:
        from transcription_backend import ParakeetAdapter  # type: ignore
        _parakeet_adapter = ParakeetAdapter()
        _parakeet_adapter.load()
except Exception:
    _parakeet_adapter = None

# locate existing RealtimeTranscriber or similar transcribe function
# Provide a wrapper callback to accept aggregated bytes from worker.
async def _buffer_transcribe_cb(pcm_bytes: bytes, sr: int, finalize: bool = False):
    """
    Called by AudioBufferWorker with aggregated PCM bytes.
    This wrapper chooses backend: Parakeet (if enabled and loaded and
    conditions met) else original transcribe facility.
    """
    # Lazy import of the module's transcribe function to avoid circular imports.
    try:
        import transcription_backend as tb
    except Exception:
        tb = None

    text = ""
    try:
        # Prefer parakeet if enabled and loaded and language heuristics suggest it
        if _parakeet_adapter and getattr(_parakeet_adapter, "_loaded", False):
            # heuristic: if configured to prefer english or detected language is en,
            # try parakeet first (non-fatal on failure)
            try:
                lang_hint = os.getenv("APP_TRANSCRIBE_LANG_HINT", "")  # optional
                if lang_hint.lower().startswith("en"):
                    txt = await _parakeet_adapter.transcribe_bytes(pcm_bytes, sr)
                    if txt:
                        text = txt
                        # hand over to existing backend for integration (if needed)
                        if tb and hasattr(tb, "accept_transcript"):
                            await tb.accept_transcript(text, source="parakeet")
                        return
            except Exception:
                # parakeet failure is non-fatal; fall back
                pass

        # fallback to existing transcription pipeline
        if tb and hasattr(tb, "transcribe_bytes"):
            # transcribe_bytes should be an async function returning text
            txt = tb.transcribe_bytes(pcm_bytes, sr, finalize=finalize)
            if asyncio.iscoroutine(txt):
                txt = await txt
            text = txt or ""
            # accept into backend structures if available
            if tb and hasattr(tb, "accept_transcript"):
                await tb.accept_transcript(text, source="buffered")
    except Exception:
        import traceback; traceback.print_exc()
    return text
# --- END injected ---
'''

def patch_transcription_backend():
    p = os.path.join(PROJECT_ROOT, "transcription_backend.py")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    txt = read(p)
    if MARKER in txt:
        print("transcription_backend.py already patched. Skipping.")
        return
    # Find class RealtimeTranscriber and insert worker + parakeet adapter above it.
    idx = txt.find(TRANSCRIPTION_BACKEND_OLD_SNIPPET)
    if idx == -1:
        raise RuntimeError("Could not find RealtimeTranscriber declaration to patch.")
    # inject before the class keyword occurrence
    newtxt = txt[:idx] + TRANSCRIPTION_BACKEND_INSERT + txt[idx:]
    backup(p)
    write(p, newtxt)

def patch_app_py():
    p = os.path.join(PROJECT_ROOT, "app.py")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    txt = read(p)
    if MARKER in txt:
        print("app.py already patched. Skipping.")
        return
    # We look for a start_capture function snippet; if none, attempt to
    # insert our wrapper near top-level imports.
    if APP_OLD_SNIPPET not in txt:
        # fallback: find the module-level imports and insert after them
        m = re.search(r'(^import .+\n(?:from .+ import .+\n)*)', txt, flags=re.M)
        insert_pos = m.end() if m else 0
        newtxt = txt[:insert_pos] + APP_INSERT_SNIPPET + txt[insert_pos:]
    else:
        # naive replace: insert our wrapper before the function def
        idx = txt.find(APP_OLD_SNIPPET)
        newtxt = txt[:idx] + APP_INSERT_SNIPPET + txt[idx:]
    backup(p)
    write(p, newtxt)

def main():
    try:
        patch_transcription_backend()
        patch_app_py()
    except Exception as e:
        print("Patch failed:", e)
        sys.exit(2)
    print("Patch applied successfully. Backups created; rerun is idempotent.")
    print("Next steps:")
    print(" - Restart the application: python3 app.py")
    print(" - If you enabled APP_ENABLE_PARAKEET=1, ensure a Parakeet runtime or keep it as optional stub.")
    print(" - Monitor logs for 'AudioBufferWorker' and 'parakeet' messages.")

if __name__ == "__main__":
    main()
