#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import re
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import httpx
import numpy as np
import sounddevice as sd
from fastapi import FastAPI, Request, UploadFile, File, Body
from fastapi.responses import (
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
    RedirectResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator, HttpUrl

# Image backend factory and interface (from your project)
from image_backend import build_image_backend, ImageBackend
from image_backend import merge_style_prompt  # helper

try:
    from image_backend import LocalComfyBackend  # type: ignore
except Exception:
    LocalComfyBackend = None  # type: ignore

# style_engine is optional and versioned; functions may be missing
try:
    import style_engine  # may expose: ensure_dirs, save_reference_from_bytes, save_reference_from_url, build_styles, StyleEngineRequest
except Exception as e:
    style_engine = None  # We'll error lazily when required
    print(f"[STYLE] style_engine not available: {e}")

# ---------- ENV helpers ----------

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


def _backend_default_size(backend_name: str) -> tuple[int, int]:
    """
    Backend-specific default sizes from .env:
    - pollinations: POLLINATIONS_WIDTH/HEIGHT, fallback 1024×1024
    - otherwise (comfy): APP_COMFY_WIDTH/HEIGHT, fallback 128×128
    """
    b = (backend_name or "comfyui").lower()
    if b == "pollinations":
        w = _env_int("POLLINATIONS_WIDTH", 1024)
        h = _env_int("POLLINATIONS_HEIGHT", 1024)
    else:
        w = _env_int("APP_COMFY_WIDTH", 128)
        h = _env_int("APP_COMFY_HEIGHT", 128)
    w = max(64, min(2048, w))
    h = max(64, min(2048, h))
    return w, h

# ---------- Optional dotenv loading ----------

ENV_PATH: Optional[str] = None
try:
    from dotenv import find_dotenv, load_dotenv
    explicit = os.environ.get("ENV_FILE")
    if explicit:
        found = find_dotenv(explicit, usecwd=True)
        if found:
            load_dotenv(found, override=True)
            ENV_PATH = found
        elif os.path.isfile(explicit):
            load_dotenv(explicit, override=True)
            ENV_PATH = os.path.abspath(explicit)
    if ENV_PATH is None:
        found = find_dotenv(".env", usecwd=True)
        if found:
            load_dotenv(found, override=True)
            ENV_PATH = found
except Exception as e:
    print(f"[ENV] dotenv not available or failed: {e}")

# ---------- Audio config ----------

AUDIO_DEVICE_PREF = _env_str("APP_AUDIO_DEVICE", "") or None
SAMPLE_RATE = _env_int("APP_SAMPLE_RATE", 48000)
FRAME_MS = _env_int("APP_FRAME_DURATION_MS", 20)
APP_STREAM_LATENCY_SEC = _env_float("APP_STREAM_LATENCY_SEC", 0.12)

DISABLE_VAD = _env_bool01("APP_DISABLE_VAD", 1)
RMS_VAD_THRESHOLD = _env_float("APP_RMS_VAD_THRESHOLD", 0.015)

SNAPSHOT_SEC = _env_float("APP_SNAPSHOT_SEC", 2.5)
MIN_BUF_SEC = _env_float("APP_MIN_BUF_SEC", 0.35)
MAX_SILENCE_MS = _env_int("APP_MAX_SILENCE_MS", 700)
MAX_SEGMENT_SEC = _env_float("APP_MAX_SEGMENT_SEC", 12.0)
FIRST_SNAPSHOT_DEADLINE_SEC = _env_float("APP_FIRST_SNAPSHOT_DEADLINE_SEC", 1.2)

# ---------- SSE heartbeat ----------

APP_SSE_TICK_SEC = _env_float("APP_SSE_TICK_SEC", 1.0)

# ---------- Whisper (pywhispercpp) ----------

WHISPER_MODEL_PATH = _env_str("APP_WHISPER_MODEL_PATH", "")
WHISPER_LANGUAGE = _env_str("APP_WHISPER_LANGUAGE", "de")
WHISPER_THREADS = _env_int("APP_WHISPER_THREADS", 2)
WHISPER_TEMPERATURE = _env_float("APP_WHISPER_TEMPERATURE", 0.0)
WHISPER_MIN_SEC = _env_float("APP_WHISPER_MIN_SEC", 0.35)
WHISPER_MIN_PEAK = _env_float("APP_WHISPER_MIN_PEAK", 0.0009)

# ---------- Text filtering ----------

TEXT_MIN_CHARS = _env_int("APP_TEXT_MIN_CHARS", 3)
TEXT_MIN_WORDS = _env_int("APP_TEXT_MIN_WORDS", 1)
FORCE_MEANINGFUL_CHECK = _env_bool01("APP_FORCE_MEANINGFUL_CHECK", 0)

CONTEXT_MAX_SEGMENTS = _env_int("APP_CONTEXT_MAX_SEGMENTS", 5)
CONTEXT_MAX_CHARS = _env_int("APP_CONTEXT_MAX_CHARS", 480)

# ---------- Output/static config ----------

OUTPUT_DIR = Path(_env_str("APP_OUTPUT_DIR", "./outputs/images")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def rel_for_ui(p: Path) -> str:
    return Path(p).name

def ensure_in_output_dir(p: Path) -> Path:
    """
    Ensure the generated file is placed in OUTPUT_DIR; move/copy if needed.
    """
    try:
        p = Path(p).resolve()
    except Exception:
        p = Path(p)
    if p.parent == OUTPUT_DIR:
        return p
    target = OUTPUT_DIR / p.name
    if target.exists():
        stem, suf = p.stem, p.suffix
        for i in range(1, 1000):
            cand = OUTPUT_DIR / f"{stem}_{i}{suf}"
            if not cand.exists():
                target = cand
                break
    try:
        with contextlib.suppress(Exception):
            p.replace(target)
            print(f"[SAVE] moved image to {target}")
            return target
        data = p.read_bytes()
        target.write_bytes(data)
        with contextlib.suppress(Exception):
            p.unlink()
        print(f"[SAVE] copied image to {target}")
        return target
    except Exception as e:
        print(f"[SAVE] failed to move/copy image: {e}")
        return p

# UI-defaults (initial)
APP_IMAGE_WIDTH = _env_int("APP_IMAGE_WIDTH", 512)
APP_IMAGE_HEIGHT = _env_int("APP_IMAGE_HEIGHT", 512)

# ---------- Workflows (ComfyUI) ----------

WORKFLOWS_DIR = Path(os.getenv("WORKFLOWS_DIR", "workflows")).resolve()
WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Ollama ----------

OLLAMA_HOST = _env_str("APP_OLLAMA_HOST", "127.0.0.1")
OLLAMA_PORT = _env_int("APP_OLLAMA_PORT", 11434)
OLLAMA_MODEL = _env_str("APP_OLLAMA_MODEL", "llama3.2:latest")
OLLAMA_TEMPERATURE = _env_float("APP_OLLAMA_TEMPERATURE", 0.2)
OLLAMA_NUM_CTX = _env_int("APP_OLLAMA_NUM_CTX", 3072)
OLLAMA_NUM_PREDICT = _env_int("APP_OLLAMA_NUM_PREDICT", 640)
OLLAMA_TOP_K = _env_int("APP_OLLAMA_TOP_K", 40)
OLLAMA_TOP_P = _env_float("APP_OLLAMA_TOP_P", 0.9)
OLLAMA_REPEAT_PENALTY = _env_float("APP_OLLAMA_REPEAT_PENALTY", 1.1)
OLLAMA_TIMEOUT_SEC = _env_float("APP_OLLAMA_TIMEOUT_SEC", 90.0)
OLLAMA_MAX_RETRIES = _env_int("APP_OLLAMA_MAX_RETRIES", 4)
OLLAMA_RETRY_BASE_DELAY = _env_float("APP_OLLAMA_RETRY_BASE_DELAY", 0.8)
LLM_INTERVAL_SEC = _env_float("APP_LLM_INTERVAL_SEC", 10.0)
OLLAMA_DISABLED = _env_bool01("APP_OLLAMA_DISABLE", 0)

OLLAMA_SYS_PROMPT = _env_str(
    "APP_OLLAMA_SYS_PROMPT",
    "Du bist ein präziser Prompt-Designer für Bildgeneratoren. Erzeuge kurze, klare, fotografische oder illustrative Bild-Prompts, ohne Meta-Kommentare, in Deutsch.",
)

def assert_local(host: str) -> None:
    """
    Hard safety: do not allow remote hosts for Ollama.
    """
    if host != "127.0.0.1":
        raise AssertionError(f"Only localhost allowed, got {host}")

assert_local(OLLAMA_HOST)

def _assert_image_backend_host() -> None:
    """
    Allow remote image backends if explicitly enabled.
    Blocks remote if not enabled.
    """
    allow_remote = _env_bool01("APP_ALLOW_REMOTE_BACKENDS", 0)
    comfy_host = _env_str("APP_COMFY_HOST", "127.0.0.1")
    if not allow_remote:
        if comfy_host not in {"127.0.0.1", "localhost"}:
            raise AssertionError(f"Remote image backends disabled, got {comfy_host}")

_assert_image_backend_host()

WARMUP_ENABLE = _env_bool01("APP_OLLAMA_WARMUP_ENABLE", 1)
WARMUP_PROMPT = _env_str("APP_OLLAMA_WARMUP_PROMPT", "Sag Hallo auf Deutsch.")
WARMUP_TIMEOUT_SEC = _env_float("APP_OLLAMA_TIMEOUT_SEC", 45.0)
WARMUP_MAX_RETRIES = _env_int("APP_OLLAMA_MAX_RETRIES", 3)
WARMUP_RETRY_DELAY = _env_float("APP_OLLAMA_RETRY_DELAY", 1.2)
WARMUP_GRACE_SEC = _env_float("APP_OLLAMA_GRACE_SEC", 10.0)

# ---------- Pydantic payloads ----------

class OllamaGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    prompt: str
    stream: bool = False
    options: dict = Field(default_factory=dict)

class OllamaChatTurn(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class OllamaChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    messages: List[OllamaChatTurn]
    stream: bool = False
    options: dict = Field(default_factory=dict)

class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(..., min_length=1)
    tags: List[str] = Field(default_factory=list)
    width: Optional[int] = None
    height: Optional[int] = None
    style_positive: Optional[str] = None

class HealthReport(BaseModel):
    ollama_ok: bool
    image_backend: str
    allow_cloud: bool
    output_dir: str
    output_dir_exists: bool
    last_prompt: Optional[str] = None
    last_llm_error: Optional[str] = None
    pollinations_key_present: bool = False

class ImageBackendSwitch(BaseModel):
    backend: Literal["comfyui", "pollinations"]
    reset: bool = False

_MIN_SIZE = 128
_MAX_SIZE = 2048

class ImageRequest(BaseModel):
    prompt: str = Field(min_length=0, max_length=2000)
    width: int | None = Field(default=None)
    height: int | None = Field(default=None)
    negative_prompt: Optional[str] = None
    style_positive: Optional[str] = None

    @field_validator("width", "height")
    @classmethod
    def _clamp_size(cls, v: int | None) -> int | None:
        if v is None:
            return v
        try:
            iv = int(v)
        except Exception:
            return None
        if iv < _MIN_SIZE:
            iv = _MIN_SIZE
        if iv > _MAX_SIZE:
            iv = _MAX_SIZE
        return iv

class DirectImageRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    width: int | None = None
    height: int | None = None
    negative_prompt: Optional[str] = None
    style_positive: Optional[str] = None

class ImageResponse(BaseModel):
    filename: str
    relpath: str
    rel: Optional[str] = None
    width: int | None = None
    height: int | None = None

class ImageSizeSettings(BaseModel):
    width: int = Field(ge=_MIN_SIZE, le=_MAX_SIZE)
    height: int = Field(ge=_MIN_SIZE, le=_MAX_SIZE)

class NegativePromptSettings(BaseModel):
    negative_prompt: str = Field(default="", max_length=4000)

class StylePromptSettings(BaseModel):
    style_positive: str = Field(default="", max_length=4000)

# Workflows API payloads
class WorkflowItem(BaseModel):
    name: str = Field(..., description="Display name (stem)")
    filename: str = Field(..., description="Base filename ending with .json")

class WorkflowList(BaseModel):
    items: List[WorkflowItem]

class WorkflowSelect(BaseModel):
    filename: str = Field(..., description="Base filename ending with .json")

    @field_validator("filename")
    @classmethod
    def safe_filename(cls, v: str) -> str:
        if "/" in v or "\\" in v:
            raise ValueError("Invalid filename")
        if not v.endswith(".json"):
            raise ValueError("Must end with .json")
        if not re.fullmatch(r"[A-Za-z0-9._\\-]+", v):
            raise ValueError("Illegal characters in filename")
        return v

# ---------- Audio utils ----------

def pick_input_device(prefer: Optional[str] = None) -> int:
    """
    Select a microphone device with preference heuristic.
    """
    devs = sd.query_devices()
    if not devs:
        raise RuntimeError("No audio devices found")
    if prefer:
        for i, d in enumerate(devs):
            if d.get("name") == prefer and d.get("max_input_channels", 0) > 0:
                return i
        for i, d in enumerate(devs):
            if prefer.lower() in (d.get("name", "").lower()) and d.get("max_input_channels", 0) > 0:
                return i
    for i, d in enumerate(devs):
        if "pulse" in (d.get("name", "").lower()) and d.get("max_input_channels", 0) > 0:
            return i
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0:
            return i
    raise RuntimeError("No input device with max_input_channels>0 found.")

def to_int16(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16, copy=False)

def rms_vad(frame: np.ndarray, rms_threshold: float = 0.01) -> bool:
    if frame.size == 0:
        return False
    rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float32), dtype=np.float64)))
    return rms >= rms_threshold

def resample_to_16k(samples: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return samples.astype(np.float32, copy=False)
    target_len = int(samples.shape[0] * (16000.0 / float(sr)))
    if target_len <= 0:
        return np.array([], dtype=np.float32)
    return np.interp(
        np.linspace(0.0, 1.0, num=target_len, endpoint=False, dtype=np.float64),
        np.linspace(0.0, 1.0, num=samples.shape[0], endpoint=False, dtype=np.float64),
        samples.astype(np.float64, copy=False),
    ).astype(np.float32, copy=False)

# ---------- Whisper init ----------

WHISPER_AVAILABLE = True
try:
    from pywhispercpp.model import Model as WhisperModel  # type: ignore
except Exception as e:
    print(f"[WARN] could not import pywhispercpp: {e}")
    WhisperModel = None  # type: ignore
    WHISPER_AVAILABLE = False

_WHISPER_MODEL: Optional[WhisperModel] = None

def init_whisper_model() -> None:
    """
    Load whisper.cpp model if available and configured.
    """
    global _WHISPER_MODEL
    if not WHISPER_AVAILABLE or _WHISPER_MODEL is not None:
        return
    if not WHISPER_MODEL_PATH or not Path(WHISPER_MODEL_PATH).is_file():
        print(f"[WHISPER] model not found/disabled: {WHISPER_MODEL_PATH}")
        return
    try:
        _WHISPER_MODEL = WhisperModel(
            WHISPER_MODEL_PATH,
            n_threads=WHISPER_THREADS,
            print_progress=False,
            print_realtime=False,
            language=WHISPER_LANGUAGE or None,
            translate=False,
            temperature=WHISPER_TEMPERATURE,
        )
        print(
            f"[WHISPER] model loaded: {Path(WHISPER_MODEL_PATH).name}, "
            f"threads={WHISPER_THREADS}, lang={WHISPER_LANGUAGE}"
        )
    except Exception as e:
        print(f"[WHISPER] initialization failed: {e}")
        _WHISPER_MODEL = None

TEXT_FIELD_RE = re.compile(r"text\s*=\s*(.+?)(?:,|$)")
META_RE = re.compile(
    r"\b(musik|music|applaus|applause|lachen|laugh|geräusch|noise|husten|cough|klatschen|klingel|ring|summen|hmm+|pause)\b",
    re.I,
)

def _parse_whisper_out(raw: object) -> str:
    if raw is None:
        return ""
    if isinstance(raw, dict):
        if isinstance(raw.get("text"), str):
            return raw["text"]
        segs = raw.get("segments")
        if isinstance(segs, list):
            return " ".join(str(s.get("text", "")).strip() for s in segs if isinstance(s, dict)).strip()
        return ""
    s = str(raw).strip()
    if not s or s == "[]":
        return ""
    if s.startswith("[") and "text=" in s:
        parts = TEXT_FIELD_RE.findall(s)
        if parts:
            cleaned = []
            for t in parts:
                t = t.strip()
                if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
                    t = t[1:-1]
                cleaned.append(t.strip())
            return " ".join(cleaned).strip()
    return s

def clean_transcript(raw: str) -> str:
    if not raw:
        return ""
    txt = " ".join(raw.split()).strip()
    if not txt:
        return ""
    if META_RE.search(txt) and len(txt.split()) <= 3:
        return ""
    if len(txt.split()) == 1 and txt.lower() in {"ja", "und", "also", "äh", "oh"}:
        return ""
    return txt

def is_meaningful_text(t: str, min_chars: int, min_words: int) -> bool:
    t = (t or "").strip()
    return bool(t) and len(t) >= min_chars and len(t.split()) >= min_words and re.search(r"[A-Za-zÄÖÜäöüß]", t)

def transcribe_chunk_with_whisper(samples: np.ndarray, sr: int) -> str:
    if not WHISPER_AVAILABLE or _WHISPER_MODEL is None:
        return ""
    if samples.size == 0:
        return ""
    peak = float(np.max(np.abs(samples)))
    if peak < WHISPER_MIN_PEAK:
        print(f"[WHISPER] below_min_peak peak={peak:.4f} th={WHISPER_MIN_PEAK:.4f}")
        return ""
    min_sec = max(0.0, float(WHISPER_MIN_SEC))
    if samples.size < int(sr * min_sec):
        pad = int(sr * min_sec) - samples.size
        samples = np.concatenate([samples, np.zeros(pad, dtype=np.float32)], axis=0)
    if sr != 16000:
        samples = resample_to_16k(samples, sr)
        if samples.size == 0:
            return ""
    try:
        if hasattr(_WHISPER_MODEL, "transcribe_float32"):
            raw = _WHISPER_MODEL.transcribe_float32(samples)
        elif hasattr(_WHISPER_MODEL, "transcribe"):
            raw = _WHISPER_MODEL.transcribe(samples)
        else:
            raw = _WHISPER_MODEL.transcribe_pcm16(to_int16(samples))
        txt = clean_transcript(_parse_whisper_out(raw))
        if txt:
            print(f"[WHISPER] text: {txt}")
        else:
            print("[WHISPER] raw→empty")
        return txt
    except KeyboardInterrupt:
        return ""
    except Exception as e:
        print(f"[WHISPER] transcription failed: {e}")
        return ""

# ---------- HTTP utils ----------

def _httpx_limits() -> httpx.Limits:
    return httpx.Limits(max_keepalive_connections=6, max_connections=12, keepalive_expiry=20.0)

def _timeout_short_http() -> httpx.Timeout:
    return httpx.Timeout(connect=2.5, read=4.0, write=3.0, pool=3.0)

def _timeout_normal() -> httpx.Timeout:
    t = min(max(5.0, OLLAMA_TIMEOUT_SEC), 120.0)
    return httpx.Timeout(connect=5.0, read=t, write=5.0, pool=5.0)

# ---------- Ollama helpers ----------

def _ollama_url(path: str) -> str:
    return f"http://{OLLAMA_HOST}:{OLLAMA_PORT}{path}"

def _ollama_options_for_prompt() -> dict:
    return {
        "temperature": OLLAMA_TEMPERATURE,
        "num_ctx": OLLAMA_NUM_CTX,
        "num_predict": OLLAMA_NUM_PREDICT,
        "top_k": OLLAMA_TOP_K,
        "top_p": OLLAMA_TOP_P,
        "repeat_penalty": OLLAMA_REPEAT_PENALTY,
    }

async def _post_with_retries(client: httpx.AsyncClient, url: str, body: dict, timeout: float) -> dict:
    delay = float(_env_float("APP_OLLAMA_RETRY_BASE_DELAY", 0.8))
    max_retries = int(_env_int("APP_OLLAMA_MAX_RETRIES", 4))
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = await client.post(url, json=body, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError, httpx.HTTPStatusError) as e:
            last_exc = e
            status = getattr(e, "response", None).status_code if getattr(e, "response", None) else None
            retryable = status in (429, 500, 502, 503) or isinstance(
                e, (httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError, httpx.ConnectError)
            )
            print(f"[OLLAMA] attempt {attempt} failed (status={status}): {e}")
            if attempt >= max_retries or not retryable:
                break
            await asyncio.sleep(delay)
            delay *= 2.0
    raise RuntimeError(f"Ollama request failed after {max_retries} attempts: {last_exc}")

async def _ollama_available() -> bool:
    try:
        async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_short_http()) as c:
            r = await c.get(_ollama_url("/api/tags"))
            r.raise_for_status()
            return True
    except Exception:
        return False

async def ollama_generate_prompt(client: httpx.AsyncClient, user_text: str) -> str:
    sys = OLLAMA_SYS_PROMPT
    payload = {
        "user_text": (user_text or "").strip(),
        "constraints": {"no_meta": True, "max_sentences": 2, "avoid_sensitive": True},
        "output_hint": "One compact image prompt, no explanations.",
    }
    prompt_text = f"<<SYS>>{sys}<</SYS>>\n\nINPUT_JSON:\n{json.dumps(payload, ensure_ascii=False)}\n\nOUTPUT:\n"
    body = {"model": OLLAMA_MODEL, "prompt": prompt_text, "stream": False, "options": _ollama_options_for_prompt()}
    data = await _post_with_retries(client, _ollama_url("/api/generate"), body, timeout=float(OLLAMA_TIMEOUT_SEC))
    return (data.get("response") or "").strip()

# ---------- Global state ----------

@dataclass
class PipelineState:
    running: bool = False
    shutting_down: bool = False
    task: Optional[asyncio.Task] = None
    listeners: List[asyncio.Queue] = field(default_factory=list)
    actual_sr: int = 16000
    device_used_index: Optional[int] = None
    device_used_name: Optional[str] = None
    last_prompt: Optional[str] = None
    last_llm_error: Optional[str] = None
    last_llm_run_ts: float = 0.0
    bg_tasks: Set[asyncio.Task] = field(default_factory=set)
    ollama_ready_at: float = 0.0
    last_pending_text: Optional[str] = None
    start_ts: float = 0.0
    image_backend_name: str = _env_str("IMAGE_BACKEND", "comfyui").lower()
    allow_cloud: bool = _env_bool01("ALLOW_CLOUD_IMAGE_BACKEND", 0)
    image_width: int = APP_IMAGE_WIDTH
    image_height: int = APP_IMAGE_HEIGHT
    negative_prompt: str = ""
    active_workflow: Optional[str] = None
    audio_stream: Any = None
    audio_stopped_broadcasted: bool = False
    style_positive: Optional[str] = None

STATE = PipelineState()
STOP_DEBOUNCE_SEC = float(os.getenv("APP_STOP_DEBOUNCE_SEC", "2.0") or "2.0")

async def safe_stop_audio_stream() -> None:
    if getattr(STATE, "audio_stream", None) is not None:
        with contextlib.suppress(Exception):
            STATE.audio_stream.stop()
        with contextlib.suppress(Exception):
            STATE.audio_stream.close()
        STATE.audio_stream = None
    if not STATE.audio_stopped_broadcasted:
        await broadcast("status", "audio_stream_stopped")
        STATE.audio_stopped_broadcasted = True

BACKEND: Optional[ImageBackend] = None

def sse_format(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"

async def broadcast(event: str, data: str) -> None:
    if STATE.shutting_down:
        return
    for q in list(STATE.listeners):
        with contextlib.suppress(Exception):
            await q.put(sse_format(event, data))

_context_buffer: deque[str] = deque(maxlen=CONTEXT_MAX_SEGMENTS)

async def _close_sse_listeners(timeout: float = 0.25) -> None:
    listeners = getattr(STATE, "listeners", None)
    if not listeners:
        return
    queues: List[asyncio.Queue] = list(listeners)

    async def _signal(q: asyncio.Queue) -> None:
        try:
            q.put_nowait("")
        except Exception:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(q.put(""), timeout=timeout)

    tasks = [asyncio.create_task(_signal(q)) for q in queues]
    with contextlib.suppress(Exception):
        await asyncio.wait(tasks, timeout=timeout + 0.25)
    try:
        listeners.clear()
    except Exception:
        setattr(STATE, "listeners", [])

def update_context_buffer(text: str) -> str:
    _context_buffer.append(text)
    ctx = " ".join(_context_buffer)
    if len(ctx) > CONTEXT_MAX_CHARS:
        ctx = ctx[-CONTEXT_MAX_CHARS:]
    return ctx

def _log_effective_config() -> None:
    print(
        "[CONFIG]",
        f"env_file= {ENV_PATH or '(none)'}",
        "| audio:",
        f"sr={SAMPLE_RATE} frame_ms={FRAME_MS} stream_lat={APP_STREAM_LATENCY_SEC}",
        "| vad:",
        f"disable={DISABLE_VAD} rms_th={RMS_VAD_THRESHOLD}",
        "| snap:",
        f"snapshot_sec={SNAPSHOT_SEC} min_buf_sec={MIN_BUF_SEC} max_sil_ms={MAX_SILENCE_MS} max_seg={MAX_SEGMENT_SEC}",
        "| whisper:",
        f"min_sec={WHISPER_MIN_SEC} min_peak={WHISPER_MIN_PEAK} lang={WHISPER_LANGUAGE}",
        "| text:",
        f"min_chars={TEXT_MIN_CHARS} min_words={TEXT_MIN_WORDS} force_meaningful={FORCE_MEANINGFUL_CHECK}",
        "| llm:",
        f"interval={LLM_INTERVAL_SEC}s model={OLLAMA_MODEL}",
        "| image:",
        f"backend={STATE.image_backend_name} allow_cloud={STATE.allow_cloud} out={OUTPUT_DIR} default={APP_IMAGE_WIDTH}x{APP_IMAGE_HEIGHT}",
        "| sse:",
        f"tick={APP_SSE_TICK_SEC}s",
    )

# ---------- Runtime-aware backend wrapper ----------

def build_image_backend_rt(backend_name: Optional[str] = None, allow_cloud: Optional[bool] = None) -> ImageBackend:
    wanted_backend = (backend_name or STATE.image_backend_name or _env_str("IMAGE_BACKEND", "comfyui")).lower()
    allowed = (STATE.allow_cloud if allow_cloud is None else bool(allow_cloud))
    old_backend = os.environ.get("IMAGE_BACKEND")
    old_allow = os.environ.get("ALLOW_CLOUD_IMAGE_BACKEND")
    try:
        os.environ["IMAGE_BACKEND"] = wanted_backend
        os.environ["ALLOW_CLOUD_IMAGE_BACKEND"] = "1" if allowed else "0"
        be = build_image_backend()
        return be
    finally:
        if old_backend is None:
            with contextlib.suppress(Exception):
                del os.environ["IMAGE_BACKEND"]
        else:
            os.environ["IMAGE_BACKEND"] = old_backend
        if old_allow is None:
            with contextlib.suppress(Exception):
                del os.environ["ALLOW_CLOUD_IMAGE_BACKEND"]
        else:
            os.environ["ALLOW_CLOUD_IMAGE_BACKEND"] = old_allow

def rel_for_ui_path(p: Path) -> str:
    return Path(p).name

# ---------- Workflow helpers ----------

def _list_workflow_files() -> List[WorkflowItem]:
    items: List[WorkflowItem] = []
    if not WORKFLOWS_DIR.exists():
        return items
    for p in WORKFLOWS_DIR.iterdir():
        if p.is_file() and p.suffix.lower() == ".json":
            items.append(WorkflowItem(name=p.stem, filename=p.name))
    items.sort(key=lambda it: it.name.lower())
    return items

def _ensure_workflow_exists(filename: str) -> Path:
    candidate = (WORKFLOWS_DIR / filename).resolve()
    if WORKFLOWS_DIR not in candidate.parents and candidate != WORKFLOWS_DIR:
        raise FileNotFoundError("Invalid workflow path")
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError("Workflow not found")
    return candidate

def _apply_active_workflow_if_local() -> None:
    if LocalComfyBackend is None or BACKEND is None:
        return
    if not isinstance(BACKEND, LocalComfyBackend):
        return
    fname = getattr(STATE, "active_workflow", None)
    if not fname:
        return
    wf_path = (WORKFLOWS_DIR / fname).resolve()
    if not wf_path.exists() or not wf_path.is_file():
        print(f"[WF] active_workflow missing on disk: {wf_path}")
        return
    try:
        if WORKFLOWS_DIR not in wf_path.parents and wf_path != WORKFLOWS_DIR:
            print(f"[WF] invalid workflow path escape blocked: {wf_path}")
            return
        BACKEND.cfg.workflow_path = wf_path  # type: ignore[attr-defined]
        print(f"[WF] applied workflow to LocalComfyBackend: {wf_path.name}")
    except Exception as e:
        print(f"[WF] failed to apply active workflow: {e}")

# ---------- Audio transcription loop ----------

async def audio_transcription_loop() -> None:
    sr = int(SAMPLE_RATE)
    frame_len = max(1, int(sr * (FRAME_MS / 1000.0)))
    STATE.audio_stopped_broadcasted = False

    try:
        device_index = pick_input_device(AUDIO_DEVICE_PREF)
        sd.default.device = (device_index, None)
        device_name = sd.query_devices(device_index).get("name", f"dev{device_index}")
        sd.default.samplerate = sr
        sd.default.channels = 1
    except Exception as e:
        print(f"[AUDIO] device setup failed: {e}")
        await broadcast("status", f"audio_error:{e}")
        STATE.running = False
        return

    STATE.device_used_index = device_index
    STATE.device_used_name = device_name
    STATE.actual_sr = sr
    await broadcast("status", f"audio_device:{device_name or device_index}")

    buf = np.zeros(0, dtype=np.float32)
    speaking = False
    last_voice_ts = 0.0
    first_snapshot_deadline = time.time() + float(FIRST_SNAPSHOT_DEADLINE_SEC)

    q: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=100)

    def callback(indata, frames, time_info, status):
        if not STATE.running:
            return
        mono = np.asarray(indata[:, 0], dtype=np.float32)
        try:
            q.put_nowait(mono.copy())
        except asyncio.QueueFull:
            with contextlib.suppress(Exception):
                _ = q.get_nowait()
            with contextlib.suppress(Exception):
                q.put_nowait(mono.copy())

    stream = sd.InputStream(
        samplerate=sr,
        channels=1,
        dtype="float32",
        blocksize=frame_len,
        callback=callback,
        latency=APP_STREAM_LATENCY_SEC,
    )
    STATE.audio_stream = stream

    try:
        stream.start()
        print("[AUDIO] stream started")
        await broadcast("status", "audio_stream_started")

        while STATE.running and not STATE.shutting_down:
            try:
                frame = await asyncio.wait_for(q.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue

            if not STATE.running:
                break

            buf = np.concatenate([buf, frame], axis=0)
            now = time.time()

            vad_ok = True
            if not DISABLE_VAD:
                vad_ok = rms_vad(frame, rms_threshold=RMS_VAD_THRESHOLD)

            if vad_ok:
                last_voice_ts = now
                if not speaking:
                    speaking = True

            have_min_buf = buf.size >= int(SAMPLE_RATE * MIN_BUF_SEC)
            silence_exceeded = (now - last_voice_ts) * 1000.0 >= MAX_SILENCE_MS
            segment_too_long = buf.size >= int(SAMPLE_RATE * MAX_SEGMENT_SEC)
            first_deadline_hit = now >= first_snapshot_deadline and have_min_buf
            should_snapshot = have_min_buf and (silence_exceeded or segment_too_long or first_deadline_hit)

            if should_snapshot:
                snap = buf.copy()
                buf = np.zeros(0, dtype=np.float32)
                speaking = False
                first_snapshot_deadline = now + float(SNAPSHOT_SEC)

                if not STATE.running:
                    break

                def _do_transcribe(arr: np.ndarray, sample_rate: int) -> str:
                    return transcribe_chunk_with_whisper(arr, sample_rate)

                try:
                    text = await asyncio.to_thread(_do_transcribe, snap, sr)
                except Exception as e:
                    print(f"[AUDIO] transcription exception: {e}")
                    text = ""

                text = (text or "").strip()
                if text and STATE.running:
                    await broadcast("transcript", text)
                    if FORCE_MEANINGFUL_CHECK and not is_meaningful_text(text, TEXT_MIN_CHARS, TEXT_MIN_WORDS):
                        pass
                    else:
                        ctx = update_context_buffer(text)
                        if not OLLAMA_DISABLED and (time.time() - STATE.last_llm_run_ts) >= float(LLM_INTERVAL_SEC):
                            if not STATE.running:
                                break
                            STATE.last_llm_run_ts = time.time()
                            task = asyncio.create_task(run_llm_and_image(ctx))
                            STATE.bg_tasks.add(task)
                            def _done_cb(t: asyncio.Task):
                                with contextlib.suppress(Exception):
                                    STATE.bg_tasks.discard(t)
                            task.add_done_callback(_done_cb)

            max_keep = int(SAMPLE_RATE * MAX_SEGMENT_SEC)
            if buf.size > (max_keep * 2):
                buf = buf[-max_keep:]

        print("[AUDIO] loop exiting")
    except asyncio.CancelledError:
        print("[AUDIO] loop cancelled")
    except Exception as e:
        print(f"[AUDIO] loop crashed: {e}")
        await broadcast("status", f"audio_loop_error:{e}")
    finally:
        await safe_stop_audio_stream()

# ---------- LLM + Image ----------

async def run_llm_and_image(text: str) -> None:
    if await _ollama_available() is False:
        await broadcast("status", "ollama_unavailable")
        STATE.last_llm_error = "ollama_unavailable"
        return
    if BACKEND is None:
        await broadcast("status", "image_backend_not_initialized")
        STATE.last_llm_error = "image_backend_not_initialized"
        return
    async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_normal()) as client:
        try:
            img_prompt = await ollama_generate_prompt(client, text)
            if not img_prompt:
                STATE.last_llm_error = "llm_empty_response"
                await broadcast("status", "llm_empty_response")
                return
            eff_prompt = merge_style_prompt(img_prompt, getattr(STATE, "style_positive", None))

            STATE.last_prompt = eff_prompt
            await broadcast("llm_prompt", eff_prompt)
            await broadcast("status", "llm_ok")
            try:
                if LocalComfyBackend is not None and isinstance(BACKEND, LocalComfyBackend):
                    if hasattr(BACKEND, "cfg") and hasattr(BACKEND.cfg, "negative"):
                        setattr(BACKEND.cfg, "negative", STATE.negative_prompt or "")
                path = await _generate_with_negative_support(
                    prompt=eff_prompt,
                    width=STATE.image_width,
                    height=STATE.image_height,
                    negative=STATE.negative_prompt,
                )
                path = ensure_in_output_dir(path)
                rel = rel_for_ui_path(path)
                await broadcast("image", rel)
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[IMAGE] generation error: {e}\n{tb}")
                await broadcast("status", f"image_error:{e}")
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[PIPELINE] llm/image pipeline error: {e}\n{tb}")
            STATE.last_llm_error = f"pipeline_error:{e}"
            await broadcast("status", f"pipeline_error:{e}")

# ---------- Helpers: negative prompt passing ----------

def _merge_negative_into_prompt(prompt: str, negative: str) -> str:
    p = (prompt or "").strip()
    n = (negative or "").strip()
    if not n:
        return p
    return f"{p}\n-- negative: {n}"

def _inline_negative_phrases(positive: str, negative: str) -> str:
    base = (positive or "").strip()
    neg = (negative or "").strip()
    if not base or not neg:
        return base

    parts_raw = re.split(r"[,\n;]+", neg)
    items = []
    for it in parts_raw:
        t = it.strip()
        if not t:
            continue
        t = re.sub(r"^(no|kein(?:e|en|er)?|keine|without|exclude|vermeide|ohne)\s+", "", t, flags=re.I).strip()
        if t:
            items.append(t)

    seen: Set[str] = set()
    uniq: List[str] = []
    for it in items:
        key = it.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(it)

    if not uniq:
        return base

    en_parts = [f"without {it}" for it in uniq]
    en_parts += [f"avoid {it}" for it in uniq[:2]]
    en_parts += [f"no {it}" for it in uniq[:2]]
    de_parts = [f"ohne {it}" for it in uniq[:2]]

    inline_clause = "; ".join(en_parts + de_parts)
    merged = f"{base}\nConstraints: {inline_clause}."
    return merged

POLLINATIONS_INLINE_NEG = _env_bool01("POLLINATIONS_INLINE_NEG", 1)

async def _generate_with_negative_support(prompt: str, width: int, height: int, negative: str) -> Path:
    if BACKEND is None:
        raise RuntimeError("image_backend_not_initialized")

    try:
        _apply_active_workflow_if_local()
    except Exception as e:
        print(f"[WF] apply before generate failed: {e}")

    kwargs: Dict[str, Any] = {"width": width, "height": height}

    backend_cls = type(BACKEND).__name__
    neg_txt = (negative or "").strip()
    use_inline_for_poll = bool(POLLINATIONS_INLINE_NEG and neg_txt and ("pollinations" in backend_cls.lower()))

    eff_prompt = prompt
    if use_inline_for_poll:
        eff_prompt = _inline_negative_phrases(prompt, neg_txt)

    print(
        f"[IMAGE REQ] backend={backend_cls} size={width}x{height} "
        f"has_negative={(neg_txt != '')} inline_for_poll={use_inline_for_poll} "
        f"neg_sample='{neg_txt[:64]}'"
    )

    try:
        return await BACKEND.generate(eff_prompt, negative=(neg_txt or ""), **kwargs)  # type: ignore[arg-type]
    except TypeError:
        merged = _merge_negative_into_prompt(eff_prompt, neg_txt)
        return await BACKEND.generate(merged, **kwargs)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[IMAGE] backend.generate failed: {e}\n{tb}")
        raise

# ---------- FastAPI app & lifespan ----------

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[ENV] loaded from: {ENV_PATH or '(env vars only)'}")
    _log_effective_config()
    init_whisper_model()

    # Verify REFS_DIR early
    try:
        REFS_DIR.mkdir(parents=True, exist_ok=True)
        tfile = REFS_DIR / ".__writetest.tmp"
        tfile.write_text("ok", encoding="utf-8")
        tfile.unlink(missing_ok=True)
        print(f"[STYLE] refs_dir ready: {REFS_DIR}")
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[STYLE] refs_dir not writable: {REFS_DIR}, err={e}\n{tb}")

    global BACKEND
    try:
        _assert_image_backend_host()
        BACKEND = build_image_backend_rt()
        print(f"[BACKEND] initialized: {type(BACKEND).__name__}")
    except Exception as e:
        BACKEND = None
        tb = traceback.format_exc()
        print(f"[BACKEND] initialization failed: {e}\n{tb}")

    try:
        dw, dh = _backend_default_size(STATE.image_backend_name)
        STATE.image_width = STATE.image_width or dw
        STATE.image_height = STATE.image_height or dh
    except Exception:
        STATE.image_width = APP_IMAGE_WIDTH
        STATE.image_height = APP_IMAGE_HEIGHT

    STATE.ollama_ready_at = time.time() + (WARMUP_GRACE_SEC if WARMUP_ENABLE else 0.0)

    # Optional style_engine ensure_dirs()
    if style_engine is not None:
        try:
            if hasattr(style_engine, "ensure_dirs"):
                style_engine.ensure_dirs()
                print("[STYLE] style_engine.ensure_dirs() ok")
            else:
                print("[STYLE] style_engine.ensure_dirs() not present; skipping")
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[STYLE] ensure_dirs failed: {e}\n{tb}")

    warmup_task: Optional[asyncio.Task] = None
    if WARMUP_ENABLE:
        async def _silent_ollama_warmup():
            try:
                payload = {"model": OLLAMA_MODEL, "prompt": WARMUP_PROMPT, "stream": False, "options": {"temperature": 0.1, "num_predict": 32}}
                async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_short_http()) as c:
                    with contextlib.suppress(Exception):
                        await c.get(_ollama_url("/api/tags"))
                async with httpx.AsyncClient(limits=_httpx_limits(), timeout=httpx.Timeout(WARMUP_TIMEOUT_SEC)) as client:
                    await client.post(_ollama_url("/api/generate"), json=payload)
                print("[WARMUP] Ollama warmup ok.")
            except Exception as e:
                print(f"[WARMUP] failed: {e}")
        warmup_task = asyncio.create_task(_silent_ollama_warmup())
    try:
        yield
    finally:
        STATE.shutting_down = True
        STATE.running = False
        if STATE.task:
            STATE.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await STATE.task
            STATE.task = None
        for t in list(STATE.bg_tasks):
            t.cancel()
        with contextlib.suppress(Exception):
            await asyncio.gather(*list(STATE.bg_tasks), return_exceptions=True)
        STATE.bg_tasks.clear()
        await _close_sse_listeners()
        if warmup_task:
            warmup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await warmup_task
        print("[LIFESPAN] cleanup done")

app = FastAPI(lifespan=lifespan)

# Static mounts
app.mount("/static", StaticFiles(directory=str(OUTPUT_DIR), html=False), name="static")
if not any(route for route in app.router.routes if getattr(route, "path", "") == "/workflows"):
    app.mount("/workflows", StaticFiles(directory=str(WORKFLOWS_DIR), html=False), name="workflows")

web_dir = Path("web").resolve()
if web_dir.exists():
    app.mount("/web", StaticFiles(directory=str(web_dir), html=True), name="web")

@app.get("/", include_in_schema=False)
async def root_redirect():
    if web_dir.exists():
        return RedirectResponse(url="/web/index.html", status_code=307)
    return JSONResponse({"ok": True, "msg": "Web UI not found, use /static for images or API endpoints."})

# ---------- Status/Health ----------

@app.get("/ping")
async def ping():
    return {"ok": True}

@app.get("/status")
async def status():
    return {"ok": True, "running": STATE.running, "shutting_down": STATE.shutting_down}

@app.get("/health", response_model=HealthReport)
async def health() -> HealthReport:
    ollama_ok = await _ollama_available()
    return HealthReport(
        ollama_ok=ollama_ok,
        image_backend=STATE.image_backend_name,
        allow_cloud=STATE.allow_cloud,
        output_dir=str(OUTPUT_DIR),
        output_dir_exists=OUTPUT_DIR.exists(),
        last_prompt=STATE.last_prompt,
        last_llm_error=STATE.last_llm_error,
        pollinations_key_present=bool(_env_str("POLLINATIONS_SECRET", "")),
    )

@app.get("/config")
async def get_config():
    wpath = WHISPER_MODEL_PATH
    masked = (wpath[:3] + "..." + wpath[-10:]) if wpath and len(wpath) > 16 else wpath

    try:
        def_w, def_h = _backend_default_size(STATE.image_backend_name)
    except Exception:
        def_w, def_h = APP_IMAGE_WIDTH, APP_IMAGE_HEIGHT

    backend_lower = (STATE.image_backend_name or "").strip().lower()
    is_comfy = backend_lower == "comfyui"
    app_disable_comfy_env = os.getenv("APP_DISABLE_COMFYUI", "1").strip().lower()
    comfy_disabled = app_disable_comfy_env in {"1", "true", "yes", "on"}

    show_workflow_selector = bool(is_comfy and not comfy_disabled)

    return {
        "env_file": ENV_PATH or "(env vars only)",
        "audio": {"device_pref": AUDIO_DEVICE_PREF, "sample_rate": SAMPLE_RATE, "frame_ms": FRAME_MS, "stream_latency_sec": APP_STREAM_LATENCY_SEC},
        "vad": {"disable_vad": DISABLE_VAD, "rms_threshold": RMS_VAD_THRESHOLD},
        "snapshot": {"snapshot_sec": SNAPSHOT_SEC, "min_buf_sec": MIN_BUF_SEC, "max_silence_ms": MAX_SILENCE_MS, "max_segment_sec": MAX_SEGMENT_SEC, "first_snapshot_deadline_sec": FIRST_SNAPSHOT_DEADLINE_SEC},
        "whisper": {"model_path": masked, "language": WHISPER_LANGUAGE, "threads": WHISPER_THREADS, "temperature": WHISPER_TEMPERATURE, "min_sec": WHISPER_MIN_SEC, "min_peak": WHISPER_MIN_PEAK},
        "text": {"min_chars": TEXT_MIN_CHARS, "min_words": TEXT_MIN_WORDS, "force_meaningful": FORCE_MEANINGFUL_CHECK},
        "context": {"max_segments": CONTEXT_MAX_SEGMENTS, "max_chars": CONTEXT_MAX_CHARS},
        "ollama": {"host": OLLAMA_HOST, "port": OLLAMA_PORT, "model": OLLAMA_MODEL, "temperature": OLLAMA_TEMPERATURE, "timeout_sec": OLLAMA_TIMEOUT_SEC, "interval_sec": LLM_INTERVAL_SEC, "disabled": OLLAMA_DISABLED},
        "image": {
            "backend": STATE.image_backend_name,
            "allow_cloud": STATE.allow_cloud,
            "output_dir": str(OUTPUT_DIR),
            "width_default": def_w,
            "height_default": def_h,
            "current_width": STATE.image_width,
            "current_height": STATE.image_height,
            "negative_prompt": STATE.negative_prompt,
            "style_positive": STATE.style_positive,
            "active_workflow": STATE.active_workflow,
            "show_workflow_selector": show_workflow_selector,
            "comfy": {
                "enabled": bool(is_comfy and not comfy_disabled),
                "disabled_via_env": comfy_disabled
            }
        },
        "sse": {"tick_sec": APP_SSE_TICK_SEC},
    }

# ---------- Workflows API ----------

@app.get("/api/workflows", response_model=WorkflowList)
async def api_list_workflows() -> WorkflowList:
    items = _list_workflow_files()
    return WorkflowList(items=items)

@app.post("/api/settings/workflow")
async def api_select_workflow(payload: WorkflowSelect):
    try:
        _ensure_workflow_exists(payload.filename)
    except FileNotFoundError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    STATE.active_workflow = payload.filename

    try:
        _apply_active_workflow_if_local()
    except Exception as e:
        print(f"[WF] hot-apply error: {e}")

    await broadcast("status", f"workflow:{STATE.active_workflow}")
    return {"ok": True, "active_workflow": STATE.active_workflow}

# ---------- SSE: /events ----------

@app.get("/events")
async def events(request: Request):
    async def gen():
        q: asyncio.Queue[str] = asyncio.Queue()
        STATE.listeners.append(q)
        try:
            await q.put(sse_format("status", "connected"))
            hb = max(0.25, float(APP_SSE_TICK_SEC))
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=hb)
                except asyncio.TimeoutError:
                    yield sse_format("status", "hb").encode("utf-8")
                    continue
                if msg == "":
                    break
                yield msg.encode("utf-8")
        except asyncio.CancelledError:
            pass
        finally:
            with contextlib.suppress(Exception):
                if q in STATE.listeners:
                    STATE.listeners.remove(q)
    return StreamingResponse(gen(), media_type="text/event-stream")

# ---------- Audio/info routes ----------

@app.get("/audio/devices")
def audio_devices():
    try:
        devs = sd.query_devices()
        ins = [
            {"index": i, "name": d.get("name"), "max_input": d.get("max_input_channels", 0)}
            for i, d in enumerate(devs)
            if d.get("max_input_channels", 0) > 0
        ]
        return {"input_devices": ins}
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[AUDIO] devices error: {e}\n{tb}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/audio/probe")
def audio_probe():
    sr = SAMPLE_RATE
    dur = 0.5
    frames = int(sr * dur)
    try:
        idx = pick_input_device(AUDIO_DEVICE_PREF)
        sd.default.device = (idx, None)
        sd.default.samplerate = sr
        sd.default.channels = 1
        data = sd.rec(frames, samplerate=sr, channels=1, dtype="float32")
        sd.wait()
        mono = np.asarray(data[:, 0], dtype=np.float32)
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        rms = float(np.sqrt(np.mean(mono * mono))) if mono.size else 0.0
        return {"device_index": idx, "sample_rate": sr, "frames": int(mono.size), "peak": round(peak, 4), "rms": round(rms, 4)}
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[AUDIO] probe error: {e}\n{tb}")
        return JSONResponse({"error": str(e)}, status_code=500)

# ---------- Control routes ----------

@app.post("/start", response_class=PlainTextResponse)
async def start_pipeline():
    print("[HTTP] /start called")
    if STATE.running:
        print("[HTTP] /start ignored (already running)")
        return PlainTextResponse("already running", status_code=200)
    if STATE.shutting_down:
        return PlainTextResponse("shutting_down", status_code=409)
    STATE.running = True
    STATE.start_ts = time.time()
    STATE.task = asyncio.create_task(audio_transcription_loop())
    await broadcast("status", "server_start_recording")
    return PlainTextResponse("started")

@app.post("/stop", response_class=PlainTextResponse)
async def stop_pipeline():
    print("[HTTP] /stop called")
    STATE.running = False
    await safe_stop_audio_stream()
    if STATE.task:
        STATE.task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            try:
                await asyncio.wait_for(STATE.task, timeout=1.5)
            except asyncio.TimeoutError:
                pass
        STATE.task = None
    await broadcast("status", "audio_stopped")
    print("[HTTP] /stop completed")
    return PlainTextResponse("stopped")

@app.post("/shutdown", response_class=PlainTextResponse)
async def shutdown_server():
    print("[HTTP] /shutdown called")
    STATE.shutting_down = True
    try:
        await stop_pipeline()
    except Exception:
        pass
    for t in list(STATE.bg_tasks):
        t.cancel()
    with contextlib.suppress(Exception):
        await asyncio.gather(*list(STATE.bg_tasks), return_exceptions=True)
    STATE.bg_tasks.clear()
    await broadcast("status", "server_stopped")
    await _close_sse_listeners()
    asyncio.create_task(_exit_after_delay())
    return PlainTextResponse("shutting down")

async def _exit_after_delay():
    await asyncio.sleep(0.2)
    os._exit(0)

# ---------- Ollama APIs ----------

@app.post("/api/ollama/generate")
async def api_ollama_generate(req: OllamaGenerateRequest):
    if await _ollama_available() is False:
        return JSONResponse({"error": "ollama_unavailable"}, status_code=503)
    body = {"model": req.model or OLLAMA_MODEL, "prompt": req.prompt, "stream": bool(req.stream), "options": req.options or {}}
    async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_normal()) as client:
        try:
            data = await _post_with_retries(client, _ollama_url("/api/generate"), body, timeout=float(OLLAMA_TIMEOUT_SEC))
            return {"response": data.get("response", "")}
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[OLLAMA] /generate error: {e}\n{tb}")
            return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/ollama/chat")
async def api_ollama_chat(req: OllamaChatRequest):
    if await _ollama_available() is False:
        return JSONResponse({"error": "ollama_unavailable"}, status_code=503)
    body = {"model": req.model or OLLAMA_MODEL, "messages": [m.model_dump() for m in req.messages], "stream": bool(req.stream), "options": req.options or {}}
    async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_normal()) as client:
        try:
            data = await _post_with_retries(client, _ollama_url("/api/chat"), body, timeout=float(OLLAMA_TIMEOUT_SEC))
            msg = (data.get("message") or {}).get("content", "") if isinstance(data, dict) else ""
            return {"response": msg}
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[OLLAMA] /chat error: {e}\n{tb}")
            return JSONResponse({"error": str(e)}, status_code=500)

# ---------- Plan → Prompt ----------

@app.post("/api/plan")
async def api_plan(req: PlanRequest):
    if await _ollama_available() is False:
        return JSONResponse({"error": "ollama_unavailable"}, status_code=503)
    sys = OLLAMA_SYS_PROMPT
    payload = {
        "user_text": (req.text or "").strip(),
        "constraints": {"no_meta": True, "max_sentences": 2, "avoid_sensitive": True},
        "output_hint": "One compact image prompt, no explanations.",
    }
    prompt_text = f"<<SYS>>{sys}<</SYS>>\n\nINPUT_JSON:\n{json.dumps(payload, ensure_ascii=False)}\n\nOUTPUT:\n"
    body = {"model": OLLAMA_MODEL, "prompt": prompt_text, "stream": False, "options": _ollama_options_for_prompt()}
    async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_normal()) as client:
        try:
            data = await _post_with_retries(client, _ollama_url("/api/generate"), body, timeout=float(OLLAMA_TIMEOUT_SEC))
            out = (data.get("response") or "").strip()
            if out:
                eff_prompt = merge_style_prompt(out, req.style_positive or getattr(STATE, "style_positive", None))
                STATE.last_prompt = eff_prompt
                await broadcast("llm_prompt", eff_prompt)
                if BACKEND is not None:
                    try:
                        if LocalComfyBackend is not None and isinstance(BACKEND, LocalComfyBackend):
                            if hasattr(BACKEND, "cfg") and hasattr(BACKEND.cfg, "negative"):
                                setattr(BACKEND.cfg, "negative", STATE.negative_prompt or "")
                        w = req.width if (req.width and req.width > 0) else STATE.image_width
                        h = req.height if (req.height and req.height > 0) else STATE.image_height
                        path = await _generate_with_negative_support(eff_prompt, width=w, height=h, negative=STATE.negative_prompt)
                        path = ensure_in_output_dir(path)
                        rel = rel_for_ui_path(path)
                        await broadcast("image", rel)
                    except Exception as e:
                        tb = traceback.format_exc()
                        print(f"[IMAGE] /api/plan generation error: {e}\n{tb}")
                        await broadcast("status", f"image_error:{e}")
            return {"prompt": out}
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[OLLAMA] /api/plan error: {e}\n{tb}")
            return JSONResponse({"error": str(e)}, status_code=500)

# ---------- Image: Direct (ComfyUI und Pollinations) ----------

@app.post("/api/image/direct", response_model=ImageResponse)
async def api_image_direct(req: DirectImageRequest):
    if BACKEND is None:
        return JSONResponse({"error": "image_backend_not_initialized"}, status_code=500)
    try:
        neg = (req.negative_prompt or STATE.negative_prompt or "").strip()
        if LocalComfyBackend is not None and isinstance(BACKEND, LocalComfyBackend):
            if hasattr(BACKEND, "cfg") and hasattr(BACKEND.cfg, "negative"):
                setattr(BACKEND.cfg, "negative", neg)
        style_src = req.style_positive if (req.style_positive and req.style_positive.strip()) else getattr(STATE, "style_positive", None)
        base_prompt = req.prompt.strip()
        eff_prompt = merge_style_prompt(base_prompt, style_src)

        w = req.width if (req.width and req.width > 0) else STATE.image_width
        h = req.height if (req.height and req.height > 0) else STATE.image_height
        path = await _generate_with_negative_support(
            prompt=eff_prompt,
            width=w,
            height=h,
            negative=neg,
        )
        path = ensure_in_output_dir(path)
        rel = rel_for_ui_path(path)
        await broadcast("image", rel)
        return ImageResponse(filename=path.name, relpath=rel, rel=rel, width=w, height=h)
    except PermissionError as e:
        return JSONResponse({"error": f"{e}"}, status_code=403)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[IMAGE] /api/image/direct error: {e}\n{tb}")
        return JSONResponse({"error": f"{e}"}, status_code=502)

# ---------- Image test endpoint ----------

@app.post("/api/image/test", response_model=ImageResponse)
async def api_image_test(req: ImageRequest):
    if BACKEND is None:
        return JSONResponse({"error": "image_backend_not_initialized"}, status_code=500)
    prompt = (req.prompt or "A colorful low-poly fox head, studio lighting, high detail, 3D render").strip()
    try:
        neg = (req.negative_prompt or STATE.negative_prompt or "").strip()
        if LocalComfyBackend is not None and isinstance(BACKEND, LocalComfyBackend):
            if hasattr(BACKEND, "cfg") and hasattr(BACKEND.cfg, "negative"):
                setattr(BACKEND.cfg, "negative", neg)
        style_src = req.style_positive if (req.style_positive and req.style_positive.strip()) else getattr(STATE, "style_positive", None)
        eff_prompt = merge_style_prompt(prompt, style_src)

        w = req.width if (req.width and req.width > 0) else STATE.image_width
        h = req.height if (req.height and req.height > 0) else STATE.image_height
        path = await _generate_with_negative_support(eff_prompt, width=w, height=h, negative=neg)
        path = ensure_in_output_dir(path)
        rel = rel_for_ui_path(path)
        await broadcast("image", rel)
        return ImageResponse(filename=path.name, relpath=rel, rel=rel, width=w, height=h)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[IMAGE] /api/image/test error: {e}\n{tb}")
        return JSONResponse({"error": f"{e}"}, status_code=502)

# ---------- Image backend switching & cloud toggle ----------

def _rebuild_backend(force_name: Optional[str] = None) -> ImageBackend:
    global BACKEND
    if force_name:
        STATE.image_backend_name = force_name.lower()
    new_backend = build_image_backend_rt(backend_name=STATE.image_backend_name, allow_cloud=STATE.allow_cloud)
    BACKEND = new_backend
    return new_backend

class ImageAllowCloudReq(BaseModel):
    allow: Optional[bool] = None
    allow_cloud: Optional[bool] = None

@app.post("/api/settings/image_allow_cloud")
async def api_image_allow_cloud(req: ImageAllowCloudReq):
    val = req.allow if req.allow is not None else req.allow_cloud
    if val is None:
        return JSONResponse({"ok": False, "error": "missing_field_allow"}, status_code=400)
    STATE.allow_cloud = bool(val)
    if STATE.image_backend_name == "pollinations" and not STATE.allow_cloud:
        STATE.image_backend_name = "comfyui"
    try:
        _rebuild_backend()
        try:
            _apply_active_workflow_if_local()
        except Exception as e:
            print(f"[WF] apply after allow_cloud switch failed: {e}")
        return {"ok": True, "allow_cloud": STATE.allow_cloud, "backend": STATE.image_backend_name}
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[BACKEND] allow_cloud switch error: {e}\n{tb}")
        return JSONResponse({"ok": False, "error": f"{e}"}, status_code=500)

@app.post("/api/settings/image_backend")
async def api_switch_image_backend(req: ImageBackendSwitch):
    target = req.backend.lower()
    if target not in {"comfyui", "pollinations"}:
        return JSONResponse({"ok": False, "error": "invalid_backend"}, status_code=400)
    if target == "pollinations" and not STATE.allow_cloud:
        return JSONResponse({"ok": False, "error": "not_allowed", "reason": "cloud_blocked"}, status_code=403)
    if target == "pollinations":
        secret = _env_str("POLLINATIONS_SECRET", "")
        if not secret:
            return JSONResponse({"ok": False, "error": "missing_secret", "reason": "missing_secret"}, status_code=400)
    try:
        cur_w, cur_h = STATE.image_width, STATE.image_height
        _rebuild_backend(force_name=target)
        if req.reset:
            def_w, def_h = _backend_default_size(target)
            STATE.image_width = def_w
            STATE.image_height = def_h
        else:
            STATE.image_width = cur_w
            STATE.image_height = cur_h

        try:
            _apply_active_workflow_if_local()
        except Exception as e:
            print(f"[WF] apply after backend switch failed: {e}")

        await broadcast("status", f"image_backend:{target}")
        return {"ok": True, "backend": target, "width": STATE.image_width, "height": STATE.image_height, "reset": bool(req.reset)}
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[BACKEND] switch error: {e}\n{tb}")
        return JSONResponse({"ok": False, "error": f"{e}"}, status_code=500)

# ---------- Image size, negative prompt & style prompt settings ----------

@app.get("/api/settings/image_size")
async def get_image_size():
    return {"width": STATE.image_width, "height": STATE.image_height}

@app.post("/api/settings/image_size")
async def set_image_size(s: ImageSizeSettings):
    STATE.image_width = int(s.width)
    STATE.image_height = int(s.height)
    await broadcast("status", f"image_size:{STATE.image_width}x{STATE.image_height}")
    return {"ok": True, "width": STATE.image_width, "height": STATE.image_height}

@app.get("/api/settings/negative_prompt")
async def get_negative_prompt():
    return {"negative_prompt": STATE.negative_prompt}

@app.post("/api/settings/negative_prompt")
async def set_negative_prompt(s: NegativePromptSettings):
    txt = (s.negative_prompt or "").strip()
    STATE.negative_prompt = txt
    if BACKEND is not None and LocalComfyBackend is not None and isinstance(BACKEND, LocalComfyBackend):
        if hasattr(BACKEND, "cfg") and hasattr(BACKEND.cfg, "negative"):
            with contextlib.suppress(Exception):
                setattr(BACKEND.cfg, "negative", txt)
    await broadcast("status", "negative_prompt:updated")
    return {"ok": True, "negative_prompt": STATE.negative_prompt}

@app.get("/api/settings/style_positive")
async def get_style_positive():
    return {"style_positive": STATE.style_positive or ""}

@app.post("/api/settings/style_positive")
async def set_style_positive(s: StylePromptSettings):
    txt = (s.style_positive or "").strip()
    STATE.style_positive = txt or None
    await broadcast("status", "style_positive:updated")
    return {"ok": True, "style_positive": STATE.style_positive or ""}

# ---------- Style API (upload/save_url/build/reset) with enhanced compatibility ----------

def detect_image_format(data: bytes) -> Optional[str]:
    """
    Detect image format safely.
    Returns lowercase format name among {'jpeg','png','webp','bmp'} or None.
    """
    if not data or len(data) < 4:
        return None
    try:
        from PIL import Image
        from io import BytesIO
        with Image.open(BytesIO(data)) as im:
            fmt = (im.format or "").lower()
            if fmt in {"jpeg", "png", "webp", "bmp"}:
                return fmt
            if fmt == "jpg":
                return "jpeg"
    except Exception:
        pass
    b = data
    if len(b) >= 2 and b[0:2] == b"\xFF\xD8":
        return "jpeg"
    if len(b) >= 8 and b[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if len(b) >= 12 and b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "webp"
    if len(b) >= 2 and b[:2] == b"BM":
        return "bmp"
    return None

REFS_DIR = Path("outputs/images/refs").resolve()
REFS_DIR.mkdir(parents=True, exist_ok=True)

def _sanitize_filename(name: str) -> str:
    name = (name or "").strip()
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.splitext(name)[0]).strip("_") or f"ref_{int(time.time()*1000)}"
    ext = os.path.splitext(name)[1].lower()
    if ext not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        ext = ".png"
    return base + ext

async def _maybe_await(fn, *args, **kwargs):
    if asyncio.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

class SaveUrlRequest(BaseModel):
    url: HttpUrl
    filename_hint: Optional[str] = None

class UploadResponse(BaseModel):
    reference_id: str
    path: str
    url_path: str

class SaveUrlResponse(UploadResponse):
    pass

class StyleBuildRequest(BaseModel):
    content_positive: Optional[str] = None
    style_text_prompt: Optional[str] = None
    reference_source: Optional[str] = Field(default=None, description="local_file|url_file|none")
    reference_id: Optional[str] = None
    use_local_style_features: bool = True
    use_ollama_vision: bool = False
    deactivate_all_styles: bool = False
    target_backend_name: Optional[str] = Field(default=None, description="comfy_local|comfy_remote|comfy_cloud|pollinations")
    ollama_vision_mode: Optional[str] = Field(default="local", description="local|remote|cloud")

class StyleBuildResponse(BaseModel):
    style_positive: str
    style_components: Dict[str, Any] = Field(default_factory=dict)
    merged_prompt_preview: Optional[str] = None
    reference_used: Optional[str] = None
    info: Optional[Dict[str, Any]] = None

def _log_style_engine_fn(fn_name: str) -> None:
    try:
        if style_engine is None:
            print(f"[STYLE] style_engine is None (fn={fn_name})")
            return
        fn = getattr(style_engine, fn_name, None)
        if fn is None:
            print(f"[STYLE] style_engine.{fn_name} not present")
            return
        sig = None
        try:
            sig = inspect.signature(fn)
        except Exception:
            pass
        print(f"[STYLE] found style_engine.{fn_name} sig={sig!s}")
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[STYLE] inspection error for {fn_name}: {e}\n{tb}")

def _local_save_reference(safe_name: str, raw: bytes) -> UploadResponse:
    REFS_DIR.mkdir(parents=True, exist_ok=True)
    out = REFS_DIR / safe_name
    out.write_bytes(raw)
    return UploadResponse(
        reference_id=safe_name,
        path=str(out),
        url_path=f"/static/{Path(out).name}",
    )

@app.post("/api/style/upload", response_model=UploadResponse)
async def api_style_upload(file: UploadFile = File(...)):
    if not file:
        return JSONResponse({"error": "no_file"}, status_code=400)
    try:
        raw = await file.read()
        print(f"[STYLE] upload received: name={file.filename!r} size={len(raw)}B")
        if len(raw) > 25 * 1024 * 1024:
            return JSONResponse({"error": "file_too_large"}, status_code=413)
        fmt = detect_image_format(raw)
        print(f"[STYLE] detect_image_format={fmt}")
        if fmt not in {"jpeg", "png", "webp", "bmp"}:
            return JSONResponse({"error": "not_an_image"}, status_code=400)
        safe_name = _sanitize_filename(file.filename or f"ref_{int(time.time()*1000)}.png")
        print(f"[STYLE] saving reference: safe_name={safe_name} refs_dir={REFS_DIR}")
        if style_engine is not None and hasattr(style_engine, "save_reference_from_bytes"):
            _log_style_engine_fn("save_reference_from_bytes")
            try:
                saved = await _maybe_await(style_engine.save_reference_from_bytes, safe_name, raw)
                if isinstance(saved, dict) and {"reference_id", "path", "url_path"} <= set(saved.keys()):
                    print(f"[STYLE] saved via engine: {saved}")
                    return UploadResponse(**saved)
                else:
                    print(f"[STYLE] engine returned unexpected structure; falling back to local save. got={saved}")
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[STYLE] save_reference_from_bytes failed: {e}\n{tb}")
        # Fallback: local save
        saved_local = _local_save_reference(safe_name, raw)
        print(f"[STYLE] saved locally: {saved_local}")
        return saved_local
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[STYLE] upload route exception: {e}\n{tb}")
        return JSONResponse({"error": "internal_error", "reason": f"{e}"}, status_code=500)

@app.post("/api/style/save_url", response_model=SaveUrlResponse)
async def api_style_save_url(req: SaveUrlRequest):
    if style_engine is None or not hasattr(style_engine, "save_reference_from_url"):
        return JSONResponse({"error": "style_engine_unavailable_or_missing_save_url"}, status_code=500)
    try:
        hint = _sanitize_filename(req.filename_hint) if req.filename_hint else None
        print(f"[STYLE] save_url: url={req.url} hint={hint} refs_dir={REFS_DIR}")
        _log_style_engine_fn("save_reference_from_url")
        saved = await _maybe_await(style_engine.save_reference_from_url, str(req.url), hint)
        print(f"[STYLE] saved (url): {saved}")
        if not isinstance(saved, dict) or not {"reference_id", "path", "url_path"} <= set(saved.keys()):
            print(f"[STYLE] unexpected return from save_reference_from_url: {saved}")
        return SaveUrlResponse(**saved)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[STYLE] save_url exception: {e}\n{tb}")
        return JSONResponse({"error": "download_or_save_failed", "reason": f"{e}"}, status_code=502)

def _infer_target_backend_name() -> str:
    b = (STATE.image_backend_name or "").strip().lower()
    if b == "pollinations":
        return "pollinations"
    return "comfy_local"

def _sanitize_for_style_engine(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure compatibility with StyleEngineRequest:
    - target_backend_name must be a non-empty string
    - remove ollama_vision_mode if the model forbids extras
    """
    payload = dict(payload)
    # Ensure target_backend_name
    tbn = payload.get("target_backend_name")
    if not isinstance(tbn, str) or not tbn.strip():
        payload["target_backend_name"] = _infer_target_backend_name()

    model = getattr(style_engine, "StyleEngineRequest", None) if style_engine else None
    if model is None:
        payload.pop("ollama_vision_mode", None)
        return payload

    valid_keys = set(getattr(model, "model_fields", {}).keys()) if hasattr(model, "model_fields") else None
    if valid_keys:
        to_drop = [k for k in payload.keys() if k not in valid_keys]
        for k in to_drop:
            payload.pop(k, None)
        if "target_backend_name" not in payload or not payload["target_backend_name"]:
            payload["target_backend_name"] = _infer_target_backend_name()
    else:
        payload.pop("ollama_vision_mode", None)

    try:
        model(**payload)
    except Exception as e:
        print(f"[STYLE] StyleEngineRequest validation failed (dict fallback): {e}")
    return payload

def _to_engine_req(req: StyleBuildRequest) -> Any:
    raw = {
        "content_positive": req.content_positive,
        "style_text_prompt": req.style_text_prompt,
        "reference_source": req.reference_source,
        "reference_id": req.reference_id,
        "use_local_style_features": req.use_local_style_features,
        "use_ollama_vision": req.use_ollama_vision,
        "deactivate_all_styles": req.deactivate_all_styles,
        "target_backend_name": req.target_backend_name,
        "ollama_vision_mode": (req.ollama_vision_mode or "local"),
    }
    payload = _sanitize_for_style_engine(raw)
    model = getattr(style_engine, "StyleEngineRequest", None) if style_engine else None
    if style_engine is None or model is None:
        return payload
    try:
        return model(**payload)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[STYLE] StyleEngineRequest init failed, using dict. err={e}\n{tb}")
        return payload

def _from_engine_result(res: Any) -> StyleBuildResponse:
    try:
        d = res.dict() if hasattr(res, "dict") else dict(res)
    except Exception:
        d = {}
    return StyleBuildResponse(
        style_positive=d.get("style_positive") or "",
        style_components=d.get("style_components") or {},
        merged_prompt_preview=d.get("merged_prompt_preview"),
        reference_used=d.get("reference_used") or d.get("ref_used"),
        info=d.get("info"),
    )

def _call_build_styles_compat(req_obj: Any) -> Any:
    """
    Call style_engine.build_styles with correct signature:
      - build_styles(req)
      - or build_styles(req, refs_dir)
    """
    if style_engine is None or not hasattr(style_engine, "build_styles"):
        raise RuntimeError("style_engine.build_styles not available")
    fn = style_engine.build_styles
    try:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
    except Exception:
        params = []
    try:
        if len(params) >= 2:
            print(f"[STYLE] calling build_styles(req, refs_dir)")
            return fn(req_obj, REFS_DIR)
        else:
            print(f"[STYLE] calling build_styles(req)")
            return fn(req_obj)
    except TypeError as e:
        print(f"[STYLE] build_styles signature TypeError: {e}. Retrying with req only.")
        return fn(req_obj)

def _fallback_style_positive(style_text_prompt: Optional[str]) -> str:
    """
    Conservative fallback ensuring visible 'comic/cartoon' style.
    """
    base_tokens = [
        "comic style", "hand-drawn", "cel shading", "bold black outlines",
        "flat colors", "high contrast", "inked lines", "cartoonish"
    ]
    user_tokens = [t.strip() for t in (style_text_prompt or "").split(",") if t.strip()]
    merged = base_tokens + user_tokens
    seen = set()
    dedup = []
    for t in merged:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            dedup.append(t)
    return ", ".join(dedup)

@app.post("/api/style/build", response_model=StyleBuildResponse)
async def api_style_build(req: StyleBuildRequest = Body(...)):
    if style_engine is None:
        return JSONResponse({"error": "style_engine_unavailable"}, status_code=500)
    try:
        print(f"[STYLE] build request (raw): {req.model_dump()}")
        eng_req = _to_engine_req(req)
        _log_style_engine_fn("build_styles")

        try:
            res = await _maybe_await(_call_build_styles_compat, eng_req)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[STYLE] build_styles failed: {e}\n{tb}")
            return JSONResponse({"error": "style_build_failed", "reason": f"{e}"}, status_code=502)

        api_res = _from_engine_result(res)
        if not api_res.style_positive.strip():
            api_res.style_positive = _fallback_style_positive(req.style_text_prompt)
            print(f"[STYLE] engine returned empty style; using fallback len={len(api_res.style_positive)}")
        else:
            print(f"[STYLE] build result: style_positive_len={len(api_res.style_positive)} ref_used={api_res.reference_used}")

        STATE.style_positive = (api_res.style_positive or "").strip() or None
        await broadcast("status", "style_positive:updated")
        return api_res
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[STYLE] build route exception: {e}\n{tb}")
        return JSONResponse({"error": "internal_error", "reason": f"{e}"}, status_code=500)

@app.post("/api/style/reset")
async def api_style_reset():
    STATE.style_positive = None
    await broadcast("status", "style_positive:updated")
    return {"ok": True, "style_positive": ""}

# ---------- Utility: Open-dir hint ----------

@app.get("/open_dir_hint")
async def open_dir_hint():
    return {"static_url": "/static/", "path": str(OUTPUT_DIR)}

# ---------- App entry ----------

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("APP_BIND_HOST", "127.0.0.1")
    port = int(os.getenv("APP_BIND_PORT", "8080"))
    uvicorn.run("app:app", host=host, port=port, reload=False)
