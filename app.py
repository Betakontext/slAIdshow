#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set

import httpx
import numpy as np
import sounddevice as sd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# ---- ENV helpers ----
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

# ---- Optional dotenv ----
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

# ---- Config: Audio ----
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

# ---- Whisper (pywhispercpp) ----
WHISPER_MODEL_PATH = _env_str("APP_WHISPER_MODEL_PATH", "")
WHISPER_LANGUAGE = _env_str("APP_WHISPER_LANGUAGE", "de")
WHISPER_THREADS = _env_int("APP_WHISPER_THREADS", 2)
WHISPER_TEMPERATURE = _env_float("APP_WHISPER_TEMPERATURE", 0.0)
WHISPER_MIN_SEC = _env_float("APP_WHISPER_MIN_SEC", 0.35)
WHISPER_MIN_PEAK = _env_float("APP_WHISPER_MIN_PEAK", 0.0009)

# ---- Text thresholds ----
TEXT_MIN_CHARS = _env_int("APP_TEXT_MIN_CHARS", 3)
TEXT_MIN_WORDS = _env_int("APP_TEXT_MIN_WORDS", 1)
FORCE_MEANINGFUL_CHECK = _env_bool01("APP_FORCE_MEANINGFUL_CHECK", 0)

CONTEXT_MAX_SEGMENTS = _env_int("APP_CONTEXT_MAX_SEGMENTS", 5)
CONTEXT_MAX_CHARS = _env_int("APP_CONTEXT_MAX_CHARS", 480)

# ---- Ollama ----
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
    "Du bist ein präziser Prompt-Designer für Bildgeneratoren. "
    "Erzeuge kurze, klare, fotografische oder illustrative Bild-Prompts, "
    "ohne Meta-Kommentare, in Deutsch.",
)

def assert_local(host: str) -> None:
    if host != "127.0.0.1":
        raise AssertionError(f"Only localhost allowed, got {host}")
assert_local(OLLAMA_HOST)

# ---- Image: Pollinations (serverseitiger Secret) ----
IMAGE_BACKEND = _env_str("IMAGE_BACKEND", "pollinations").lower()
ALLOW_CLOUD_IMAGE_BACKEND = _env_bool01("ALLOW_CLOUD_IMAGE_BACKEND", 0)
POLLINATIONS_API_BASE = _env_str("POLLINATIONS_API_BASE", "https://gen.pollinations.ai").rstrip("/")
POLLINATIONS_SECRET = _env_str("POLLINATIONS_SECRET", "")
POLLINATIONS_MODEL = _env_str("POLLINATIONS_MODEL", "") or None
POLLINATIONS_WIDTH = _env_int("POLLINATIONS_WIDTH", 1440)
POLLINATIONS_HEIGHT = _env_int("POLLINATIONS_HEIGHT", 900)
POLLINATIONS_NOLOGO = _env_bool01("POLLINATIONS_NOLOGO", 1)
POLLINATIONS_SEED = os.getenv("POLLINATIONS_SEED")
try:
    POLLINATIONS_SEED_INT: Optional[int] = int(POLLINATIONS_SEED) if POLLINATIONS_SEED is not None else None
except Exception:
    POLLINATIONS_SEED_INT = None

# New: v1 JSON toggle/size
POLLINATIONS_USE_V1 = _env_bool01("POLLINATIONS_USE_V1", 1)
POLLINATIONS_SIZE = _env_str("POLLINATIONS_SIZE", "")  # e.g. "1024x1024"

# ---- Output ----
OUTPUT_DIR = Path(_env_str("APP_OUTPUT_DIR", "./outputs/images")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Warmup (Ollama) ----
WARMUP_ENABLE = _env_bool01("APP_OLLAMA_WARMUP_ENABLE", 1)
WARMUP_PROMPT = _env_str("APP_OLLAMA_WARMUP_PROMPT", "Sag Hallo auf Deutsch.")
WARMUP_TIMEOUT_SEC = _env_float("APP_OLLAMA_WARMUP_TIMEOUT_SEC", 45.0)
WARMUP_MAX_RETRIES = _env_int("APP_OLLAMA_WARMUP_MAX_RETRIES", 3)
WARMUP_RETRY_DELAY = _env_float("APP_OLLAMA_WARMUP_RETRY_DELAY", 1.2)
WARMUP_GRACE_SEC = _env_float("APP_OLLAMA_WARMUP_GRACE_SEC", 10.0)

# ---- Pydantic payloads ----
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

class HealthReport(BaseModel):
    ollama_ok: bool
    image_backend: str
    allow_cloud: bool
    output_dir: str
    output_dir_exists: bool
    last_prompt: Optional[str] = None
    last_llm_error: Optional[str] = None
    pollinations_key_present: bool = False

# ---- Audio utils ----
def pick_input_device(prefer: Optional[str] = None) -> int:
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

# ---- Whisper init ----
WHISPER_AVAILABLE = True
try:
    from pywhispercpp.model import Model as WhisperModel  # type: ignore
except Exception as e:
    print(f"[WARN] could not import pywhispercpp: {e}")
    WhisperModel = None  # type: ignore
    WHISPER_AVAILABLE = False

_WHISPER_MODEL: Optional[WhisperModel] = None

def init_whisper_model() -> None:
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
            return " " .join(cleaned).strip()
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

# ---- HTTP utils ----
def _httpx_limits() -> httpx.Limits:
    return httpx.Limits(max_keepalive_connections=6, max_connections=12, keepalive_expiry=20.0)

def _timeout_short() -> httpx.Timeout:
    return httpx.Timeout(connect=2.5, read=4.0, write=3.0, pool=3.0)

def _timeout_normal() -> httpx.Timeout:
    t = min(max(5.0, OLLAMA_TIMEOUT_SEC), 120.0)
    return httpx.Timeout(connect=5.0, read=t, write=5.0, pool=5.0)

# ---- Ollama helpers ----
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
    delay = OLLAMA_RETRY_BASE_DELAY
    last_exc: Optional[Exception] = None
    for attempt in range(1, OLLAMA_MAX_RETRIES + 1):
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
            if attempt >= OLLAMA_MAX_RETRIES or not retryable:
                break
            await asyncio.sleep(delay)
            delay *= 2.0
    raise RuntimeError(f"Ollama request failed after {OLLAMA_MAX_RETRIES} attempts: {last_exc}")

async def _ollama_available() -> bool:
    try:
        async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_short()) as c:
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

# ---- Pollinations: GET builder (kept) ----
def _build_pollinations_image_url(
    api_base: str,
    prompt: str,
    model: Optional[str],
    width: Optional[int],
    height: Optional[int],
    nologo: bool,
    seed: Optional[int],
) -> str:
    from urllib.parse import quote, urlencode
    base = (api_base or "").rstrip("/")
    encoded_prompt = quote(prompt, safe="")
    url = f"{base}/image/{encoded_prompt}"
    params = {}
    if model:
        params["model"] = model
    if width:
        params["width"] = str(width)
    if height:
        params["height"] = str(height)
    if nlogo:
        params["nologo"] = "true"
    if seed is not None:
        params["seed"] = str(seed)
    if params:
        url = f"{url}?{urlencode(params)}"
    return url

# ---- Pollinations: GET (legacy) ----
async def fetch_pollinations_image_secure(prompt: str, out_dir: Path) -> Path:
    if IMAGE_BACKEND != "pollinations":
        raise RuntimeError("IMAGE_BACKEND!=pollinations")
    if not ALLOW_CLOUD_IMAGE_BACKEND:
        raise RuntimeError("Cloud image backend not allowed (ALLOW_CLOUD_IMAGE_BACKEND=1)")
    if not POLLINATIONS_SECRET:
        raise RuntimeError("POLLINATIONS_SECRET missing in .env")
    from urllib.parse import urlparse
    p = urlparse(POLLINATIONS_API_BASE)
    if not (p.scheme in ("http", "https") and p.netloc and " " not in POLLINATIONS_API_BASE):
        raise RuntimeError(f"invalid_api_base:{POLLINATIONS_API_BASE}")

    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"img_{uuid.uuid4().hex}.jpg"
    target = out_dir / filename

    url = _build_pollinations_image_url(
        POLLINATIONS_API_BASE,
        prompt,
        POLLINATIONS_MODEL,
        POLLINATIONS_WIDTH,
        POLLINATIONS_HEIGHT,
        POLLINATIONS_NOLOGO,
        POLLINATIONS_SEED_INT,
    )
    headers = {"Authorization": f"Bearer {POLLINATIONS_SECRET}"}
    timeout = httpx.Timeout(connect=8.0, read=90.0, write=8.0, pool=8.0)
    limits = httpx.Limits(max_keepalive_connections=10, max_connections=20, keepalive_expiry=30.0)

    async def _one(client: httpx.AsyncClient) -> bytes:
        print(f"[POLLINATIONS] GET {url}")
        r = await client.get(url, headers=headers, follow_redirects=True)
        ct = (r.headers.get("content-type") or "").lower()
        status = r.status_code
        if status >= 400:
            body_preview = ""
            try:
                body_json = r.json()
                body_preview = json.dumps(body_json)[:300]
            except Exception:
                body_preview = (r.text or "")[:300]
            raise httpx.HTTPStatusError(f"HTTP {status} {ct} body={body_preview}", request=r.request, response=r)
        if not ct.startswith("image/"):
            body_preview = ""
            try:
                body_json = r.json()
                body_preview = json.dumps(body_json)[:300]
            except Exception:
                body_preview = (r.text or "")[:300]
            raise httpx.HTTPStatusError(f"non-image ct={ct} body={body_preview}", request=r.request, response=r)
        return r.content

    delay = 1.0
    content: Optional[bytes] = None
    last_exc: Optional[Exception] = None
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        for attempt in range(1, 6):
            try:
                content = await _one(client)
                break
            except httpx.HTTPStatusError as e:
                code = e.response.status_code if e.response else None
                print(f"[POLLINATIONS] attempt {attempt} HTTP {code}: {e}")
                last_exc = e
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError) as e:
                print(f"[POLLINATIONS] attempt {attempt} net error: {e}")
                last_exc = e
            except Exception as e:
                print(f"[POLLINATIONS] attempt {attempt} failed: {e}")
                last_exc = e
            if attempt < 5:
                await asyncio.sleep(delay)
                delay *= 1.7

    if content is None:
        raise RuntimeError(f"pollinations_all_attempts_failed: {last_exc}")

    target.write_bytes(content)
    if target.stat().st_size < 1024:
        raise RuntimeError("pollinations_too_small_image")
    print(f"[POLLINATIONS] saved {target} ({target.stat().st_size} bytes)")
    return target

# ---- Pollinations v1 JSON and GET fallback ----
from base64 import b64decode

class _PollinationsV1Datum(BaseModel):
    b64_json: str
    revised_prompt: Optional[str] = None

class _PollinationsV1Response(BaseModel):
    created: int
    data: List[_PollinationsV1Datum]

def _size_from_wh(width: int, height: int) -> str:
    if width > 0 and height > 0:
        return f"{width}x{height}"
    return "1024x1024"

async def fetch_pollinations_v1_image(prompt: str, out_dir: Path) -> Path:
    if IMAGE_BACKEND != "pollinations":
        raise RuntimeError("IMAGE_BACKEND!=pollinations")
    if not ALLOW_CLOUD_IMAGE_BACKEND:
        raise RuntimeError("Cloud image backend not allowed (ALLOW_CLOUD_IMAGE_BACKEND=1)")
    if not POLLINATIONS_SECRET:
        raise RuntimeError("POLLINATIONS_SECRET missing in .env")
    size = POLLINATIONS_SIZE or _size_from_wh(POLLINATIONS_WIDTH, POLLINATIONS_HEIGHT)
    url = f"{POLLINATIONS_API_BASE.rstrip('/')}/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {POLLINATIONS_SECRET}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": POLLINATIONS_MODEL or "flux",
        "prompt": prompt,
        "size": size,
    }
    timeout = httpx.Timeout(connect=8.0, read=120.0, write=8.0, pool=8.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        delay = 1.0
        last_exc: Optional[Exception] = None
        for attempt in range(1, 6):
            try:
                r = await client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                j = r.json()
                parsed = _PollinationsV1Response.model_validate(j)
                if not parsed.data:
                    raise RuntimeError("pollinations_v1_empty_data")
                raw = b64decode(parsed.data[0].b64_json, validate=True)
                out_dir.mkdir(parents=True, exist_ok=True)
                target = out_dir / f"img_{uuid.uuid4().hex}.jpg"
                target.write_bytes(raw)
                if target.stat().st_size < 1024:
                    raise RuntimeError("pollinations_v1_too_small")
                print(f"[POLLINATIONS v1] saved {target} ({target.stat().st_size} bytes)")
                return target
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError, httpx.HTTPStatusError, ValidationError) as e:
                print(f"[POLLINATIONS v1] attempt {attempt} failed: {e}")
                last_exc = e
                if attempt < 5:
                    await asyncio.sleep(delay)
                    delay *= 1.7
                continue
        raise RuntimeError(f"pollinations_v1_all_attempts_failed: {last_exc}")

# GET fallback with extra params (key, enhance, safe, nologo)
import urllib.parse
def _get_params_for_pollinations() -> Dict[str, str]:
    params: Dict[str, str] = {}
    w = POLLINATIONS_WIDTH if POLLINATIONS_WIDTH > 0 else 1024
    h = POLLINATIONS_HEIGHT if POLLINATIONS_HEIGHT > 0 else 1024
    params["width"] = str(w)
    params["height"] = str(h)
    if POLLINATIONS_MODEL:
        params["model"] = POLLINATIONS_MODEL
    if POLLINATIONS_SEED_INT is not None:
        params["seed"] = str(POLLINATIONS_SEED_INT)
    if _env_bool01("POLLINATIONS_NOLOGO", 0):
        params["nologo"] = "true"
    if _env_bool01("POLLINATIONS_ENHANCE", 0):
        params["enhance"] = "true"
    safe = _env_str("POLLINATIONS_SAFE", "")
    if safe:
        params["safe"] = safe
    if POLLINATIONS_SECRET:
        params["key"] = POLLINATIONS_SECRET
    return params

async def fetch_pollinations_get_image(prompt: str, out_dir: Path) -> Path:
    if IMAGE_BACKEND != "pollinations":
        raise RuntimeError("IMAGE_BACKEND!=pollinations")
    if not ALLOW_CLOUD_IMAGE_BACKEND:
        raise RuntimeError("Cloud image backend not allowed (ALLOW_CLOUD_IMAGE_BACKEND=1)")
    base_url = f"{POLLINATIONS_API_BASE.rstrip('/')}/image"
    encoded_prompt = urllib.parse.quote(prompt, safe="")
    url = f"{base_url}/{encoded_prompt}"
    params = _get_params_for_pollinations()
    timeout = httpx.Timeout(connect=8.0, read=120.0, write=8.0, pool=8.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        delay = 1.0
        last_exc: Optional[Exception] = None
        for attempt in range(1, 5):
            try:
                r = await client.get(url, params=params)
                r.raise_for_status()
                content = r.content
                if not content or len(content) < 1024:
                    raise RuntimeError("pollinations_get_too_small")
                out_dir.mkdir(parents=True, exist_ok=True)
                target = out_dir / f"img_{uuid.uuid4().hex}.jpg"
                target.write_bytes(content)
                print(f"[POLLINATIONS GET] saved {target} ({target.stat().st_size} bytes)")
                return target
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError, httpx.HTTPStatusError) as e:
                print(f"[POLLINATIONS GET] attempt {attempt} failed: {e}")
                last_exc = e
                if attempt < 4:
                    await asyncio.sleep(delay)
                    delay *= 1.7
                continue
        raise RuntimeError(f"pollinations_get_all_attempts_failed: {last_exc}")

async def fetch_image(prompt: str, out_dir: Path) -> Path:
    # Fassade mit Fallback: v1 → GET
    if IMAGE_BACKEND != "pollinations":
        raise RuntimeError("Unsupported IMAGE_BACKEND")
    if not ALLOW_CLOUD_IMAGE_BACKEND:
        raise RuntimeError("Cloud image backend not allowed (ALLOW_CLOUD_IMAGE_BACKEND=1)")
    if POLLINATIONS_USE_V1:
        try:
            return await fetch_pollinations_v1_image(prompt, out_dir)
        except Exception as e:
            print(f"[POLLINATIONS] v1 failed, trying GET fallback: {e}")
            return await fetch_pollinations_get_image(prompt, out_dir)
    return await fetch_pollinations_get_image(prompt, out_dir)

# ---- State & SSE ----
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

STATE = PipelineState()
STOP_DEBOUNCE_SEC = float(os.getenv("APP_STOP_DEBOUNCE_SEC", "2.0") or "2.0")

def sse_format(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"

async def broadcast(event: str, data: str) -> None:
    if STATE.shutting_down:
        return
    for q in list(STATE.listeners):
        with contextlib.suppress(Exception):
            await q.put(sse_format(event, data))

_context_buffer: deque[str] = deque(maxlen=CONTEXT_MAX_SEGMENTS)

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
        f"backend={IMAGE_BACKEND} allow_cloud={ALLOW_CLOUD_IMAGE_BACKEND} api_base={POLLINATIONS_API_BASE} key_present={bool(POLLINATIONS_SECRET)}",
    )

# ---- Audio main loop ----
async def audio_transcription_loop() -> None:
    try:
        dev_index = pick_input_device(AUDIO_DEVICE_PREF)
        dev = sd.query_devices()[dev_index]
        dev_name = dev.get("name", f"idx:{dev_index}")
    except Exception as e:
        await broadcast("status", f"audio_device_pick_failed: {e}")
        print(f"[AUDIO] device pick failed: {e}")
        return
    try:
        sd.default.device = (dev_index, None)
        sd.default.samplerate = SAMPLE_RATE
        sd.default.channels = 1
    except Exception as e:
        print(f"[AUDIO] could not set sd.default.*: {e}")

    frame_samples = int(SAMPLE_RATE * FRAME_MS / 1000)
    audio_frames: deque[np.ndarray] = deque()
    got_cb_frames = 0
    overflow_count = 0
    got_first_frame_peak_logged = False

    def sd_callback(indata, frames, timeinfo, status):
        nonlocal got_cb_frames, overflow_count, got_first_frame_peak_logged
        try:
            if status:
                if "overflow" in str(status).lower():
                    overflow_count += 1
            mono = np.asarray(indata[:, 0], dtype=np.float32).copy()
            if not got_first_frame_peak_logged and mono.size > 0:
                p = float(np.max(np.abs(mono)))
                r = float(np.sqrt(np.mean(mono * mono)))
                print(f"[AUDIO] first_frame peak={p:.4f} rms={r:.4f}")
                got_first_frame_peak_logged = True
            audio_frames.append(mono)
            got_cb_frames += 1
        except Exception as e:
            print(f"[AUDIO] callback exception: {e}")

    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=(dev_index, None),
            blocksize=frame_samples,
            latency=APP_STREAM_LATENCY_SEC,
            callback=sd_callback,
        )
        stream.start()
        t0 = time.time()
        while time.time() - t0 < 1.5 and got_cb_frames == 0 and STATE.running:
            await asyncio.sleep(0.05)
        if got_cb_frames == 0:
            with contextlib.suppress(Exception):
                stream.stop()
                stream.close()
            await broadcast("status", "audio_callback_no_frames")
            print("[AUDIO] ERROR: callback produced no frames; aborting")
            return
        print(
            f"[AUDIO] open OK: device_index={dev_index}, name='{dev_name}', "
            f"samplerate={SAMPLE_RATE}, blocksize={frame_samples}, latency={APP_STREAM_LATENCY_SEC}s"
        )
        await broadcast("status", "audio_open_ok")
    except Exception as e:
        await broadcast("status", f"audio_stream_failed: {e}")
        print(f"[AUDIO] stream open failed: {e}")
        return

    STATE.actual_sr = SAMPLE_RATE
    STATE.device_used_index = dev_index
    STATE.device_used_name = dev_name

    last_snapshot = time.time()
    last_tick = time.time()
    total_frames = 0
    current_segment_frames: List[np.ndarray] = []
    silence_ms = 0
    endpoint_silence_ms = MAX_SILENCE_MS
    max_segment_frames = int(((float(MAX_SEGMENT_SEC) * 1000) / FRAME_MS))

    await broadcast("status", f"recording_start sr={SAMPLE_RATE}")
    first_snapshot_deadline = time.time() + FIRST_SNAPSHOT_DEADLINE_SEC

    try:
        while STATE.running:
            await asyncio.sleep(FRAME_MS / 1000.0)
            if not STATE.running:
                break
            now = time.time()
            if now - last_tick >= 1.0:
                last_tick = now
                with contextlib.suppress(Exception):
                    await broadcast(
                        "status",
                        f"tick frames={total_frames} seg_frames={len(current_segment_frames)} sr={SAMPLE_RATE} overflows={overflow_count}",
                    )

            if not audio_frames:
                continue
            frame = audio_frames.popleft()
            total_frames += 1
            target = frame_samples
            if len(frame) != target:
                if len(frame) < target:
                    import numpy as _np
                    frame = _np.pad(frame, (0, target - len(frame)))
                else:
                    frame = frame[:target]
            frame = np.clip(frame, -1.0, 1.0)

            if DISABLE_VAD:
                is_speech = True
            else:
                is_speech = rms_vad(frame, rms_threshold=RMS_VAD_THRESHOLD)

            current_segment_frames.append(frame)
            if len(current_segment_frames) > max_segment_frames:
                is_segment_end = True
            else:
                silence_ms = 0 if is_speech else (silence_ms + FRAME_MS)
                is_segment_end = silence_ms >= endpoint_silence_ms

            buf_sec = (len(current_segment_frames) * FRAME_MS) / 1000.0
            do_snapshot_time = (now - last_snapshot) >= SNAPSHOT_SEC or (now >= first_snapshot_deadline)
            do_snapshot = (do_snapshot_time or is_segment_end) and (buf_sec >= MIN_BUF_SEC)

            if do_snapshot:
                last_snapshot = now
                if now >= first_snapshot_deadline:
                    first_snapshot_deadline = float("inf")
                seg = np.frombuffer(b"".join([f.tobytes() for f in current_segment_frames]), dtype=np.float32)
                seg_dur = seg.size / float(SAMPLE_RATE) if SAMPLE_RATE > 0 else 0.0
                print(f"[SNAPSHOT] trigger buf_sec={buf_sec:.2f} seg_dur={seg_dur:.2f}s frames={len(current_segment_frames)}")
                if seg.size > 0:
                    peak = float(np.max(np.abs(seg)))
                    if peak >= WHISPER_MIN_PEAK:
                        await broadcast("status", f"whisper_start buf_sec={buf_sec:.2f}")
                        txt = transcribe_chunk_with_whisper(seg, SAMPLE_RATE)
                        if txt:
                            await broadcast("transcript", txt)
                            meaningful_ok = (not FORCE_MEANINGFUL_CHECK) or is_meaningful_text(
                                txt, TEXT_MIN_CHARS, TEXT_MIN_WORDS
                            )
                            now_ts = time.time()
                            if now_ts < STATE.ollama_ready_at:
                                STATE.last_pending_text = txt
                            else:
                                can_run_llm = (now_ts - STATE.last_llm_run_ts >= LLM_INTERVAL_SEC)
                                if not meaningful_ok:
                                    await broadcast("status", "llm_blocked_meaningful")
                                elif not can_run_llm:
                                    await broadcast("status", "llm_wait_interval")
                                elif OLLAMA_DISABLED:
                                    await broadcast("status", "ollama_disabled")
                                else:
                                    await broadcast("status", "llm_start")
                                    STATE.last_llm_run_ts = now_ts
                                    t = asyncio.create_task(run_llm_and_image(update_context_buffer(txt)))
                                    STATE.bg_tasks.add(t)
                                    t.add_done_callback(lambda fut: STATE.bg_tasks.discard(fut))
                        else:
                            await broadcast("status", "whisper_empty(no_text)")
                    else:
                        print(f"[SNAPSHOT] low_peak({peak:.3f}) – ignored")
                        await broadcast("status", f"low_peak({peak:.3f})")
                if is_segment_end:
                    current_segment_frames.clear()
                    silence_ms = 0
                    print("[SNAPSHOT] segment_end → buffer cleared")
    except asyncio.CancelledError:
        pass
    finally:
        with contextlib.suppress(Exception):
            stream.stop()
            stream.close()
        await broadcast("status", "audio_stream_closed")
        await broadcast("status", "recording_stop")
        print("[AUDIO] stream closed")

# ---- LLM + Pollinations image ----
async def run_llm_and_image(text: str) -> None:
    if await _ollama_available() is False:
        await broadcast("status", "ollama_unavailable")
        STATE.last_llm_error = "ollama_unavailable"
        return
    async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_normal()) as client:
        try:
            img_prompt = await ollama_generate_prompt(client, text)
            if not img_prompt:
                STATE.last_llm_error = "llm_empty_response"
                await broadcast("status", "llm_empty_response")
                return
            STATE.last_prompt = img_prompt
            await broadcast("llm_prompt", img_prompt)
            await broadcast("status", "llm_ok")

            if IMAGE_BACKEND == "pollinations":
                if not ALLOW_CLOUD_IMAGE_BACKEND:
                    await broadcast("status", "image_backend_blocked")
                    return
                try:
                    # unified fetch with v1→GET fallback
                    path = await fetch_image(img_prompt, OUTPUT_DIR)
                    rel = path.name
                    await broadcast("image", rel)
                except Exception as e:
                    await broadcast("status", f"pollinations_error:{e}")
            else:
                await broadcast("status", f"image_backend_unsupported:{IMAGE_BACKEND}")

        except Exception as e:
            STATE.last_llm_error = f"pipeline_error:{e}"
            await broadcast("status", f"pipeline_error:{e}")

# ---- FastAPI & Lifespan ----
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[ENV] loaded from: {ENV_PATH or '(env vars only)'}")
    _log_effective_config()
    init_whisper_model()
    STATE.ollama_ready_at = time.time() + (WARMUP_GRACE_SEC if WARMUP_ENABLE else 0.0)
    warmup_task: Optional[asyncio.Task] = None
    if WARMUP_ENABLE:
        async def _silent_ollama_warmup():
            try:
                payload = {"model": OLLAMA_MODEL, "prompt": WARMUP_PROMPT, "stream": False, "options": {"temperature": 0.1, "num_predict": 32}}
                async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_short()) as c:
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
app.mount("/static", StaticFiles(directory=str(OUTPUT_DIR), html=False), name="static")

# ---- UI mit Vollbild-Button und sanftem Übergang ----
INDEX_HTML = """<!doctype html>
<html lang="de"><head><meta charset="utf-8"><title>Vorlesen → Bilder (Serverseitig)</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0b1020;--panel:#141a2e;--line:rgba(255,255,255,.06);--fg:#e8ecf1;--accent:#1f6feb}
html,body{height:100%}
body{font-family:system-ui,sans-serif;margin:0;padding:0;background:var(--bg);color:var(--fg);display:flex;flex-direction:column}
header{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap;padding:1rem}
button{background:var(--accent);color:#fff;border:0;padding:.6rem 1rem;border-radius:8px;cursor:pointer}
button.stop{background:#c53b3b}
#status{opacity:.9;font-size:.9rem}
main{display:flex;flex-direction:column;gap:.75rem;padding:0 1rem 1rem 1rem;flex:1;min-height:0}
#live{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.panel{background:var(--panel);border-radius:10px;overflow:hidden;box-shadow:0 0 0 1px var(--line) inset}
.panel-title{font-size:.9rem;font-weight:600;padding:.5rem .75rem;color:#c7d1df;border-bottom:1px solid var(--line)}
.panel-body{padding:.6rem .75rem;min-height:52px;white-space:pre-wrap;word-break:break-word}
.mono{font-family:ui-monospace,Menlo,Consolas,"Liberation Mono",monospace}
#viewer{background:var(--panel);border-radius:10px;box-shadow:0 0 0 1px var(--line) inset;display:flex;flex-direction:column;min-height:0;flex:1;position:relative}
#stage{position:relative;flex:1;min-height:0;display:flex;align-items:center;justify-content:center;background:#0e1426;overflow:hidden}
#stage img.layer{position:absolute;inset:0;margin:auto;max-width:100%;max-height:100%;width:auto;height:auto;object-fit:contain;display:block;opacity:0;transition:opacity 450ms ease}
#stage img.layer.show{opacity:1}
#stage .placeholder{color:#8fa0b8;opacity:.9;font-size:.95rem}
#stage .caption{position:absolute;left:8px;bottom:6px;background:rgba(0,0,0,.45);color:#d7e1ef;padding:.25rem .5rem;border-radius:6px;font-size:.8rem;pointer-events:none}
.controls{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap}
input[type="text"]{background:#0e1426;color:#e8ecf1;border:1px solid #283355;border-radius:8px;padding:.45rem .6rem;min-width:260px}
.small{font-size:.85rem;opacity:.85}
.warn{color:#f9cdcd}
/* Vollbild-Button rechts unten */
#fsbtn{position:absolute;right:10px;bottom:10px;background:rgba(31,111,235,.9);border-radius:8px;padding:.5rem .75rem;z-index:5}
</style>
</head><body>
<header class="controls">
  <button id="start">Start</button>
  <button id="stop" class="stop">Stop</button>
  <button id="shutdown" class="stop">Server beenden</button>
  <button id="llmtest">LLM Test</button>
  <input id="manual" type="text" placeholder="Manueller Text → Prompt" />
  <button id="send">Senden</button>
  <div id="status">Ready.</div>
</header>
<main>
  <section id="live">
    <div class="panel"><div class="panel-title">Transcript</div><div id="transcript" class="panel-body mono"></div></div>
    <div class="panel"><div class="panel-title warn">Prompt (serverseitig)</div><div id="prompt" class="panel-body"></div></div>
  </section>
  <div id="viewer">
    <div id="stage">
      <div class="placeholder">Noch kein Bild. Sprich etwas oder sende einen Prompt.</div>
      <img class="layer" id="imgA" alt="">
      <img class="layer" id="imgB" alt="">
      <div class="caption" id="cap"></div>
      <button id="fsbtn" title="Vollbild umschalten">Vollbild</button>
    </div>
  </div>
</main>
<script>
const statusEl=document.getElementById('status');
const promptEl=document.getElementById('prompt');
const transcriptEl=document.getElementById('transcript');
const stage=document.getElementById('stage');
const imgA=document.getElementById('imgA');
const imgB=document.getElementById('imgB');
const cap=document.getElementById('cap');
const fsbtn=document.getElementById('fsbtn');

let evtSrc=null, busy=false, toggle=false;

function setStatus(m){statusEl.textContent=m}
function setPrompt(p){promptEl.textContent=p||''}
function setTranscript(t){transcriptEl.textContent=t||''}

function crossfadeTo(rel){
  const ph = stage.querySelector('.placeholder');
  if(ph) ph.remove();
  const next = toggle ? imgA : imgB;
  const current = toggle ? imgB : imgA;
  toggle = !toggle;

  // Cache-buster und Pfad
  const url = '/static/'+rel+'?t='+Date.now();
  next.classList.remove('show');
  next.onload = ()=>{
    // erst wenn geladen, dann einblenden
    if(current) current.classList.remove('show');
    next.classList.add('show');
    cap.textContent = new Date().toLocaleTimeString();
  };
  next.src = url;
}

async function start(){
  if(busy) return; busy=true;
  try{
    if(evtSrc){ try{evtSrc.close();}catch(_){} evtSrc=null; }
    evtSrc=new EventSource('/events');
    evtSrc.addEventListener('status', e=>setStatus(e.data));
    evtSrc.addEventListener('transcript', e=>setTranscript(e.data));
    evtSrc.addEventListener('llm_prompt', e=>setPrompt(e.data));
    evtSrc.addEventListener('image', e=>crossfadeTo(e.data));
    await new Promise(res=>{
      const check=()=>{ if(evtSrc && evtSrc.readyState===1) res(); else setTimeout(check,50); }; check();
    });
    await fetch('/start',{method:'POST'});
  } finally { busy=false; }
}

async function stop(){
  if(busy) return; busy=true;
  try{
    await fetch('/stop',{method:'POST'});
    if(evtSrc){ try{evtSrc.close();}catch(_){} evtSrc=null; }
    setStatus('stopped');
  } finally { busy=false; }
}

async function shutdown(){
  if(busy) return; busy=true;
  try{
    await fetch('/shutdown',{method:'POST'});
    if(evtSrc){ try{evtSrc.close();}catch(_){} evtSrc=null; }
    setStatus('server shutting down');
  } finally { busy=false; }
}

async function llmTest(){
  const cfg = await (await fetch('/config')).json();
  const body={model: cfg.ollama.model, prompt:"<<SYS>>Schneller Test-Prompt<</SYS>>\\n\\nINPUT_JSON:\\n{\\"user_text\\": \\"Ein roter Ballon über einer Stadt im Sonnenuntergang\\", \\"constraints\\":{\\"no_meta\\":true,\\"max_sentences\\":2,\\"avoid_sensitive\\":true}}\\n\\nOUTPUT:\\n", stream:false, options:{}};
  const r=await fetch('/api/ollama/generate',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});
  const j=await r.json();
  const out = j.response || j.error || JSON.stringify(j);
  setPrompt(out);
}

async function sendManual(){
  const t=document.getElementById('manual').value.trim();
  if(!t) return;
  const r=await fetch('/api/plan',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({text:t,tags:[]})});
  const j=await r.json();
  if(j.prompt){ setPrompt(j.prompt); }
  else if(j.error){ setStatus('Error: '+j.error); }
}

// Vollbild-Toggle
fsbtn.addEventListener('click', async ()=>{
  const el = document.documentElement; // ganze Seite in Vollbild
  try{
    if(!document.fullscreenElement){
      await el.requestFullscreen();
      fsbtn.textContent = 'Vollbild beenden';
    }else{
      await document.exitFullscreen();
      fsbtn.textContent = 'Vollbild';
    }
  }catch(e){
    console.warn('Fullscreen failed', e);
  }
});

document.getElementById('start').addEventListener('click', start);
document.getElementById('stop').addEventListener('click', stop);
document.getElementById('shutdown').addEventListener('click', shutdown);
document.getElementById('llmtest').addEventListener('click', llmTest);
document.getElementById('send').addEventListener('click', sendManual);
</script>
</body></html>"""

# ---- Routes ----
from fastapi import APIRouter
router = APIRouter()

@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)

@app.get("/favicon.ico")
def favicon() -> Response:
    data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0cIDATx\x9cc``\x00\x00\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
    return Response(content=data, media_type="image/png")

@app.get("/events")
async def events(request: Request):
    async def gen():
        q: asyncio.Queue[str] = asyncio.Queue()
        STATE.listeners.append(q)
        try:
            await q.put(sse_format("status", "connected"))
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    yield sse_format("status", "hb").encode("utf-8")
                    continue
                if not msg:
                    break
                yield msg.encode("utf-8")
        except (asyncio.CancelledError, Exception):
            pass
        finally:
            with contextlib.suppress(ValueError):
                if q in STATE.listeners:
                    STATE.listeners.remove(q)
    return StreamingResponse(gen(), media_type="text/event-stream")

async def _close_sse_listeners():
    for q in list(STATE.listeners):
        with contextlib.suppress(Exception):
            await q.put("")
    STATE.listeners.clear()

@app.get("/config")
async def get_config():
    wpath = WHISPER_MODEL_PATH
    masked = (wpath[:3] + "..." + wpath[-10:]) if wpath and len(wpath) > 16 else wpath
    return {
        "env_file": ENV_PATH or "(env vars only)",
        "audio": {"device_pref": AUDIO_DEVICE_PREF, "sample_rate": SAMPLE_RATE, "frame_ms": FRAME_MS, "stream_latency_sec": APP_STREAM_LATENCY_SEC},
        "vad": {"disable_vad": DISABLE_VAD, "rms_threshold": RMS_VAD_THRESHOLD},
        "snapshot": {"snapshot_sec": SNAPSHOT_SEC, "min_buf_sec": MIN_BUF_SEC, "max_silence_ms": MAX_SILENCE_MS, "max_segment_sec": MAX_SEGMENT_SEC, "first_snapshot_deadline_sec": FIRST_SNAPSHOT_DEADLINE_SEC},
        "whisper": {"model_path": masked, "language": WHISPER_LANGUAGE, "threads": WHISPER_THREADS, "temperature": WHISPER_TEMPERATURE, "min_sec": WHISPER_MIN_SEC, "min_peak": WHISPER_MIN_PEAK},
        "text": {"min_chars": TEXT_MIN_CHARS, "min_words": TEXT_MIN_WORDS, "force_meaningful": FORCE_MEANINGFUL_CHECK},
        "context": {"max_segments": CONTEXT_MAX_SEGMENTS, "max_chars": CONTEXT_MAX_CHARS},
        "ollama": {"host": OLLAMA_HOST, "port": OLLAMA_PORT, "model": OLLAMA_MODEL, "temperature": OLLAMA_TEMPERATURE, "timeout_sec": OLLAMA_TIMEOUT_SEC, "interval_sec": LLM_INTERVAL_SEC, "disabled": OLLAMA_DISABLED},
        "image": {"backend": IMAGE_BACKEND, "allow_cloud": ALLOW_CLOUD_IMAGE_BACKEND, "api_base": POLLINATIONS_API_BASE, "width": POLLINATIONS_WIDTH, "height": POLLINATIONS_HEIGHT, "seed": POLLINATIONS_SEED_INT, "server_key_present": bool(POLLINATIONS_SECRET), "use_v1": POLLINATIONS_USE_V1, "size": POLLINATIONS_SIZE or _size_from_wh(POLLINATIONS_WIDTH, POLLINATIONS_HEIGHT)},
        "output_dir": str(OUTPUT_DIR),
    }

@app.get("/health", response_model=HealthReport)
async def health() -> HealthReport:
    ollama_ok = await _ollama_available()
    return HealthReport(
        ollama_ok=ollama_ok,
        image_backend=IMAGE_BACKEND,
        allow_cloud=ALLOW_CLOUD_IMAGE_BACKEND,
        output_dir=str(OUTPUT_DIR),
        output_dir_exists=OUTPUT_DIR.exists(),
        last_prompt=STATE.last_prompt,
        last_llm_error=STATE.last_llm_error,
        pollinations_key_present=bool(POLLINATIONS_SECRET),
    )

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
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/start", response_class=PlainTextResponse)
async def start_pipeline():
    print("[HTTP] /start called")
    if STATE.running:
        print("[HTTP] /start ignored (already running)")
        return PlainTextResponse("already running", status_code=200)
    STATE.shutting_down = False
    STATE.running = True
    STATE.start_ts = time.time()
    STATE.task = asyncio.create_task(audio_transcription_loop())
    await broadcast("status", "server_start_recording")
    return PlainTextResponse("started")

@app.post("/stop", response_class=PlainTextResponse)
async def stop_pipeline():
    print("[HTTP] /stop called")
    now = time.time()
    if STATE.running and (now - STATE.start_ts) < STOP_DEBOUNCE_SEC:
        msg = f"stop_ignored_debounce({now - STATE.start_ts:.2f}s)"
        print(f"[HTTP] /stop ignored: {msg}")
        await broadcast("status", msg)
        return PlainTextResponse("stop_ignored_debounce", status_code=200)

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
    await broadcast("status", "server_stopped")
    await _close_sse_listeners()
    print("[HTTP] /stop completed")
    return PlainTextResponse("stopped")

@app.post("/shutdown", response_class=PlainTextResponse)
async def shutdown_server():
    print("[HTTP] /shutdown called")
    STATE.shutting_down = True
    await stop_pipeline()
    asyncio.create_task(_exit_after_delay())
    return PlainTextResponse("shutting down")

async def _exit_after_delay():
    await asyncio.sleep(0.2)
    os._exit(0)

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
            return JSONResponse({"error": str(e)}, status_code=500)

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
                STATE.last_prompt = out
                await broadcast("llm_prompt", out)
                if IMAGE_BACKEND == "pollinations" and ALLOW_CLOUD_IMAGE_BACKEND:
                    try:
                        path = await fetch_image(out, OUTPUT_DIR)
                        rel = path.name
                        await broadcast("image", rel)
                    except Exception as e:
                        await broadcast("status", f"pollinations_error:{e}")
            return {"prompt": out}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/pollinations/test")
async def api_pollinations_test():
    if IMAGE_BACKEND != "pollinations" or not ALLOW_CLOUD_IMAGE_BACKEND:
        return JSONResponse({"error": "pollinations_disabled"}, status_code=400)
    try:
        path = await fetch_image("Ein bunter Sonnenuntergang", OUTPUT_DIR)
        rel = path.name
        return {"rel": rel}
    except Exception as e:
        return JSONResponse({"error": f"{e}"}, status_code=502)

# ---- App entry ----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=False)
