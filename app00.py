#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# NOTE: This is your provided app.py with a focused fix:
# - Remove httpx.AsyncClient monkeypatch with private kwargs
# - Introduce a guarded client wrapper using composition
# - Replace call sites to use the factory instead of global monkeypatch
# - Add a small helper to filter httpx kwargs
# All other logic remains the same.

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import ipaddress
import json
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import httpx
import numpy as np
import sounddevice as sd
from fastapi import FastAPI, Request, UploadFile, File, Query
from fastapi.responses import (
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Image backend factory and interface
from image_backend import build_image_backend, ImageBackend
try:
    from image_backend import LocalComfyBackend  # type: ignore
except Exception:
    LocalComfyBackend = None  # type: ignore

# Style engine
from style_engine import (
    StyleConfig,
    build_prompt as build_style_prompt,
    ReferenceStore,
    prepare_backend_style,
)

try:
    from style_engine import apply_ip_adapter_to_workflow, reference_store  # type: ignore
except Exception:
    apply_ip_adapter_to_workflow = None  # type: ignore
    reference_store = None  # type: ignore

try:
    from style_engine import resolve_reference_urls_for_pollinations  # type: ignore
except Exception:
    resolve_reference_urls_for_pollinations = None  # type: ignore

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

APP_POLLINATIONS_REF_MODE = _env_str("APP_POLLINATIONS_REF_MODE", "auto").lower()

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

def rel_for_ui_path(p: Path) -> str:
    return Path(p).name

def rel_for_ui(p: Path) -> str:
    return rel_for_ui_path(p)

def ensure_in_output_dir(p: Path) -> Path:
    """Ensure generated image is in OUTPUT_DIR; move/copy if needed. German: Sicher in Static-Verzeichnis bringen."""
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

APP_IMAGE_WIDTH = _env_int("APP_IMAGE_WIDTH", 512)
APP_IMAGE_HEIGHT = _env_int("APP_IMAGE_HEIGHT", 512)

STYLE_CFG_DIR = Path(_env_str("APP_STYLE_CFG_DIR", "./outputs/config")).resolve()
STYLE_CFG_DIR.mkdir(parents=True, exist_ok=True)
STYLE_CFG_PATH = STYLE_CFG_DIR / "style.json"
STYLE_REFS_DIR = Path(_env_str("APP_STYLE_REF_DIR", "./outputs/style_refs")).resolve()
STYLE_REFS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Workflows (ComfyUI) ----------
WORKFLOWS_DIR = Path(os.getenv("WORKFLOWS_DIR", "workflows")).resolve()
WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Ollama ----------
OLLAMA_HOST = _env_str("APP_OLLAMA_HOST", "127.0.0.1")
OLLAMA_PORT = _env_int("APP_OLLAMA_PORT", 11434)
OLLAMA_MODEL = _env_str("APP_OLLAMA_MODEL", "gemma3:1b")
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
    """Enforce Ollama host to be localhost only."""
    if host != "127.0.0.1":
        raise AssertionError(f"Only localhost allowed for Ollama, got {host}")

assert_local(OLLAMA_HOST)

def _assert_image_backend_host() -> None:
    """German: Remote Backends nur wenn explizit erlaubt; sonst comfy_host muss localhost sein."""
    allow_remote = _env_bool01("APP_ALLOW_REMOTE_BACKENDS", 0)
    comfy_host = _env_str("APP_COMFY_HOST", "127.0.0.1")
    if not allow_remote:
        if comfy_host not in {"127.0.0.1", "localhost"}:
            raise AssertionError(
                f"Remote image backends are disabled. Set APP_ALLOW_REMOTE_BACKENDS=1 to allow non-localhost hosts "
                f"(current APP_COMFY_HOST={comfy_host})."
            )

_assert_image_backend_host()

COMFY_REMOTE_WHITELIST = [s.strip() for s in os.getenv("APP_COMFY_REMOTE_WHITELIST", "").split(",") if s.strip()]

WARMUP_ENABLE = _env_bool01("APP_OLLAMA_WARMUP_ENABLE", 1)
WARMUP_PROMPT = _env_str("APP_OLLAMA_WARMUP_PROMPT", "Sag Hallo auf Deutsch.")
WARMUP_TIMEOUT_SEC = _env_float("APP_OLLAMA_TIMEOUT_SEC", 45.0)
WARMUP_MAX_RETRIES = _env_int("APP_OLLAMA_MAX_RETRIES", 3)
WARMUP_RETRY_DELAY = _env_float("APP_OLLAMA_RETRY_DELAY", 1.2)
WARMUP_GRACE_SEC = _env_float("APP_OLLAMA_GRACE_SEC", 10.0)

# ---------- Reference URL signing ----------
APP_REF_HOST = _env_str("APP_REF_HOST", "")
APP_REF_TTL_SEC = _env_int("APP_REF_TTL_SEC", 180)
APP_REF_SECRET = _env_str("APP_REF_SECRET", "")

def _is_local_or_lan_host(url: str) -> bool:
    """Allow only localhost or RFC1918/ULA hosts for signed references."""
    try:
        u = url.strip().lower()
        if not (u.startswith("http://") or u.startswith("https://")):
            return False
        host = u.split("://", 1)[1].split("/", 1)[0]
        host = host.split("@")[-1].split("]")[-1].split(":")[0]
        if host in {"127.0.0.1", "localhost"}:
            return True
        parts = host.split(".")
        if len(parts) == 4 and all(p.isdigit() for p in parts):
            a, b, c, d = [int(p) for p in parts]
            if a == 10:
                return True
            if a == 192 and b == 168:
                return True
            if a == 172 and 16 <= b <= 31:
                return True
        return False
    except Exception:
        return False

def _safe_basename(name: str) -> str:
    if "/" in name or "\\" in name:
        raise ValueError("invalid path separator")
    if not re.fullmatch(r"[A-Za-z0-9._\-]+", name or ""):
        raise ValueError("illegal characters")
    return name

def _hmac_sign(msg: str, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), msg.encode("utf-8"), hashlib.sha256).hexdigest()

def _hmac_verify(msg: str, sig: str, secret: str) -> bool:
    try:
        expected = _hmac_sign(msg, secret)
        return hmac.compare_digest(expected, sig)
    except Exception:
        return False

def build_signed_url(filename: str, now_ts: Optional[int] = None) -> str:
    """Build a signed, time-limited reference URL for a style ref image."""
    base = _safe_basename(filename)
    host = APP_REF_HOST or ""
    if not host:
        host = f"http://{os.getenv('APP_BIND_HOST','127.0.0.1')}:{int(os.getenv('APP_BIND_PORT','8080') or '8080')}"
    if not _is_local_or_lan_host(host):
        raise AssertionError("APP_REF_HOST must be localhost or LAN")
    if not APP_REF_SECRET:
        raise AssertionError("APP_REF_SECRET must be set for URL reference mode")
    ts = int(now_ts or time.time())
    exp = ts + int(APP_REF_TTL_SEC)
    payload = f"{base}:{exp}"
    sig = _hmac_sign(payload, APP_REF_SECRET)
    return f"{host.rstrip('/')}/ref/{base}?ts={exp}&sig={sig}"

def _verify_and_open_ref(basename: str, ts: int, sig: str) -> Tuple[bytes, str]:
    if not APP_REF_SECRET:
        raise PermissionError("ref_disabled")
    now = int(time.time())
    if ts < now:
        raise PermissionError("expired")
    base = _safe_basename(basename)
    msg = f"{base}:{ts}"
    if not _hmac_verify(msg, sig or "", APP_REF_SECRET):
        raise PermissionError("bad_sig")
    src = (STYLE_REFS_DIR / base).resolve()
    if src.parent != STYLE_REFS_DIR or not src.exists() or not src.is_file():
        raise FileNotFoundError("not_found")
    data = src.read_bytes()
    m = "image/png"
    low = base.lower()
    if low.endswith(".jpg") or low.endswith(".jpeg"):
        m = "image/jpeg"
    elif low.endswith(".webp"):
        m = "image/webp"
    elif low.endswith(".bmp"):
        m = "image/bmp"
    return data, m

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
    backend: Literal["comfyui", "comfyui_remote", "pollinations"]
    reset: bool = False

class ImageRequest(BaseModel):
    prompt: str = Field(min_length=0, max_length=2000)
    width: int | None = Field(default=None)
    height: int | None = Field(default=None)
    negative_prompt: Optional[str] = None

    @field_validator("width", "height")
    @classmethod
    def _clamp_size(cls, v: int | None) -> int | None:
        _MIN_SIZE = 64
        _MAX_SIZE = 2048
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

class ImageResponse(BaseModel):
    filename: str
    relpath: str
    rel: Optional[str] = None
    width: int | None = None
    height: int | None = None

class ImageSizeSettings(BaseModel):
    width: int = Field(ge=64, le=2048)
    height: int = Field(ge=64, le=2048)

class NegativePromptSettings(BaseModel):
    negative_prompt: str = Field(default="", max_length=4000)

class StyleSettingsPayload(BaseModel):
    style_preset: str = Field(default="photo")
    style_details: str = Field(default="")
    negative_base: str = Field(default="")
    color_scheme: str = Field(default="")
    use_reference: bool = Field(default=False)
    reference_id: Optional[str] = None
    reference_strength: float = Field(default=0.6, ge=0.0, le=1.0)

class StyleRefOnPayload(BaseModel):
    reference_id: str = Field(..., min_length=1)
    reference_strength: float = Field(default=0.6, ge=0.0, le=1.0)
    reference_cloud: Optional[bool] = None

class StyleRefOnResponse(BaseModel):
    ok: bool
    reference_id: Optional[str] = None
    error: Optional[str] = None

class StyleRefOffResponse(BaseModel):
    ok: bool

class ReferenceUploadResponse(BaseModel):
    ok: bool
    reference_id: Optional[str] = None
    error: Optional[str] = None

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

class ComfyTargetReq(BaseModel):
    target: Literal["local", "remote"]
    host: Optional[str] = None
    port: Optional[int] = None

# ---------- Audio utils ----------
def pick_input_device(prefer: Optional[str] = None) -> int:
    """Pick an audio input device with max_input_channels>0. German: einfache Heuristik."""
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
    """Simple RMS-based VAD; German: für echte Robustheit WebRTC-VAD nutzen."""
    if frame.size == 0:
        return False
    rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float32), dtype=np.float64)))
    return rms >= rms_threshold

def resample_to_16k(samples: np.ndarray, sr: int) -> np.ndarray:
    """Linear interpolation resampling to 16kHz."""
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
    """Initialize whisper.cpp model if path present."""
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
    """Normalize whisper output dicts/segments/strings into final text."""
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
    """Filter filler words and meta noise markers."""
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
    """Run whisper.cpp on a mono float32 chunk; includes peak/min-sec guards."""
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
def _httpx_limits_app() -> httpx.Limits:
    return httpx.Limits(max_keepalive_connections=6, max_connections=12, keepalive_expiry=20.0)

def _timeout_short_http() -> httpx.Timeout:
    return httpx.Timeout(connect=2.5, read=4.0, write=3.0, pool=3.0)

def _timeout_normal() -> httpx.Timeout:
    t = min(max(5.0, OLLAMA_TIMEOUT_SEC), 120.0)
    return httpx.Timeout(connect=5.0, read=t, write=5.0, pool=5.0)

def _debug_enabled() -> bool:
    v = (os.getenv("APP_DEBUG", "0") or "").strip().lower()
    return v in {"1", "true", "yes", "on"}

# ---------- NetGuard (composition-based guarded client) ----------
class NetGuardConfig:
    def __init__(self, comfy_host: str, forbid_prefixes: Optional[List[str]] = None) -> None:
        self.comfy_host = (comfy_host or "").strip().lower()
        self.forbid_prefixes = forbid_prefixes or ["10.", "192.168.", "172.16.", "172.17.", "172.18.", "172.19.", "172.20.", "fd", "fe80"]

def _url_host(u: str) -> str:
    try:
        from urllib.parse import urlparse
        p = urlparse(u)
        return (p.hostname or "").lower()
    except Exception:
        return ""

def _is_forbidden_non_comfy_host(host: str, cfg: NetGuardConfig) -> bool:
    """Block non-Comfy requests to private ranges. German: Ausnahmen nur für Comfy-Host."""
    h = (host or "").strip().lower()
    if not h:
        return False
    if cfg.comfy_host and h == cfg.comfy_host:
        return True
    for pref in cfg.forbid_prefixes:
        if h.startswith(pref):
            return True
    return False

ALLOWED_HTTPX_CLIENT_KWARGS = {
    "headers","params","auth","timeout","verify","proxies","limits",
    "transport","app","cookies","http2","follow_redirects","base_url",
    "event_hooks","trust_env","mounts"
}

def _safe_httpx_kwargs(kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Filter unknown kwargs to avoid TypeError on httpx.AsyncClient construction."""
    if not kwargs:
        return {}
    return {k: v for k, v in kwargs.items() if k in ALLOWED_HTTPX_CLIENT_KWARGS}

class GuardedAsyncClient:
    """
    A thin composition-based wrapper that enforces net-guard rules without altering httpx API.
    """
    def __init__(self, is_comfy_backend: bool, guard_cfg: NetGuardConfig, **kwargs: Any):
        self._is_comfy_backend = bool(is_comfy_backend)
        self._guard_cfg = guard_cfg
        self._client = httpx.AsyncClient(**_safe_httpx_kwargs(kwargs))

    def _guard(self, url: str) -> None:
        if self._is_comfy_backend:
            return
        if not self._guard_cfg:
            return
        host = _url_host(url)
        if _is_forbidden_non_comfy_host(host, self._guard_cfg):
            raise RuntimeError(f"[NETGUARD] Non-Comfy backend attempted to reach forbidden host {host}: {url}")

    async def get(self, url: str, *args, **kwargs):
        self._guard(url)
        return await self._client.get(url, *args, **kwargs)

    async def post(self, url: str, *args, **kwargs):
        self._guard(url)
        return await self._client.post(url, *args, **kwargs)

    async def request(self, method: str, url: str, *args, **kwargs):
        self._guard(url)
        return await self._client.request(method, url, *args, **kwargs)

    async def aclose(self):
        await self._client.aclose()

    # Async context manager support
    async def __aenter__(self):
        await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return await self._client.__aexit__(exc_type, exc, tb)

def make_async_client(is_comfy_backend: bool, *, limits: Optional[httpx.Limits] = None, timeout: Optional[httpx.Timeout] = None) -> GuardedAsyncClient | httpx.AsyncClient:
    """
    Factory: return a guarded client for non-Comfy contexts, plain client otherwise.
    We still use the guard even for Comfy=False (non-Comfy), but allow Comfy host only to Comfy.
    """
    kwargs: Dict[str, Any] = {}
    if limits is not None:
        kwargs["limits"] = limits
    if timeout is not None:
        kwargs["timeout"] = timeout
    # Always return the GuardedAsyncClient; for is_comfy_backend=True guard is a no-op
    return GuardedAsyncClient(is_comfy_backend=is_comfy_backend, guard_cfg=NetGuardConfig(STATE.comfy_host), **kwargs)

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

async def _post_with_retries(client: Any, url: str, body: dict, timeout: float) -> dict:
    """
    Robust POST with exponential backoff for local Ollama under load.
    Note: client can be GuardedAsyncClient or httpx.AsyncClient.
    """
    delay = float(_env_float("APP_OLLAMA_RETRY_BASE_DELAY", 0.8))
    max_retries = int(_env_int("APP_OLLAMA_MAX_RETRIES", 4))
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = await client.post(url, json=body, timeout=timeout)
            if hasattr(resp, "raise_for_status"):
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
        async with make_async_client(is_comfy_backend=False, limits=_httpx_limits_app(), timeout=_timeout_short_http()) as c:
            r = await c.get(_ollama_url("/api/tags"))
            r.raise_for_status()
            return True
    except Exception:
        return False

async def ollama_generate_prompt(client: Any, user_text: str) -> str:
    """Call Ollama /api/generate to optimize T2I prompt."""
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
    style_cfg: StyleConfig = field(default_factory=StyleConfig)
    comfy_target: Literal["local", "remote"] = "local"
    comfy_host: str = _env_str("APP_COMFY_HOST", "127.0.0.1")
    comfy_port: int = _env_int("APP_COMFY_PORT", 8188)
    comfy_whitelist: List[str] = field(default_factory=lambda: COMFY_REMOTE_WHITELIST.copy())

STATE = PipelineState()
STOP_DEBOUNCE_SEC = float(os.getenv("APP_STOP_DEBOUNCE_SEC", "2.0") or "2.0")

# ---------- Style persistence helpers ----------
def _load_style_cfg() -> StyleConfig:
    if STYLE_CFG_PATH.exists():
        try:
            data = json.loads(STYLE_CFG_PATH.read_text(encoding="utf-8"))
            cfg = StyleConfig(**data)
            cfg.persisted_path = STYLE_CFG_PATH
            return cfg
        except Exception as e:
            print(f"[STYLE] failed to load persisted style config: {e}")
    cfg = StyleConfig()
    cfg.persisted_path = STYLE_CFG_PATH
    return cfg

def _save_style_cfg(cfg: StyleConfig) -> None:
    try:
        data = cfg.model_dump()
        data.pop("persisted_path", None)
        STYLE_CFG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[STYLE] failed to persist style config: {e}")

async def safe_stop_audio_stream() -> None:
    """Stop and close audio stream safely and notify listeners once."""
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
    """Put SSE event to all listener queues."""
    if STATE.shutting_down:
        return
    for q in list(STATE.listeners):
        with contextlib.suppress(Exception):
            await q.put(sse_format(event, data))

_context_buffer: deque[str] = deque(maxlen=CONTEXT_MAX_SEGMENTS)

async def _close_sse_listeners(timeout: float = 0.25) -> None:
    """Signal-close all SSE listeners quickly."""
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
    """Maintain a rolling context buffer for LLM prompt building."""
    _context_buffer.append(text)
    ctx = " ".join(_context_buffer)
    if len(ctx) > CONTEXT_MAX_CHARS:
        ctx = ctx[-CONTEXT_MAX_CHARS:]
    return ctx

def _log_effective_config() -> None:
    """Print effective configuration snapshot once on startup."""
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
        "| comfy:",
        f"target={STATE.comfy_target} host={STATE.comfy_host}:{STATE.comfy_port}",
        "| style:",
        f"preset={STATE.style_cfg.style_preset} use_ref={STATE.style_cfg.use_reference}",
        "| sse:",
        f"tick={APP_SSE_TICK_SEC}s",
    )

def _apply_view_mode_for_target(host: str) -> None:
    """
    German: Bei Remote erzwingen wir 'query'-Modus, lokal lassen wir 'auto' (Pfad).
    """
    # Keep as env var for backends/bridge that read it
    if host in {"127.0.0.1", "localhost"}:
        os.environ["APP_COMFY_FORCE_VIEW_MODE"] = os.getenv("APP_COMFY_FORCE_VIEW_MODE", "auto") or "auto"
        print("[COMFY VIEW MODE] auto/path (host=127.0.0.1)")
    else:
        os.environ["APP_COMFY_FORCE_VIEW_MODE"] = "query"
        print("[COMFY VIEW MODE] query (remote host)")

def _apply_comfy_target(host: str, port: int) -> None:
    """
    Apply comfy host/port into STATE and environment, then rebuild backend.
    """
    STATE.comfy_host = host.strip()
    STATE.comfy_port = int(port)
    STATE.comfy_target = "local" if STATE.comfy_host in {"127.0.0.1", "localhost"} else "remote"
    _apply_view_mode_for_target(STATE.comfy_host)
    try:
        _rebuild_backend()
        _apply_active_workflow_if_local()
    except Exception as e:
        print(f"[BACKEND] rebuild after target apply failed: {e}")
    print(f"[BACKEND] switched -> {STATE.image_backend_name} | comfy_target={STATE.comfy_host}:{STATE.comfy_port} | view_mode={os.getenv('APP_COMFY_FORCE_VIEW_MODE','auto')}")

# ---------- Runtime-aware backend wrapper ----------
def _apply_env_for_backend() -> None:
    os.environ["IMAGE_BACKEND"] = STATE.image_backend_name
    os.environ["ALLOW_CLOUD_IMAGE_BACKEND"] = "1" if STATE.allow_cloud else "0"
    os.environ["APP_COMFY_HOST"] = STATE.comfy_host
    os.environ["APP_COMFY_PORT"] = str(STATE.comfy_port)

def build_image_backend_rt(backend_name: Optional[str] = None, allow_cloud: Optional[bool] = None) -> ImageBackend:
    """Build backend honoring current runtime STATE, then restore env snapshot."""
    wanted_backend = (backend_name or STATE.image_backend_name or _env_str("IMAGE_BACKEND", "comfyui")).lower()
    allowed = (STATE.allow_cloud if allow_cloud is None else bool(allow_cloud))

    snap = {
        "IMAGE_BACKEND": os.environ.get("IMAGE_BACKEND"),
        "ALLOW_CLOUD_IMAGE_BACKEND": os.environ.get("ALLOW_CLOUD_IMAGE_BACKEND"),
        "APP_COMFY_HOST": os.environ.get("APP_COMFY_HOST"),
        "APP_COMFY_PORT": os.environ.get("APP_COMFY_PORT"),
    }
    try:
        STATE.image_backend_name = wanted_backend
        STATE.allow_cloud = allowed
        _apply_env_for_backend()
        be = build_image_backend()
        return be
    finally:
        for k, v in snap.items():
            if v is None:
                with contextlib.suppress(Exception):
                    del os.environ[k]
            else:
                os.environ[k] = v

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
    """If LocalComfyBackend active, set its workflow path from STATE.active_workflow."""
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

# ---------- Reference→Denoise Mapping ----------
def _map_reference_strength_to_denoise(strength: float) -> float:
    """Map reference strength to denoise param for KSampler."""
    s = float(max(0.0, min(1.0, strength)))
    denoise = 0.85 - 0.55 * s
    return float(max(0.30, min(0.95, denoise)))

def _calc_effective_denoise_from_style(style: StyleConfig) -> Optional[float]:
    if not style or not getattr(style, "use_reference", False):
        return None
    if not getattr(style, "reference_id", None):
        return None
    if not _env_bool01("APP_REFERENCE_DENOISE_ENABLE", 1):
        return None
    return _map_reference_strength_to_denoise(getattr(style, "reference_strength", 0.6))

def _apply_denoise_to_local_comfy(backend: ImageBackend, denoise: Optional[float]) -> None:
    """Try to apply denoise to LocalComfyBackend via cfg or method."""
    if denoise is None or LocalComfyBackend is None or not isinstance(backend, LocalComfyBackend):
        return
    try:
        if hasattr(backend, "cfg") and hasattr(backend.cfg, "denoise"):
            setattr(backend.cfg, "denoise", float(denoise))
            print(f"[COMFY] denoise set via cfg: {denoise:.3f}")
            return
        if hasattr(backend, "set_sampler_denoise"):
            backend.set_sampler_denoise(float(denoise))  # type: ignore[attr-defined]
            print(f"[COMFY] denoise set via set_sampler_denoise: {denoise:.3f}")
            return
        print("[COMFY] denoise not supported on backend; skipping")
    except Exception as e:
        print(f"[COMFY] failed to set denoise: {e}")

def _patch_workflow_with_reference_if_needed(workflow_json: Dict[str, Any]) -> Dict[str, Any]:
    """Patch a Comfy workflow with an IP-Adapter reference if available."""
    try:
        sc = STATE.style_cfg
    except Exception:
        return workflow_json
    if not sc or not getattr(sc, "use_reference", False) or not getattr(sc, "reference_id", None):
        return workflow_json
    if reference_store is None or apply_ip_adapter_to_workflow is None:
        return workflow_json
    try:
        ref_path = reference_store.get_path(sc.reference_id)
    except Exception as e:
        print(f"[STYLE] reference_store.get_path failed: {e}")
        return workflow_json
    if not ref_path or not Path(ref_path).exists():
        print(f"[STYLE] reference file missing for id={sc.reference_id}")
        return workflow_json
    comfy_host = os.getenv("APP_COMFY_HOST", STATE.comfy_host).strip()
    comfy_port = int(os.getenv("APP_COMFY_PORT", str(STATE.comfy_port)) or STATE.comfy_port)
    try:
        patched = apply_ip_adapter_to_workflow(
            workflow_json,
            reference_path=Path(ref_path),
            reference_strength=getattr(sc, "reference_strength", 0.6),
            host=comfy_host,
            port=comfy_port,
            ref_image_node_id="8",
            ref_image_key="image",
            ipadapter_node_id="",
        )
        return patched if isinstance(patched, dict) else workflow_json
    except Exception as e:
        print(f"[STYLE] workflow patch failed: {e}")
        return workflow_json

async def _resolve_reference_for_pollinations(style: StyleConfig) -> tuple[Optional[str], Optional[float]]:
    """Resolve a public URL for Pollinations reference or None."""
    try:
        if not style or not getattr(style, "use_reference", False):
            return None, None
        rid = getattr(style, "reference_id", None)
        if not rid:
            return None, None
        if resolve_reference_urls_for_pollinations is None:
            print("[STYLE] resolve_reference_urls_for_pollinations not available")
            return None, None
        urls = await resolve_reference_urls_for_pollinations(style, STYLE_REFS_DIR)
        if urls and isinstance(urls[0], str) and urls[0].startswith("http"):
            strength = float(getattr(style, "reference_strength", 0.6))
            return urls[0], strength
        return None, None
    except Exception as e:
        print(f"[STYLE] pollinations resolve failed: {e}")
        return None, None

def _get_local_reference_path(style: StyleConfig) -> Optional[Path]:
    """Return local path for reference if present."""
    try:
        if not style or not getattr(style, "use_reference", False):
            return None
        rid = getattr(style, "reference_id", None)
        if not rid:
            return None
        store = ReferenceStore(STYLE_REFS_DIR)
        p = store.get_path(rid)
        if p and Path(p).exists():
            return Path(p)
    except Exception as e:
        print(f"[STYLE] local ref path resolve failed: {e}")
    return None

# ---------- Audio transcription loop ----------
async def audio_transcription_loop() -> None:
    """Main audio loop: capture frames, VAD, snapshot, transcribe, dispatch LLM+Image."""
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
        # German: Audio-Callback muss non-blocking bleiben → Queue mit Drop-on-full
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
        print("[AUDIO] loop exiting]")
    except asyncio.CancelledError:
        print("[AUDIO] loop cancelled")
    except Exception as e:
        print(f"[AUDIO] loop crashed: {e}")
        await broadcast("status", f"audio_loop_error:{e}")
    finally:
        await safe_stop_audio_stream()

# ---------- LLM + Image (style-aware) ----------
async def run_llm_and_image(text: str) -> None:
    """LLM tuning via Ollama, style build, then image generation."""
    if await _ollama_available() is False:
        await broadcast("status", "ollama_unavailable")
        STATE.last_llm_error = "ollama_unavailable"
        return
    if BACKEND is None:
        await broadcast("status", "image_backend_not_initialized")
        STATE.last_llm_error = "image_backend_not_initialized"
        return
    # Use guarded client for Ollama (non-Comfy)
    async with make_async_client(is_comfy_backend=False, limits=_httpx_limits_app(), timeout=_timeout_normal()) as client:
        try:
            base_prompt = await ollama_generate_prompt(client, text)
            if not base_prompt:
                STATE.last_llm_error = "llm_empty_response"
                await broadcast("status", "llm_empty_response")
                return
            built = build_style_prompt(base_prompt, STATE.style_cfg)
            STATE.last_prompt = built.positive
            await broadcast("llm_prompt", built.positive)
            await broadcast("status", "llm_ok")
            try:
                neg_global = (STATE.negative_prompt or "").strip()
                if LocalComfyBackend is not None and isinstance(BACKEND, LocalComfyBackend):
                    if hasattr(BACKEND, "cfg") and hasattr(BACKEND.cfg, "negative"):
                        setattr(BACKEND.cfg, "negative", neg_global or built.negative)
                eff_negative = (built.negative or neg_global or "").strip()
                is_comfy = (LocalComfyBackend is not None and isinstance(BACKEND, LocalComfyBackend))
                # Prepare style for Comfy locally (does not use httpx here unless backend does internally)
                if is_comfy:
                    try:
                        prepare_backend_style(BACKEND, STATE.style_cfg, STYLE_REFS_DIR)
                    except Exception as e:
                        print(f"[STYLE] prepare_backend_style (comfy) failed: {e}")
                denoise_override = _calc_effective_denoise_from_style(STATE.style_cfg)
                path = await _generate_with_negative_support(
                    prompt=built.positive,
                    width=STATE.image_width,
                    height=STATE.image_height,
                    negative=eff_negative,
                    denoise=denoise_override,
                )
                path = ensure_in_output_dir(path)
                rel = rel_for_ui_path(path)
                await broadcast("image", rel)
            except Exception as e:
                await broadcast("status", f"image_error:{e}")
        except Exception as e:
            STATE.last_llm_error = f"pipeline_error:{e}"
            await broadcast("status", f"pipeline_error:{e}")

# ---------- Helpers: negative prompt passing ----------
def _merge_negative_into_prompt(prompt: str, negative: str) -> str:
    p = (prompt or "").strip()
    n = (negative or "").strip()
    if not n:
        return p
    return f"{p}\n-- negative: {n}"

async def _generate_with_negative_support(prompt: str, width: int, height: int, negative: str, denoise: Optional[float] = None) -> Path:
    """Generate image with backend, with negative prompt fallback merging."""
    if BACKEND is None:
        raise RuntimeError("image_backend_not_initialized")
    try:
        _apply_active_workflow_if_local()
    except Exception as e:
        print(f"[WF] apply before generate failed: {e}")
    _apply_denoise_to_local_comfy(BACKEND, denoise)
    try:
        if LocalComfyBackend is not None and isinstance(BACKEND, LocalComfyBackend):
            if hasattr(BACKEND, "get_workflow_json") and hasattr(BACKEND, "set_workflow_json"):
                try:
                    wf_json = await BACKEND.get_workflow_json()  # type: ignore[attr-defined]
                    if isinstance(wf_json, dict):
                        wf_patched = _patch_workflow_with_reference_if_needed(wf_json)
                        if wf_patched is not wf_json:
                            await BACKEND.set_workflow_json(wf_patched)  # type: ignore[attr-defined]
                            print("[WF] workflow patched (reference staged + basename set).")
                except Exception as e:
                    print(f"[WF] patch via backend json failed: {e}")
    except Exception as e:
        print(f"[WF] backend patch hook error: {e}")
    gen_kwargs: Dict[str, Any] = {"width": width, "height": height}
    try:
        gen_kwargs["negative_prompt"] = (negative or "")
    except Exception:
        pass
    backend_name = (STATE.image_backend_name or "").lower()
    if backend_name == "pollinations":
        # German: Referenz-Bildweitergabe, multipart bevorzugt wenn lokal vorhanden, sonst URL (signiert)
        use_multipart = APP_POLLINATIONS_REF_MODE in {"auto", "multipart"}
        local_ref: Optional[Path] = None
        if use_multipart:
            local_ref = _get_local_reference_path(STATE.style_cfg)
        if local_ref is not None and local_ref.exists():
            gen_kwargs["style_reference_path"] = local_ref
            print("[STYLE] pollinations: using multipart local reference")
        else:
            ref_url, ref_strength = await _resolve_reference_for_pollinations(STATE.style_cfg)
            if ref_url and ref_strength is not None:
                gen_kwargs["style_reference_url"] = ref_url
                gen_kwargs["style_reference_strength"] = float(ref_strength)
                print(f"[STYLE] pollinations: using media_url with strength={ref_strength:.2f}")
            else:
                print("[STYLE] pollinations: no reference attached (none or resolver unavailable)")
    try:
        return await BACKEND.generate(prompt, **gen_kwargs)  # type: ignore[arg-type]
    except TypeError:
        merged = _merge_negative_into_prompt(prompt, negative)
        try:
            gen_kwargs.pop("negative_prompt", None)
            return await BACKEND.generate(merged, **gen_kwargs)
        except TypeError:
            return await BACKEND.generate(merged, width=width, height=height)

# ---------- FastAPI app & lifespan ----------
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan: init style, whisper, backends, warmup, and cleanup."""
    print(f"[ENV] loaded from: {ENV_PATH or '(env vars only)'}")
    try:
        STATE.style_cfg = _load_style_cfg()
    except Exception as e:
        print(f"[STYLE] load failed, using defaults: {e}")
        STATE.style_cfg = StyleConfig()
        STATE.style_cfg.persisted_path = STYLE_CFG_PATH
    _log_effective_config()
    init_whisper_model()
    STATE.comfy_host = _env_str("APP_COMFY_HOST", STATE.comfy_host)
    STATE.comfy_port = _env_int("APP_COMFY_PORT", STATE.comfy_port)
    STATE.comfy_target = "local" if STATE.comfy_host in {"127.0.0.1", "localhost"} else "remote"
    global BACKEND
    try:
        _assert_image_backend_host()
        BACKEND = build_image_backend_rt()
        print(f"[BACKEND] initialized: {type(BACKEND).__name__}")
    except Exception as e:
        BACKEND = None
        print(f"[BACKEND] initialization failed: {e}")
    try:
        dw, dh = _backend_default_size(STATE.image_backend_name)
        STATE.image_width = STATE.image_width or dw
        STATE.image_height = STATE.image_height or dh
    except Exception:
        STATE.image_width = APP_IMAGE_WIDTH
        STATE.image_height = APP_IMAGE_HEIGHT
    STATE.ollama_ready_at = time.time() + (WARMUP_GRACE_SEC if WARMUP_ENABLE else 0.0)
    print(f"[STATIC] /static -> {OUTPUT_DIR}")
    try:
        example = next(iter([p.name for p in OUTPUT_DIR.glob('*')]), "(none)")
        print(f"[STATIC] example file: {example}")
    except Exception:
        pass
    try:
        if LocalComfyBackend is not None and BACKEND is not None and isinstance(BACKEND, LocalComfyBackend):
            if not hasattr(BACKEND, "_copy_latest_from_comfy"):
                async def _copy_latest_from_comfy_shim() -> List[Path]:
                    items = [p for p in OUTPUT_DIR.glob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
                    items.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    return items[:4]
                setattr(BACKEND, "_copy_latest_from_comfy", _copy_latest_from_comfy_shim)
                print("[BACKEND] installed _copy_latest_from_comfy shim on LocalComfyBackend")
    except Exception as e:
        print(f"[BACKEND] shim install failed: {e}")
    warmup_task: Optional[asyncio.Task] = None
    if WARMUP_ENABLE:
        async def _silent_ollama_warmup():
            try:
                payload = {"model": OLLAMA_MODEL, "prompt": WARMUP_PROMPT, "stream": False, "options": {"temperature": 0.1, "num_predict": 32}}
                async with make_async_client(is_comfy_backend=False, limits=_httpx_limits_app(), timeout=_timeout_short_http()) as c:
                    with contextlib.suppress(Exception):
                        await c.get(_ollama_url("/api/tags"))
                async with make_async_client(is_comfy_backend=False, limits=_httpx_limits_app(), timeout=httpx.Timeout(WARMUP_TIMEOUT_SEC)) as client:
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
app.mount("/style_refs", StaticFiles(directory=str(STYLE_REFS_DIR), html=False), name="style_refs")
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


@app.post("/api/settings/image_backend")
async def api_switch_image_backend(req: ImageBackendSwitch):
    """German: Backend zwischen lokal, remote (WireGuard) und Pollinations umschalten."""
    target = req.backend.lower()
    if target not in {"comfyui", "comfyui_remote", "pollinations"}:
        return JSONResponse({"ok": False, "error": "invalid_backend"}, status_code=400)
    if target == "pollinations" and not STATE.allow_cloud:
        return JSONResponse({"ok": False, "error": "not_allowed", "reason": "cloud_blocked"}, status_code=403)
    if target == "pollinations":
        secret = _env_str("POLLINATIONS_SECRET", "")
        if not secret:
            return JSONResponse({"ok": False, "error": "missing_secret", "reason": "missing_secret"}, status_code=400)

    # Preserve current size unless reset requested
    cur_w, cur_h = STATE.image_width, STATE.image_height

    # Apply target-specific comfy host/port and view mode
    if target == "comfyui":
        _apply_comfy_target("127.0.0.1", _env_int("COMFY_LOCAL_PORT", _env_int("APP_COMFY_PORT", 8188)))
    elif target == "comfyui_remote":
        allow_remote = _env_bool01("APP_ALLOW_REMOTE_BACKENDS", 0)
        if not allow_remote:
            return JSONResponse({"ok": False, "error": "remote_not_allowed"}, status_code=403)
        rhost = _env_str("COMFY_REMOTE_HOST", STATE.comfy_host)
        rport = _env_int("COMFY_REMOTE_PORT", STATE.comfy_port or 8188)
        if not _host_in_whitelist(rhost, STATE.comfy_whitelist):
            return JSONResponse({"ok": False, "error": "host_not_whitelisted"}, status_code=403)
        _apply_comfy_target(rhost, rport)
    else:
        # pollinations: keep current comfy_host but block direct path; no target apply needed
        os.environ["APP_COMFY_FORCE_VIEW_MODE"] = os.getenv("APP_COMFY_FORCE_VIEW_MODE", "auto") or "auto"
        print("[BACKEND] pollinations selected; Comfy direct disabled")

    # Rebuild backend with new name
    try:
        _rebuild_backend(force_name=target)
        if req.reset:
            def_w, def_h = _backend_default_size(target)
            STATE.image_width, STATE.image_height = def_w, def_h
        else:
            STATE.image_width, STATE.image_height = cur_w, cur_h
        try:
            _apply_active_workflow_if_local()
        except Exception as e:
            print(f"[WF] apply after backend switch failed: {e}")
        await broadcast("status", f"image_backend:{target}")
        return {"ok": True, "backend": target, "width": STATE.image_width, "height": STATE.image_height, "reset": bool(req.reset)}
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"{e}"}, status_code=500)

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
    """German: Liefert UI-Config & Diagnosewerte."""
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
    sc = STATE.style_cfg
    style_ui = {
        "style_preset": sc.style_preset,
        "style_details": sc.style_details,
        "negative_base": sc.negative_base,
        "color_scheme": sc.color_scheme,
        "use_reference": sc.use_reference,
        "reference_id": sc.reference_id,
        "reference_strength": sc.reference_strength,
        "reference_base_url": "/style_refs",
    }
    cfg_ref = {
        "host": APP_REF_HOST or f"http://{os.getenv('APP_BIND_HOST','127.0.0.1')}:{int(os.getenv('APP_BIND_PORT','8080') or '8080')}",
        "ttl_sec": APP_REF_TTL_SEC,
        "secret_set": bool(APP_REF_SECRET),
    }
    comfy_block = {
        "enabled": bool(is_comfy and not comfy_disabled),
        "disabled_via_env": comfy_disabled,
        "comfyui_target": STATE.comfy_target,
    }
    if _debug_enabled():
        comfy_block["host"] = STATE.comfy_host
        comfy_block["port"] = STATE.comfy_port
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
            "active_workflow": STATE.active_workflow,
            "show_workflow_selector": show_workflow_selector,
            "comfy": comfy_block,
        },
        "style": style_ui,
        "style_ref_url": cfg_ref,
        "sse": {"tick_sec": APP_SSE_TICK_SEC},
    }

# ---------- Backend switch ----------
@app.post("/api/settings/image_backend")
async def api_switch_image_backend(req: ImageBackendSwitch):
    """German: Backend zwischen lokal, remote (WireGuard) und Pollinations umschalten."""
    target = req.backend.lower()
    if target not in {"comfyui", "comfyui_remote", "pollinations"}:
        return JSONResponse({"ok": False, "error": "invalid_backend"}, status_code=400)
    if target == "pollinations" and not STATE.allow_cloud:
        return JSONResponse({"ok": False, "error": "not_allowed", "reason": "cloud_blocked"}, status_code=403)
    if target == "pollinations":
        secret = _env_str("POLLINATIONS_SECRET", "")
        if not secret:
            return JSONResponse({"ok": False, "error": "missing_secret", "reason": "missing_secret"}, status_code=400)

    # Preserve current size unless reset requested
    cur_w, cur_h = STATE.image_width, STATE.image_height

    # Apply target-specific comfy host/port and view mode
    if target == "comfyui":
        _apply_comfy_target("127.0.0.1", _env_int("COMFY_LOCAL_PORT", _env_int("APP_COMFY_PORT", 8188)))
    elif target == "comfyui_remote":
        allow_remote = _env_bool01("APP_ALLOW_REMOTE_BACKENDS", 0)
        if not allow_remote:
            return JSONResponse({"ok": False, "error": "remote_not_allowed"}, status_code=403)
        rhost = _env_str("COMFY_REMOTE_HOST", STATE.comfy_host)
        rport = _env_int("COMFY_REMOTE_PORT", STATE.comfy_port or 8188)
        if not _host_in_whitelist(rhost, STATE.comfy_whitelist):
            return JSONResponse({"ok": False, "error": "host_not_whitelisted"}, status_code=403)
        _apply_comfy_target(rhost, rport)
    else:
        # pollinations: keep current comfy_host but block direct path; no target apply needed
        os.environ["APP_COMFY_FORCE_VIEW_MODE"] = os.getenv("APP_COMFY_FORCE_VIEW_MODE", "auto") or "auto"
        print("[BACKEND] pollinations selected; Comfy direct disabled")

    # Rebuild backend with new name
    try:
        _rebuild_backend(force_name=target)
        if req.reset:
            def_w, def_h = _backend_default_size(target)
            STATE.image_width, STATE.image_height = def_w, def_h
        else:
            STATE.image_width, STATE.image_height = cur_w, cur_h
        try:
            _apply_active_workflow_if_local()
        except Exception as e:
            print(f"[WF] apply after backend switch failed: {e}")
        await broadcast("status", f"image_backend:{target}")
        return {"ok": True, "backend": target, "width": STATE.image_width, "height": STATE.image_height, "reset": bool(req.reset)}
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"{e}"}, status_code=500)

# ---------- Workflows API ----------
@app.get("/api/workflows", response_model=WorkflowList)
async def api_list_workflows() -> WorkflowList:
    items = _list_workflow_files()
    return WorkflowList(items=items)

@app.post("/api/settings/workflow")
async def api_select_workflow(payload: WorkflowSelect):
    """German: Setzt aktiven Workflow (lokal Comfy)."""
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

# ---------- NEW: Comfy target switch ----------
def _host_in_whitelist(host: str, rules: List[str]) -> bool:
    """Check if host allowed against CIDR/IP/host rules."""
    h = (host or "").strip()
    if not h:
        return False
    try:
        ip = ipaddress.ip_address(h)
        for r in rules:
            r = r.strip()
            if not r:
                continue
            try:
                net = ipaddress.ip_network(r, strict=False)
                if ip in net:
                    return True
            except ValueError:
                pass
        return h in rules
    except ValueError:
        return h in rules

@app.post("/api/settings/comfy_target")
async def api_comfy_target(req: ComfyTargetReq):
    """German: Hart umschalten auf lokalen oder entfernten Comfy-Host (WireGuard)."""
    allow_remote = _env_bool01("APP_ALLOW_REMOTE_BACKENDS", 0)
    if req.target == "local":
        STATE.comfy_target = "local"
        STATE.comfy_host = "127.0.0.1"
        STATE.comfy_port = _env_int("APP_COMFY_PORT", STATE.comfy_port)
    else:
        if not allow_remote:
            return JSONResponse({"ok": False, "error": "remote_not_allowed"}, status_code=403)
        host = (req.host or STATE.comfy_host or "").strip()
        port = int(req.port or STATE.comfy_port or 8188)
        if not host:
            return JSONResponse({"ok": False, "error": "missing_host"}, status_code=400)
        if not _host_in_whitelist(host, STATE.comfy_whitelist):
            return JSONResponse({"ok": False, "error": "host_not_whitelisted"}, status_code=403)
        STATE.comfy_target = "remote"
        STATE.comfy_host = host
        STATE.comfy_port = port
    try:
        _rebuild_backend()
        try:
            _apply_active_workflow_if_local()
        except Exception as e:
            print(f"[WF] apply after comfy_target switch failed: {e}")
        await broadcast("status", f"comfy_target:{STATE.comfy_target}")
        return {"ok": True, "target": STATE.comfy_target, "host": STATE.comfy_host, "port": STATE.comfy_port}
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"{e}"}, status_code=500)

class ComfyPresetRequest(BaseModel):
    preset: Literal["local", "remote"]
    host: Optional[str] = None
    port: Optional[int] = None

@app.post("/api/settings/comfy_preset")
async def api_comfy_preset(req: ComfyPresetRequest):
    """
    German: Preset switch for Comfy. 'local' sets 127.0.0.1, 'remote' uses COMFY_REMOTE_* or UI overrides.
    """
    allow_remote = _env_bool01("APP_ALLOW_REMOTE_BACKENDS", 0)
    if req.preset == "local":
        host = "127.0.0.1"
        port = _env_int("COMFY_LOCAL_PORT", _env_int("APP_COMFY_PORT", 8188))
    else:
        if not allow_remote:
            return JSONResponse({"ok": False, "error": "remote_not_allowed"}, status_code=403)
        host = (req.host or os.getenv("COMFY_REMOTE_HOST") or STATE.comfy_host or "").strip()
        port = int(req.port or os.getenv("COMFY_REMOTE_PORT") or STATE.comfy_port or 8188)
        # German: Whitelist-Prüfung gegen bekannte sicheren Ziele
        if not _host_in_whitelist(host, STATE.comfy_whitelist):
            return JSONResponse({"ok": False, "error": "host_not_whitelisted"}, status_code=403)

    # German: Wendet Host/Port an, setzt view-mode und triggert Backend-Rebuild
    _apply_comfy_target(host, port)

    # German: UI-Benachrichtigung
    await broadcast("status", f"comfy_preset:{req.preset}")

    return {
        "ok": True,
        "preset": req.preset,
        "host": STATE.comfy_host,
        "port": STATE.comfy_port,
        "target": STATE.comfy_target,
    }

# ---------- SSE ----------
@app.get("/events")
async def events(request: Request):
    """SSE endpoint for UI updates (status, transcript, llm_prompt, image)."""
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
    """List available input devices."""
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
    """Record a short snippet to probe device peak and RMS."""
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

# ---------- Control routes ----------
@app.post("/start", response_class=PlainTextResponse)
async def start_pipeline():
    """Start audio capture and processing loop."""
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
    """Stop audio processing loop gracefully."""
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
    """Shutdown entire server; exits process after brief delay."""
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
class _OllamaErr(JSONResponse):
    pass

@app.post("/api/ollama/generate")
async def api_ollama_generate(req: OllamaGenerateRequest):
    """Direct Ollama generate endpoint proxy with retries."""
    if await _ollama_available() is False:
        return JSONResponse({"error": "ollama_unavailable"}, status_code=503)
    body = {"model": req.model or OLLAMA_MODEL, "prompt": req.prompt, "stream": bool(req.stream), "options": req.options or {}}
    async with make_async_client(is_comfy_backend=False, limits=_httpx_limits_app(), timeout=_timeout_normal()) as client:
        try:
            data = await _post_with_retries(client, _ollama_url("/api/generate"), body, timeout=float(OLLAMA_TIMEOUT_SEC))
            return {"response": data.get("response", "")}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/ollama/chat")
async def api_ollama_chat(req: OllamaChatRequest):
    """Direct Ollama chat endpoint proxy with retries."""
    if await _ollama_available() is False:
        return JSONResponse({"error": "ollama_unavailable"}, status_code=503)
    body = {"model": req.model or OLLAMA_MODEL, "messages": [m.model_dump() for m in req.messages], "stream": bool(req.stream), "options": req.options or {}}
    async with make_async_client(is_comfy_backend=False, limits=_httpx_limits_app(), timeout=_timeout_normal()) as client:
        try:
            data = await _post_with_retries(client, _ollama_url("/api/chat"), body, timeout=float(OLLAMA_TIMEOUT_SEC))
            msg = (data.get("message") or {}).get("content", "") if isinstance(data, dict) else ""
            return {"response": msg}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

# ---------- Style settings & reference upload ----------
@app.get("/api/settings/style")
async def get_style_settings():
    """Return current style settings for UI."""
    sc = STATE.style_cfg
    return {
        "style_preset": sc.style_preset,
        "style_details": sc.style_details,
        "negative_base": sc.negative_base,
        "color_scheme": sc.color_scheme,
        "use_reference": sc.use_reference,
        "reference_id": sc.reference_id,
        "reference_strength": sc.reference_strength,
        "reference_base_url": "/style_refs",
    }

@app.post("/api/settings/style")
async def set_style_settings(payload: StyleSettingsPayload):
    """Update style settings and persist."""
    try:
        sc = STATE.style_cfg
        sc.style_preset = (payload.style_preset or sc.style_preset).strip()
        sc.style_details = (payload.style_details or "").strip()
        sc.negative_base = (payload.negative_base or "").strip()
        sc.color_scheme = (payload.color_scheme or "").strip()
        sc.use_reference = bool(payload.use_reference)
        sc.reference_id = (payload.reference_id or None)
        sc.reference_strength = float(payload.reference_strength)
        _save_style_cfg(sc)
        if BACKEND is not None and LocalComfyBackend is not None and isinstance(BACKEND, LocalComfyBackend):
            try:
                prepare_backend_style(BACKEND, STATE.style_cfg, STYLE_REFS_DIR)
            except Exception as e:
                print(f"[STYLE] prepare_backend_style on update failed: {e}")
        await broadcast("status", "style_updated")
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"{e}"}, status_code=400)

@app.post("/api/reference/upload", response_model=ReferenceUploadResponse)
async def upload_reference_image(file: UploadFile = File(...)):
    """Upload a reference image to STYLE_REFS_DIR and enable it."""
    try:
        raw = await file.read()
    except Exception as e:
        return ReferenceUploadResponse(ok=False, error=f"read_error:{e}")
    try:
        store = ReferenceStore(STYLE_REFS_DIR)
        rid, path = store.put(file.filename or "ref.png", raw)
        STATE.style_cfg.reference_id = rid
        STATE.style_cfg.use_reference = True
        _save_style_cfg(STATE.style_cfg)
        if BACKEND is not None and LocalComfyBackend is not None and isinstance(BACKEND, LocalComfyBackend):
            with contextlib.suppress(Exception):
                prepare_backend_style(BACKEND, STATE.style_cfg, STYLE_REFS_DIR)
        return ReferenceUploadResponse(ok=True, reference_id=rid)
    except ValueError as e:
        return ReferenceUploadResponse(ok=False, error=f"invalid_image:{e}")
    except Exception as e:
        return ReferenceUploadResponse(ok=False, error=f"store_error:{e}")

@app.get("/api/reference/thumbnail")
async def reference_thumbnail(id: str = Query(..., min_length=1), size: int = Query(256, ge=32, le=2048)):
    """Return a PNG thumbnail of a stored reference."""
    src = (STYLE_REFS_DIR / id).resolve()
    if src.parent != STYLE_REFS_DIR or not src.exists() or not src.is_file():
        return JSONResponse({"error": "not_found"}, status_code=404)
    try:
        from PIL import Image  # Pillow
    except Exception:
        return Response(src.read_bytes(), media_type="image/png")
    try:
        im = Image.open(src).convert("RGB")
        im.thumbnail((size, size))
        import io as _io
        buf = _io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        buf.seek(0)
        return Response(buf.read(), media_type="image/png")
    except Exception as e:
        return JSONResponse({"error": f"thumb_error:{e}"}, status_code=500)

@app.post("/api/style/reference/on", response_model=StyleRefOnResponse)
async def api_style_reference_on(p: StyleRefOnPayload):
    """Enable an existing reference by id and strength."""
    try:
        rid = (p.reference_id or "").strip()
        if not rid:
            return StyleRefOnResponse(ok=False, error="missing_reference_id")
        store = ReferenceStore(STYLE_REFS_DIR)
        path = store.get_path(rid)
        if not path or not Path(path).exists():
            return StyleRefOnResponse(ok=False, error="not_found")
        sc = STATE.style_cfg
        sc.use_reference = True
        sc.reference_id = rid
        sc.reference_strength = float(max(0.0, min(1.0, p.reference_strength)))
        if p.reference_cloud is not None:
            try:
                sc.reference_cloud = bool(p.reference_cloud)  # type: ignore[attr-defined]
            except Exception:
                pass
        _save_style_cfg(sc)
        if BACKEND is not None and LocalComfyBackend is not None and isinstance(BACKEND, LocalComfyBackend):
            with contextlib.suppress(Exception):
                prepare_backend_style(BACKEND, STATE.style_cfg, STYLE_REFS_DIR)
        await broadcast("status", "style_reference:on")
        return StyleRefOnResponse(ok=True, reference_id=rid)
    except Exception as e:
        return StyleRefOnResponse(ok=False, error=f"{e}")

@app.post("/api/style/reference/off", response_model=StyleRefOffResponse)
async def api_style_reference_off():
    """Disable any active reference."""
    try:
        sc = STATE.style_cfg
        sc.use_reference = False
        sc.reference_id = None
        _save_style_cfg(sc)
        if BACKEND is not None and LocalComfyBackend is not None and isinstance(BACKEND, LocalComfyBackend):
            with contextlib.suppress(Exception):
                prepare_backend_style(BACKEND, STATE.style_cfg, STYLE_REFS_DIR)
        await broadcast("status", "style_reference:off")
        return StyleRefOffResponse(ok=True)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"{e}"}, status_code=500)

@app.get("/api/style/reference/thumbnail")
async def api_style_reference_thumbnail():
    """Return the direct static path for current reference, for UI preview."""
    sc = STATE.style_cfg
    if not sc or not getattr(sc, "use_reference", False) or not getattr(sc, "reference_id", None):
        return {"thumbnail": None}
    rid = getattr(sc, "reference_id", None)
    try:
        store = ReferenceStore(STYLE_REFS_DIR)
        p = store.get_path(rid)
        if not p or not Path(p).exists():
            return {"thumbnail": None}
        return {"thumbnail": f"/style_refs/{rid}"}
    except Exception:
        return {"thumbnail": None}

# ---------- Reference file delivery ----------
@app.get("/ref/{filename}")
async def get_reference_file(filename: str, ts: int = Query(...), sig: str = Query(...)):
    """Deliver a reference file when signature and TTL are valid."""
    try:
        data, mime = _verify_and_open_ref(filename, int(ts), sig or "")
        headers = {
            "Cache-Control": f"private, max-age={min(30, max(0, int(APP_REF_TTL_SEC//3)))}, must-revalidate",
            "X-Content-Type-Options": "nosniff",
            "X-Accel-Buffering": "no",
        }
        return Response(content=data, media_type=mime, headers=headers)
    except PermissionError as e:
        return JSONResponse({"error": f"forbidden:{e}"}, status_code=401)
    except FileNotFoundError:
        return JSONResponse({"error": "not_found"}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": f"server_error:{e}"}, status_code=500)

# ---------- Plan → Prompt ----------
@app.post("/api/plan")
async def api_plan(req: PlanRequest):
    """One-shot plan: text -> tuned prompt; triggers image gen in background."""
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
    async with make_async_client(is_comfy_backend=False, limits=_httpx_limits_app(), timeout=_timeout_normal()) as client:
        try:
            data = await _post_with_retries(client, _ollama_url("/api/generate"), body, timeout=float(OLLAMA_TIMEOUT_SEC))
            base_out = (data.get("response") or "").strip()
            built = build_style_prompt(base_out, STATE.style_cfg)
            if built.positive:
                STATE.last_prompt = built.positive
                await broadcast("llm_prompt", built.positive)
                if BACKEND is not None:
                    try:
                        neg_global = (STATE.negative_prompt or "").strip()
                        if LocalComfyBackend is not None and isinstance(BACKEND, LocalComfyBackend):
                            if hasattr(BACKEND, "cfg") and hasattr(BACKEND.cfg, "negative"):
                                setattr(BACKEND.cfg, "negative", neg_global or built.negative)
                        eff_w = req.width if (req.width and req.width > 0) else STATE.image_width
                        eff_h = req.height if (req.height and req.height > 0) else STATE.image_height
                        eff_negative = (built.negative or neg_global or "").strip()
                        is_comfy = (LocalComfyBackend is not None and isinstance(BACKEND, LocalComfyBackend))
                        if is_comfy:
                            with contextlib.suppress(Exception):
                                prepare_backend_style(BACKEND, STATE.style_cfg, STYLE_REFS_DIR)
                        denoise_override = _calc_effective_denoise_from_style(STATE.style_cfg)
                        path = await _generate_with_negative_support(
                            built.positive,
                            width=eff_w,
                            height=eff_h,
                            negative=eff_negative,
                            denoise=denoise_override,
                        )
                        path = ensure_in_output_dir(path)
                        rel = rel_for_ui_path(path)
                        await broadcast("image", rel)
                    except Exception as e:
                        await broadcast("status", f"image_error:{e}")
            return {"prompt": built.positive}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

# ---------- Image: Direct ----------
@app.post("/api/image/direct", response_model=ImageResponse)
async def api_image_direct(req: DirectImageRequest):
    """Generate an image directly from prompt; disabled for pollinations backend."""
    # German: Bei Cloud-Backend blockieren, damit kein 502 durch Comfy entsteht
    if (STATE.image_backend_name or "").lower() == "pollinations":
        return JSONResponse({"error": "direct_disabled_for_pollinations"}, status_code=400)

    if BACKEND is None:
        return JSONResponse({"error": "image_backend_not_initialized"}, status_code=500)
    try:
        built = build_style_prompt(req.prompt.strip(), STATE.style_cfg)
        neg_req = (req.negative_prompt or "").strip()
        neg_global = (STATE.negative_prompt or "").strip()
        eff_negative = (built.negative or neg_req or neg_global or "").strip()
        if LocalComfyBackend is not None and isinstance(BACKEND, LocalComfyBackend):
            if hasattr(BACKEND, "cfg") and hasattr(BACKEND.cfg, "negative"):
                setattr(BACKEND.cfg, "negative", eff_negative)
        w = req.width if (req.width and req.width > 0) else STATE.image_width
        h = req.height if (req.height and req.height > 0) else STATE.image_height
        is_comfy = (LocalComfyBackend is not None and isinstance(BACKEND, LocalComfyBackend))
        if is_comfy:
            with contextlib.suppress(Exception):
                prepare_backend_style(BACKEND, STATE.style_cfg, STYLE_REFS_DIR)
        denoise_override = _calc_effective_denoise_from_style(STATE.style_cfg)
        path = await _generate_with_negative_support(
            prompt=built.positive,
            width=w,
            height=h,
            negative=eff_negative,
            denoise=denoise_override,
        )
        path = ensure_in_output_dir(path)
        rel = rel_for_ui_path(path)
        await broadcast("image", rel)
        return ImageResponse(filename=path.name, relpath=rel, rel=rel, width=w, height=h)
    except PermissionError as e:
        return JSONResponse({"error": f"{e}"}, status_code=403)
    except Exception as e:
        return JSONResponse({"error": f"{e}"}, status_code=502)

# ---------- Image test ----------
@app.post("/api/image/test", response_model=ImageResponse)
async def api_image_test(req: ImageRequest):
    """Generate a test image using either provided prompt or a default one."""
    if BACKEND is None:
        return JSONResponse({"error": "image_backend_not_initialized"}, status_code=500)
    topic = (req.prompt or "A colorful low-poly fox head").strip()
    try:
        built = build_style_prompt(topic, STATE.style_cfg)
        neg_req = (req.negative_prompt or "").strip()
        neg_global = (STATE.negative_prompt or "").strip()
        eff_negative = (built.negative or neg_req or neg_global or "").strip()
        if LocalComfyBackend is not None and isinstance(BACKEND, LocalComfyBackend):
            if hasattr(BACKEND, "cfg") and hasattr(BACKEND.cfg, "negative"):
                setattr(BACKEND.cfg, "negative", eff_negative)
        w = req.width if (req.width and req.width > 0) else STATE.image_width
        h = req.height if (req.height and req.height > 0) else STATE.image_height
        is_comfy = (LocalComfyBackend is not None and isinstance(BACKEND, LocalComfyBackend))
        if is_comfy:
            with contextlib.suppress(Exception):
                prepare_backend_style(BACKEND, STATE.style_cfg, STYLE_REFS_DIR)
        denoise_override = _calc_effective_denoise_from_style(STATE.style_cfg)
        path = await _generate_with_negative_support(built.positive, width=w, height=h, negative=eff_negative, denoise=denoise_override)
        path = ensure_in_output_dir(path)
        rel = rel_for_ui_path(path)
        await broadcast("image", rel)
        return ImageResponse(filename=path.name, relpath=rel, rel=rel, width=w, height=h)
    except Exception as e:
        return JSONResponse({"error": f"{e}"}, status_code=502)

# ---------- Image backend switching & cloud toggle ----------
def _rebuild_backend(force_name: Optional[str] = None) -> ImageBackend:
    """Rebuild global BACKEND using current STATE and optional backend name override."""
    global BACKEND
    if force_name:
        STATE.image_backend_name = force_name.lower()
    BACKEND = build_image_backend_rt(backend_name=STATE.image_backend_name, allow_cloud=STATE.allow_cloud)
    return BACKEND

class ImageAllowCloudReq(BaseModel):
    allow: Optional[bool] = None
    allow_cloud: Optional[bool] = None

@app.post("/api/settings/image_allow_cloud")
async def api_image_allow_cloud(req: ImageAllowCloudReq):
    """Enable/disable cloud backend usage at runtime."""
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
        return JSONResponse({"ok": False, "error": f"{e}"}, status_code=500)

# ---------- Image size & negative prompt settings ----------
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

# ---------- Utility ----------
@app.get("/open_dir_hint")
async def open_dir_hint():
    return {"static_url": "/static/", "path": str(OUTPUT_DIR)}

@app.get("/api/image/latest")
async def api_image_latest():
    """Return the latest generated image file from OUTPUT_DIR."""
    try:
        items = [p for p in OUTPUT_DIR.glob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
        items.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if not items:
            return JSONResponse({"error": "no_images"}, status_code=404)
        rel = rel_for_ui_path(items[0])
        return {"rel": rel, "filename": items[0].name}
    except Exception as e:
        return JSONResponse({"error": f"{e}"}, status_code=500)

# ---------- App entry ----------
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("APP_BIND_HOST", "127.0.0.1")
    port = int(os.getenv("APP_BIND_PORT", "8080"))
    uvicorn.run("app:app", host=host, port=port, reload=False)
