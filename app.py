#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import asyncio, contextlib, os, re, time, json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Set, Dict, Any
import numpy as np
import httpx
import sounddevice as sd
from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict, field_validator
from contextlib import asynccontextmanager

# ---- ENV helpers ----
ENV_PATH: Optional[str] = None
try:
    from dotenv import load_dotenv, find_dotenv
    explicit = os.environ.get("ENV_FILE")
    if explicit:
        found = find_dotenv(explicit, usecwd=True)
        if found:
            load_dotenv(found, override=True); ENV_PATH = found
        elif os.path.isfile(explicit):
            load_dotenv(explicit, override=True); ENV_PATH = os.path.abspath(explicit)
    if ENV_PATH is None:
        found = find_dotenv(".env", usecwd=True)
        if found:
            load_dotenv(found, override=True); ENV_PATH = found
except Exception as e:
    print(f"[ENV] dotenv not available or failed: {e}")

def _env_str(k: str, d: str) -> str: return (os.getenv(k, d) or "").strip()
def _env_int(k: str, d: int) -> int:
    try: return int(os.getenv(k, str(d)))
    except: return d
def _env_float(k: str, d: float) -> float:
    try: return float(os.getenv(k, str(d)))
    except: return d
def _env_bool01(k: str, d: int=0) -> bool:
    v = (os.getenv(k, str(d)) or "").strip().lower()
    return v in {"1","true","yes","on"}

# ---- webrtcvad optional ----
try:
    import webrtcvad  # type: ignore
    WEBRTCVAD_AVAILABLE = True
except Exception as e:
    print(f"[WARN] could not import webrtcvad: {e}. Falling back to RMS-VAD.")
    webrtcvad = None  # type: ignore
    WEBRTCVAD_AVAILABLE = False

# ---- whisper.cpp via pywhispercpp optional ----
WHISPER_AVAILABLE = True
try:
    from pywhispercpp.model import Model as WhisperModel
except Exception as e:
    print(f"[WARN] could not import pywhispercpp: {e}. Whisper disabled.")
    WhisperModel = None  # type: ignore
    WHISPER_AVAILABLE = False

# ---- Config ----
AUDIO_DEVICE_PREF = _env_str("APP_AUDIO_DEVICE", "") or None
SAMPLE_RATE = _env_int("APP_SAMPLE_RATE", 48000)
FRAME_MS = _env_int("APP_FRAME_DURATION_MS", 20)
APP_STREAM_LATENCY_SEC = _env_float("APP_STREAM_LATENCY_SEC", 0.12)
SSE_TICK_SEC = _env_float("APP_SSE_TICK_SEC", 1.0)

DISABLE_VAD = _env_bool01("APP_DISABLE_VAD", 1)
USE_WEBRTC_VAD = _env_bool01("APP_USE_WEBRTC_VAD", 0) and WEBRTCVAD_AVAILABLE
VAD_AGGR = _env_int("APP_VAD_AGGRESSIVENESS", 0)
RMS_VAD_THRESHOLD = _env_float("APP_RMS_VAD_THRESHOLD", 0.02)

SNAPSHOT_SEC = _env_float("APP_SNAPSHOT_SEC", 4.0)
MIN_BUF_SEC = _env_float("APP_MIN_BUF_SEC", 0.8)
MAX_SILENCE_MS = _env_int("APP_MAX_SILENCE_MS", 1500)
MAX_SEGMENT_SEC = _env_float("APP_MAX_SEGMENT_SEC", 15.0)
FIRST_SNAPSHOT_DEADLINE_SEC = _env_float("APP_FIRST_SNAPSHOT_DEADLINE_SEC", 0.8)

WHISPER_MODEL_PATH = _env_str("APP_WHISPER_MODEL_PATH", "")
WHISPER_LANGUAGE = _env_str("APP_WHISPER_LANGUAGE", "de")
WHISPER_THREADS = _env_int("APP_WHISPER_THREADS", 2)
WHISPER_TEMPERATURE = _env_float("APP_WHISPER_TEMPERATURE", 0.0)
WHISPER_MIN_SEC = _env_float("APP_WHISPER_MIN_SEC", 0.6)
WHISPER_MIN_PEAK = _env_float("APP_WHISPER_MIN_PEAK", 0.004)

TEXT_MIN_CHARS = _env_int("APP_TEXT_MIN_CHARS", 6)
TEXT_MIN_WORDS = _env_int("APP_TEXT_MIN_WORDS", 2)
FORCE_MEANINGFUL_CHECK = _env_bool01("APP_FORCE_MEANINGFUL_CHECK", 1)

CONTEXT_MAX_SEGMENTS = _env_int("APP_CONTEXT_MAX_SEGMENTS", 5)
CONTEXT_MAX_CHARS = _env_int("APP_CONTEXT_MAX_CHARS", 480)

# Ollama
OLLAMA_HOST = _env_str("APP_OLLAMA_HOST", "127.0.0.1")
OLLAMA_PORT = _env_int("APP_OLLAMA_PORT", 11434)
OLLAMA_MODEL = _env_str("APP_OLLAMA_MODEL", "phi3:mini")
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
OLLAMA_SYS_PROMPT = _env_str("APP_OLLAMA_SYS_PROMPT",
    "Du bist ein präziser Prompt-Designer für Bildgeneratoren. "
    "Erzeuge kurze, klare, fotografische oder illustrative Bild-Prompts, "
    "ohne Meta-Kommentare, in Deutsch.")

# ComfyUI
COMFY_HOST = _env_str("APP_COMFY_HOST", "127.0.0.1")
COMFY_PORT = _env_int("APP_COMFY_PORT", 8188)
DISABLE_COMFYUI = _env_bool01("APP_DISABLE_COMFYUI", 1)

OUTPUT_DIR = Path(_env_str("APP_OUTPUT_DIR", "./outputs/images")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Warmup
WARMUP_ENABLE = _env_bool01("APP_OLLAMA_WARMUP_ENABLE", 1)
WARMUP_PROMPT = _env_str("APP_OLLAMA_WARMUP_PROMPT", "Sag Hallo auf Deutsch.")
WARMUP_TIMEOUT_SEC = _env_float("APP_OLLAMA_WARMUP_TIMEOUT_SEC", 45.0)
WARMUP_MAX_RETRIES = _env_int("APP_OLLAMA_WARMUP_MAX_RETRIES", 3)
WARMUP_RETRY_DELAY = _env_float("APP_OLLAMA_WARMUP_RETRY_DELAY", 1.2)
WARMUP_GRACE_SEC = _env_float("APP_OLLAMA_WARMUP_GRACE_SEC", 10.0)

def assert_local(host: str) -> None:
    # Nur localhost zulassen
    if host != "127.0.0.1":
        raise AssertionError(f"Only localhost allowed, got {host}")
assert_local(OLLAMA_HOST); assert_local(COMFY_HOST)

# ---- Cloud model detection ----
def is_cloud_model(name: str) -> bool:
    return ":cloud" in (name or "")

# ---- Pydantic payloads ----
class OllamaGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    prompt: str
    stream: bool = False
    options: dict = Field(default_factory=dict)

class OllamaChatTurn(BaseModel):
    role: Literal["system","user","assistant"]
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

class ComfyPromptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: dict
    @field_validator("prompt")
    @classmethod
    def ensure_non_empty(cls, v: dict) -> dict:
        if not isinstance(v, dict) or not v:
            raise ValueError("prompt must not be empty")
        return v

class ComfyPromptResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    prompt_id: str = Field(alias="prompt_id")

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
            if prefer.lower() in (d.get("name","").lower()) and d.get("max_input_channels",0) > 0:
                return i
    for i, d in enumerate(devs):
        if "pulse" in (d.get("name","").lower()) and d.get("max_input_channels",0) > 0:
            return i
    for i, d in enumerate(devs):
        if d.get("max_input_channels",0) > 0:
            return i
    raise RuntimeError("No input device with max_input_channels>0 found.")

def to_int16(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16, copy=False)

def rms_vad(frame: np.ndarray, rms_threshold: float = 0.01) -> bool:
    # Einfache RMS-VAD
    if frame.size == 0: return False
    rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float32), dtype=np.float64)))
    return rms >= rms_threshold

def resample_to_16k(samples: np.ndarray, sr: int) -> np.ndarray:
    # Leichtgewichtiges Resampling auf 16 kHz
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

# ---- Whisper ----
_WHISPER_MODEL: Optional[WhisperModel] = None
def init_whisper_model() -> None:
    global _WHISPER_MODEL
    if not WHISPER_AVAILABLE or _WHISPER_MODEL is not None: return
    if not WHISPER_MODEL_PATH or not Path(WHISPER_MODEL_PATH).is_file():
        print(f"[WHISPER] model not found/disabled: {WHISPER_MODEL_PATH}"); return
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
        print(f"[WHISPER] model loaded: {Path(WHISPER_MODEL_PATH).name}, threads={WHISPER_THREADS}, lang={WHISPER_LANGUAGE}")
    except Exception as e:
        print(f"[WHISPER] initialization failed: {e}"); _WHISPER_MODEL=None

TEXT_FIELD_RE = re.compile(r'text\s*=\s*(.+?)(?:,|$)')
META_RE = re.compile(r"\b(musik|music|applaus|applause|lachen|laugh|geräusch|noise|husten|cough|klatschen|klingel|ring|summen|hmm+|pause)\b", re.I)

def _parse_whisper_out(raw: object) -> str:
    # Robust gegen dict/segments/str
    if raw is None: return ""
    if isinstance(raw, dict):
        if isinstance(raw.get("text"), str): return raw["text"]
        segs = raw.get("segments")
        if isinstance(segs, list):
            return " ".join(str(s.get("text","")).strip() for s in segs if isinstance(s, dict)).strip()
        return ""
    s = str(raw).strip()
    if not s or s == "[]": return ""
    if s.startswith("[") and "text=" in s:
        parts = TEXT_FIELD_RE.findall(s)
        if parts:
            cleaned = []
            for t in parts:
                t=t.strip()
                if len(t)>=2 and t[0]==t[-1] and t[0] in "\"'": t=t[1:-1]
                cleaned.append(t.strip())
            return " ".join(cleaned).strip()
    return s

def clean_transcript(raw: str) -> str:
    if not raw: return ""
    txt = " ".join(raw.split()).strip()
    if not txt: return ""
    if META_RE.search(txt) and len(txt.split()) <= 3: return ""
    if len(txt.split()) == 1 and txt.lower() in {"ja","und","also","äh","oh"}: return ""
    return txt

def is_meaningful_text(t: str, min_chars: int, min_words: int) -> bool:
    t=(t or "").strip()
    return bool(t) and len(t)>=min_chars and len(t.split())>=min_words and re.search(r"[A-Za-zÄÖÜäöüß]", t)

def transcribe_chunk_with_whisper(samples: np.ndarray, sr: int) -> str:
    if not WHISPER_AVAILABLE or _WHISPER_MODEL is None: return ""
    if samples.size == 0: return ""
    peak = float(np.max(np.abs(samples)))
    if peak < WHISPER_MIN_PEAK:
        print(f"[WHISPER] below_min_peak peak={peak:.4f} th={WHISPER_MIN_PEAK:.4f}"); return ""
    min_sec = max(0.0, float(WHISPER_MIN_SEC))
    if samples.size < int(sr * min_sec):
        pad = int(sr * min_sec) - samples.size
        samples = np.concatenate([samples, np.zeros(pad, dtype=np.float32)], axis=0)
    if sr != 16000:
        samples = resample_to_16k(samples, sr)
        if samples.size == 0: return ""
    try:
        if hasattr(_WHISPER_MODEL, "transcribe_float32"):
            raw = _WHISPER_MODEL.transcribe_float32(samples)
        elif hasattr(_WHISPER_MODEL, "transcribe"):
            raw = _WHISPER_MODEL.transcribe(samples)
        else:
            raw = _WHISPER_MODEL.transcribe_pcm16(to_int16(samples))
        txt = clean_transcript(_parse_whisper_out(raw))
        if txt: print(f"[WHISPER] text: {txt}")
        else: print("[WHISPER] raw→empty")
        return txt
    except KeyboardInterrupt:
        return ""
    except Exception as e:
        print(f"[WHISPER] transcription failed: {e}"); return ""

# ---- HTTP utils ----
def _httpx_limits() -> httpx.Limits: return httpx.Limits(max_keepalive_connections=0, max_connections=6)
def _timeout_short() -> httpx.Timeout: return httpx.Timeout(connect=2.0, read=3.0, write=3.0, pool=2.0)
def _timeout_normal() -> httpx.Timeout:
    base = OLLAMA_TIMEOUT_SEC
    if is_cloud_model(OLLAMA_MODEL):
        base = max(base, 75.0)
    t = min(max(5.0, base), 120.0)
    return httpx.Timeout(connect=5.0, read=t, write=5.0, pool=5.0)

async def http_post_json(client: httpx.AsyncClient, url: str, json: dict, timeout: float = 30.0) -> dict:
    r = await client.post(url, json=json, timeout=timeout); r.raise_for_status(); return r.json()
async def http_get_json(client: httpx.AsyncClient, url: str, timeout: float = 15.0) -> dict:
    r = await client.get(url, timeout=timeout); r.raise_for_status(); return r.json()

# ---- Ollama helpers ----
def _ollama_url(path: str) -> str: return f"http://{OLLAMA_HOST}:{OLLAMA_PORT}{path}"
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
    delay = OLLAMA_RETRY_BASE_DELAY; last_exc=None
    for attempt in range(1, OLLAMA_MAX_RETRIES+1):
        try:
            resp = await client.post(url, json=body, timeout=timeout)
            resp.raise_for_status(); return resp.json()
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError, httpx.HTTPStatusError) as e:
            last_exc=e
            status = getattr(e, "response", None).status_code if getattr(e, "response", None) else None
            retryable = status in (429,500,502,503) or isinstance(e, (httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError, httpx.ConnectError))
            print(f"[OLLAMA] attempt {attempt} failed (status={status}): {e}")
            if attempt >= OLLAMA_MAX_RETRIES or not retryable: break
            await asyncio.sleep(delay); delay *= 1.8 if is_cloud_model(OLLAMA_MODEL) else 2.0
    raise RuntimeError(f"Ollama request failed after {OLLAMA_MAX_RETRIES} attempts: {last_exc}")

def _neutral_system_prompt(lang: str = "de") -> str:
    if OLLAMA_SYS_PROMPT:
        return OLLAMA_SYS_PROMPT
    if (lang or "de").lower().startswith("en"):
        return ("You are an expert image prompt engineer. Write concise, vivid prompts for image generation models (e.g., SDXL). "
                "Avoid meta text. Focus on subject, style, lighting, composition, attributes. 1–2 sentences. Reply in user's language.")
    return ("Du bist Expert/in für Bild-Prompts. Schreibe kurze, prägnante, anschauliche Prompts für Bildgeneratoren "
            "(z. B. SDXL), ohne Meta-Kommentare. Fokus: Motiv, Stil, Licht, Komposition, Attribute. 1–2 Sätze.")

def _neutral_prompt_payload(user_text: str, tags: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "user_text": (user_text or "").strip(),
        "tags": tags or [],
        "constraints": {"no_meta": True, "max_sentences": 2, "avoid_sensitive": True},
        "output_hint": "One compact image prompt, no explanations."
    }

def _build_neutral_prompt_text(user_text: str, lang: str = "de") -> str:
    sys = _neutral_system_prompt(lang)
    payload = _neutral_prompt_payload(user_text)
    return f"<<SYS>>{sys}<</SYS>>\n\nINPUT_JSON:\n{json.dumps(payload, ensure_ascii=False)}\n\nOUTPUT:\n"

def _make_ollama_prompt(user_text: str) -> str:
    return _build_neutral_prompt_text(user_text, lang=WHISPER_LANGUAGE or "de")

async def _ollama_available() -> bool:
    try:
        async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_short()) as c:
            r = await c.get(_ollama_url("/api/tags")); r.raise_for_status()
            return True
    except Exception:
        return False

async def ollama_generate_prompt(client: httpx.AsyncClient, user_text: str) -> str:
    prompt_text = _build_neutral_prompt_text(user_text, lang=WHISPER_LANGUAGE or "de")
    options = _ollama_options_for_prompt()
    body = {"model": OLLAMA_MODEL, "prompt": prompt_text, "stream": False, "options": options}
    url = _ollama_url("/api/generate")
    timeout = min(120.0, max(OLLAMA_TIMEOUT_SEC, 75.0) if is_cloud_model(OLLAMA_MODEL) else OLLAMA_TIMEOUT_SEC)
    data = await _post_with_retries(client, url, body, timeout=timeout)
    return (data.get("response") or "").strip()

async def _silent_ollama_warmup() -> None:
    if not WARMUP_ENABLE: return
    deadline = time.time() + 12.0
    while time.time() < deadline:
        try:
            async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_short()) as client:
                r = await client.get(_ollama_url("/api/tags"))
                if r.status_code == 200: break
        except Exception:
            await asyncio.sleep(0.4)
    else:
        print("[WARMUP] Ollama not reachable within 12s, skip warmup."); return
    payload = {"model": OLLAMA_MODEL, "prompt": WARMUP_PROMPT, "stream": False, "options": {"temperature": 0.1, "num_predict": 32}}
    for attempt in range(1, WARMUP_MAX_RETRIES+1):
        try:
            async with httpx.AsyncClient(limits=_httpx_limits(), timeout=httpx.Timeout(WARMUP_TIMEOUT_SEC)) as client:
                resp = await client.post(_ollama_url("/api/generate"), json=payload)
                if resp.status_code == 200:
                    print("[WARMUP] Ollama warmup ok."); return
                else:
                    print(f"[WARMUP] status={resp.status_code} body={resp.text[:200]}")
        except Exception as e:
            print(f"[WARMUP] attempt {attempt} failed: {e}")
        await asyncio.sleep(WARMUP_RETRY_DELAY)
    print("[WARMUP] warmup failed after retries.")

# ---- ComfyUI ----
def _comfy_url(path: str) -> str: return f"http://{COMFY_HOST}:{COMFY_PORT}{path}"
async def _comfy_available(client: httpx.AsyncClient) -> bool:
    try: _ = await http_get_json(client, _comfy_url("/queue")); return True
    except Exception:
        try: _ = await http_get_json(client, _comfy_url("/history")); return True
        except Exception: return False

def build_comfy_prompt_from_text(text: str) -> ComfyPromptRequest:
    # Platzhalter – bitte mit deinem echten ComfyUI-Workflow ersetzen
    payload = {"3": {"inputs": {"text": text}, "class_type": "CLIPTextEncode", "_meta": {"title": "PromptEncoder"}}}
    return ComfyPromptRequest(prompt=payload)

async def comfyui_run_and_wait(client: httpx.AsyncClient, req: ComfyPromptRequest, poll_interval: float = 1.0) -> Tuple[str, Optional[str]]:
    resp = await http_post_json(client, _comfy_url("/prompt"), req.model_dump())
    pr = ComfyPromptResponse.model_validate(resp)
    prompt_id = pr.prompt_id
    url_hist = _comfy_url(f"/history/{prompt_id}")
    print(f"[COMFY] prompt id={prompt_id} submitted, waiting for result...")
    for _ in range(600):
        await asyncio.sleep(poll_interval)
        hist = await http_get_json(client, url_hist)
        outputs = hist.get(prompt_id, {}).get("outputs", {})
        for _node_id, node_out in outputs.items():
            imgs = node_out.get("images") or []
            if imgs:
                img = imgs[0]
                filename = img.get("filename"); subfolder = img.get("subfolder") or ""
                rel_path = f"{subfolder}/{filename}" if subfolder else filename
                return prompt_id, rel_path
    return prompt_id, None

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

STATE = PipelineState()

def sse_format(event: str, data: str) -> str: return f"event: {event}\ndata: {data}\n\n"

async def broadcast(event: str, data: str) -> None:
    if STATE.shutting_down: return
    for q in list(STATE.listeners):
        with contextlib.suppress(Exception):
            await q.put(sse_format(event, data))

_context_buffer: deque[str] = deque(maxlen=CONTEXT_MAX_SEGMENTS)
def update_context_buffer(text: str) -> str:
    _context_buffer.append(text)
    ctx = " ".join(_context_buffer)
    if len(ctx) > CONTEXT_MAX_CHARS: ctx = ctx[-CONTEXT_MAX_CHARS:]
    return ctx

def _log_effective_config() -> None:
    print(
        "[CONFIG] env_file=", ENV_PATH or "(none)",
        "| audio:", f"sr={SAMPLE_RATE} frame_ms={FRAME_MS} stream_lat={APP_STREAM_LATENCY_SEC}",
        "| vad:", f"disable={DISABLE_VAD} webrtc={USE_WEBRTC_VAD} aggr={VAD_AGGR} rms_th={RMS_VAD_THRESHOLD}",
        "| snap:", f"snapshot_sec={SNAPSHOT_SEC} min_buf_sec={MIN_BUF_SEC} max_sil_ms={MAX_SILENCE_MS} max_seg={MAX_SEGMENT_SEC}",
        "| whisper:", f"min_sec={WHISPER_MIN_SEC} min_peak={WHISPER_MIN_PEAK} lang={WHISPER_LANGUAGE}",
        "| text:", f"min_chars={TEXT_MIN_CHARS} min_words={TEXT_MIN_WORDS} force_meaningful={FORCE_MEANINGFUL_CHECK}",
        "| llm:", f"interval={LLM_INTERVAL_SEC}s model={OLLAMA_MODEL} cloud={is_cloud_model(OLLAMA_MODEL)}",
        "| comfy:", f"disabled={DISABLE_COMFYUI}",
        "| warmup:", f"enable={WARMUP_ENABLE} grace={WARMUP_GRACE_SEC}s",
    )

# ---- Audio main loop ----
def _frame_to_webrtc_bytes(frame: np.ndarray, sr: int) -> bytes:
    return to_int16(frame).tobytes()

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
        sd.default.device = (dev_index, None); sd.default.samplerate = SAMPLE_RATE; sd.default.channels = 1
    except Exception as e:
        print(f"[AUDIO] could not set sd.default.*: {e}")

    frame_samples = int(SAMPLE_RATE * FRAME_MS / 1000)
    audio_frames: deque[np.ndarray] = deque()
    got_cb_frames = 0; overflow_count = 0

    def sd_callback(indata, frames, timeinfo, status):
        nonlocal got_cb_frames, overflow_count
        try:
            if status:
                if "overflow" in str(status).lower(): overflow_count += 1
            mono = np.asarray(indata[:, 0], dtype=np.float32).copy()
            audio_frames.append(mono); got_cb_frames += 1
        except Exception as e:
            print(f"[AUDIO] callback exception: {e}")

    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            device=(dev_index, None), blocksize=frame_samples,
            latency=APP_STREAM_LATENCY_SEC, callback=sd_callback,
        )
        stream.start()
        t0 = time.time()
        while time.time() - t0 < 1.5 and got_cb_frames == 0 and STATE.running:
            await asyncio.sleep(0.05)
        if got_cb_frames == 0:
            with contextlib.suppress(Exception): stream.stop(); stream.close()
            await broadcast("status", "audio_callback_no_frames")
            print("[AUDIO] ERROR: callback produced no frames; aborting"); return
        print(f"[AUDIO] open OK: device_index={dev_index}, name='{dev_name}', samplerate={SAMPLE_RATE}, blocksize={frame_samples}, latency={APP_STREAM_LATENCY_SEC}s")
    except Exception as e:
        await broadcast("status", f"audio_stream_failed: {e}")
        print(f"[AUDIO] stream open failed: {e}"); return

    STATE.actual_sr = SAMPLE_RATE; STATE.device_used_index = dev_index; STATE.device_used_name = dev_name

    last_snapshot = time.time(); last_tick = time.time(); total_frames = 0
    current_segment_frames: List[np.ndarray] = []; silence_ms = 0
    endpoint_silence_ms = MAX_SILENCE_MS
    max_segment_frames = int(((float(MAX_SEGMENT_SEC) * 1000) / FRAME_MS))

    vad = None
    if USE_WEBRTC_VAD:
        try: vad = webrtcvad.Vad(int(VAD_AGGR))
        except Exception as e: print(f"[VAD] webrtcvad init failed: {e}"); vad=None

    await broadcast("status", f"recording_start sr={SAMPLE_RATE}")
    await broadcast("status", f"device_used idx={dev_index} name={dev_name}")
    if is_cloud_model(OLLAMA_MODEL):
        await broadcast("status", "Hinweis: Cloud-LLM aktiv, Prompts werden an ollama.com gesendet.")

    first_snapshot_deadline = time.time() + FIRST_SNAPSHOT_DEADLINE_SEC

    try:
        while STATE.running:
            await asyncio.sleep(FRAME_MS/1000.0)
            if not STATE.running: break
            now = time.time()
            if now - last_tick >= SSE_TICK_SEC:
                last_tick = now
                with contextlib.suppress(Exception):
                    await broadcast("status", f"tick frames={total_frames} seg_frames={len(current_segment_frames)} sr={SAMPLE_RATE} overflows={overflow_count}")

            if not audio_frames: continue
            frame = audio_frames.popleft(); total_frames += 1
            target = frame_samples
            if len(frame)!=target:
                if len(frame)<target: frame = np.pad(frame, (0, target-len(frame)))
                else: frame = frame[:target]
            frame = np.clip(frame, -1.0, 1.0)

            if DISABLE_VAD: is_speech=True
            else:
                if vad is not None:
                    try: is_speech = vad.is_speech(_frame_to_webrtc_bytes(frame, SAMPLE_RATE), SAMPLE_RATE)
                    except Exception: is_speech = rms_vad(frame, rms_threshold=RMS_VAD_THRESHOLD)
                else:
                    is_speech = rms_vad(frame, rms_threshold=RMS_VAD_THRESHOLD)

            current_segment_frames.append(frame)
            if len(current_segment_frames) > max_segment_frames:
                is_segment_end = True
            else:
                silence_ms = 0 if is_speech else (silence_ms + FRAME_MS)
                is_segment_end = (silence_ms >= endpoint_silence_ms)

            buf_sec = (len(current_segment_frames) * FRAME_MS) / 1000.0
            do_snapshot_time = ((now - last_snapshot) >= SNAPSHOT_SEC) or (now >= first_snapshot_deadline)
            do_snapshot = (do_snapshot_time or is_segment_end) and (buf_sec >= MIN_BUF_SEC)

            if do_snapshot:
                last_snapshot = now
                if now >= first_snapshot_deadline: first_snapshot_deadline = float("inf")
                seg = np.frombuffer(b"".join([f.tobytes() for f in current_segment_frames]), dtype=np.float32)
                if seg.size > 0:
                    peak = float(np.max(np.abs(seg)))
                    if peak >= WHISPER_MIN_PEAK:
                        txt = transcribe_chunk_with_whisper(seg, SAMPLE_RATE)
                        if txt:
                            await broadcast("transcript", txt)
                            meaningful_ok = (not FORCE_MEANINGFUL_CHECK) or is_meaningful_text(txt, TEXT_MIN_CHARS, TEXT_MIN_WORDS)
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
                                    t = asyncio.create_task(run_llm_and_optionally_image(update_context_buffer(txt)))
                                    STATE.bg_tasks.add(t)
                                    t.add_done_callback(lambda fut: STATE.bg_tasks.discard(fut))
                        else:
                            await broadcast("status", "whisper_empty(no_text)")
                    else:
                        await broadcast("status", f"low_peak({peak:.3f})")
                if is_segment_end:
                    current_segment_frames.clear(); silence_ms = 0
    except asyncio.CancelledError:
        pass
    finally:
        with contextlib.suppress(Exception):
            stream.stop(); stream.close()
        await broadcast("status", "recording_stop")
        print("[AUDIO] stream closed")

# ---- LLM + Image ----
async def run_llm_and_optionally_image(text: str) -> None:
    # LLM Schritt (Ollama) → optional Bild (ComfyUI)
    if await _ollama_available() is False:
        await broadcast("status", "ollama_unavailable"); STATE.last_llm_error = "ollama_unavailable"; return
    async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_normal()) as client:
        try:
            img_prompt = await ollama_generate_prompt(client, text)
            if not img_prompt:
                STATE.last_llm_error = "llm_empty_response"; await broadcast("status","llm_empty_response"); return
            STATE.last_prompt = img_prompt
            await broadcast("llm_prompt", img_prompt); await broadcast("status","llm_ok")
            if DISABLE_COMFYUI:
                await broadcast("status","comfy_disabled"); return
            if not await _comfy_available(client):
                await broadcast("status","comfy_unavailable"); return
            req = build_comfy_prompt_from_text(img_prompt)
            pid, rel_img_path = await comfyui_run_and_wait(client, req)
            if rel_img_path: await broadcast("image", rel_img_path)
            else: await broadcast("status", "comfy_timeout")
        except Exception as e:
            STATE.last_llm_error = f"pipeline_error:{e}"
            await broadcast("status", f"pipeline_error:{e}")

# ---- FastAPI & Lifespan ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[ENV] loaded from: {ENV_PATH or '(env vars only)'}")
    _log_effective_config()
    init_whisper_model()
    STATE.ollama_ready_at = time.time() + (WARMUP_GRACE_SEC if WARMUP_ENABLE else 0.0)
    warmup_task: Optional[asyncio.Task] = None
    if WARMUP_ENABLE:
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

INDEX_HTML = """<!doctype html>
<html lang="de"><head><meta charset="utf-8"><title>Vorlesen → Bilder (Lokal)</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font-family:system-ui,sans-serif;margin:0;padding:1rem;background:#0b1020;color:#e8ecf1}
header{display:flex;gap:1rem;align-items:center;flex-wrap:wrap;margin-bottom:.75rem}
button{background:#1f6feb;color:#fff;border:0;padding:.6rem 1rem;border-radius:8px;cursor:pointer}
button.stop{background:#c53b3b}#status{opacity:.9;font-size:.9rem}
#live{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:.75rem}
.panel{background:#141a2e;border-radius:10px;overflow:hidden;box-shadow:0 0 0 1px rgba(255,255,255,.06) inset}
.panel-title{font-size:.9rem;font-weight:600;padding:.5rem .75rem;color:#c7d1df;border-bottom:1px solid rgba(255,255,255,.06)}
.panel-body{padding:.6rem .75rem;min-height:52px;white-space:pre-wrap;word-break:break-word}
.mono{font-family:ui-monospace,Menlo,Consolas,"Liberation Mono",monospace}
#grid{margin-top:.5rem;display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}
.card{background:#141a2e;border-radius:10px;overflow:hidden;box-shadow:0 0 0 1px rgba(255,255,255,.06) inset}
.card img{width:100%;display:block}.cap{padding:.5rem .75rem;font-size:.85rem;color:#c7d1df}
.controls{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap}
input[type="text"]{background:#0e1426;color:#e8ecf1;border:1px solid #283355;border-radius:8px;padding:.45rem .6rem;min-width:260px}
.small{font-size:.85rem;opacity:.85}
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
    <div class="panel"><div class="panel-title">Prompt</div><div id="prompt" class="panel-body"></div></div>
  </section>
  <div class="small" id="notice"></div>
  <h4>Images</h4>
  <div id="grid"></div>
</main>
<script>
const statusEl=document.getElementById('status');
const promptEl=document.getElementById('prompt');
const transcriptEl=document.getElementById('transcript');
const grid=document.getElementById('grid');
const notice=document.getElementById('notice');
let evtSrc=null, hb=null;
function setStatus(m){statusEl.textContent=m}
function setPrompt(p){promptEl.textContent=p||''}
function setTranscript(t){transcriptEl.textContent=t||''}
function markCloudNotice(){
  fetch('/config').then(r=>r.json()).then(cfg=>{
    if(cfg.ollama && cfg.ollama.cloud===true){
      notice.textContent='Hinweis: Cloud-LLM aktiv, Prompts werden an ollama.com gesendet.';
    }else{
      notice.textContent='Rein lokal (Ollama & ComfyUI auf 127.0.0.1).';
    }
  }).catch(()=>{});
}
async function start(){
  if(evtSrc) evtSrc.close();
  try{
    const r=await fetch('/health/ollama'); const j=await r.json();
    if(!j.ok){ console.warn('Ollama not OK', j); }
  }catch(e){ console.warn('Ollama health failed', e); }
  evtSrc=new EventSource('/events');
  evtSrc.addEventListener('status', e=>setStatus(e.data));
  evtSrc.addEventListener('transcript', e=>setTranscript(e.data));
  evtSrc.addEventListener('llm_prompt', e=>setPrompt(e.data));
  evtSrc.addEventListener('image', e=>{
    const rel=e.data; const src='/static/'+rel;
    const card=document.createElement('div'); card.className='card';
    const img=document.createElement('img'); img.src=src+'?t='+Date.now();
    const cap=document.createElement('div'); cap.className='cap'; cap.textContent=new Date().toLocaleTimeString();
    card.appendChild(img); card.appendChild(cap); grid.prepend(card);
  });
  await new Promise(res=>{
    const check=()=>{ if(evtSrc && evtSrc.readyState===1) res(); else setTimeout(check,50); }; check();
  });
  hb=setInterval(()=>setStatus('live '+new Date().toLocaleTimeString()), 3000);
  await fetch('/start',{method:'POST'});
}
async function stop(){
  await fetch('/stop',{method:'POST'});
  if(evtSrc){ evtSrc.close(); evtSrc=null; }
  if(hb){ clearInterval(hb); hb=null; }
  setStatus('stopped');
}
async function shutdown(){
  await fetch('/shutdown',{method:'POST'});
  if(evtSrc){ evtSrc.close(); evtSrc=null; }
  if(hb){ clearInterval(hb); hb=null; }
  setStatus('server shutting down');
}
async function llmTest(){
  const body={model:(await (await fetch('/config')).json()).ollama.model, prompt:"<<SYS>>Schneller Test-Prompt<</SYS>>\\n\\nINPUT_JSON:\\n{\\"user_text\\": \\"Ein roter Ballon über einer Stadt im Sonnenuntergang\\", \\"tags\\": [\\"photo\\"], \\"constraints\\":{\\"no_meta\\":true,\\"max_sentences\\":2,\\"avoid_sensitive\\":true}}\\n\\nOUTPUT:\\n", stream:false, options:{}};
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
document.getElementById('start').addEventListener('click', start);
document.getElementById('stop').addEventListener('click', stop);
document.getElementById('shutdown').addEventListener('click', shutdown);
document.getElementById('llmtest').addEventListener('click', llmTest);
document.getElementById('send').addEventListener('click', sendManual);
markCloudNotice();
</script>
</body></html>"""

app.get("/", response_class=HTMLResponse)(lambda: HTMLResponse(INDEX_HTML))

@app.get("/events")
async def events(request: Request):
    async def gen():
        q: asyncio.Queue[str] = asyncio.Queue()
        STATE.listeners.append(q)
        try:
            await q.put(sse_format("status","connected"))
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    yield sse_format("status","hb").encode("utf-8")
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
    masked = (wpath[:3]+"..."+wpath[-10:]) if wpath and len(wpath)>16 else wpath
    return {
        "env_file": ENV_PATH or "(env vars only)",
        "audio": {"device_pref": AUDIO_DEVICE_PREF, "sample_rate": SAMPLE_RATE, "frame_ms": FRAME_MS, "stream_latency_sec": APP_STREAM_LATENCY_SEC},
        "vad": {"disable_vad": DISABLE_VAD, "use_webrtc_vad": USE_WEBRTC_VAD, "aggr": VAD_AGGR, "rms_threshold": RMS_VAD_THRESHOLD},
        "snapshot": {"snapshot_sec": SNAPSHOT_SEC, "min_buf_sec": MIN_BUF_SEC, "max_silence_ms": MAX_SILENCE_MS, "max_segment_sec": MAX_SEGMENT_SEC, "first_snapshot_deadline_sec": FIRST_SNAPSHOT_DEADLINE_SEC},
        "whisper": {"model_path": masked, "language": WHISPER_LANGUAGE, "threads": WHISPER_THREADS, "temperature": WHISPER_TEMPERATURE, "min_sec": WHISPER_MIN_SEC, "min_peak": WHISPER_MIN_PEAK},
        "text": {"min_chars": TEXT_MIN_CHARS, "min_words": TEXT_MIN_WORDS, "force_meaningful": FORCE_MEANINGFUL_CHECK},
        "context": {"max_segments": CONTEXT_MAX_SEGMENTS, "max_chars": CONTEXT_MAX_CHARS},
        "ollama": {"host": OLLAMA_HOST, "port": OLLAMA_PORT, "model": OLLAMA_MODEL, "cloud": is_cloud_model(OLLAMA_MODEL), "temperature": OLLAMA_TEMPERATURE, "timeout_sec": OLLAMA_TIMEOUT_SEC, "interval_sec": LLM_INTERVAL_SEC, "disabled": OLLAMA_DISABLED},
        "comfy": {"host": COMFY_HOST, "port": COMFY_PORT, "disabled": DISABLE_COMFYUI},
        "warmup": {"enable": WARMUP_ENABLE, "grace_sec": WARMUP_GRACE_SEC, "prompt": WARMUP_PROMPT},
        "output_dir": str(OUTPUT_DIR),
    }

@app.post("/start", response_class=PlainTextResponse)
async def start_pipeline():
    if STATE.running:
        return PlainTextResponse("already running", status_code=200)
    STATE.shutting_down = False
    STATE.running = True
    STATE.task = asyncio.create_task(audio_transcription_loop())
    return PlainTextResponse("started")

@app.post("/stop", response_class=PlainTextResponse)
async def stop_pipeline():
    if not STATE.running and not STATE.task and not STATE.bg_tasks:
        await _close_sse_listeners()
        return PlainTextResponse("not running", status_code=200)
    STATE.running = False
    if STATE.task:
        STATE.task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await STATE.task
        STATE.task = None
    for t in list(STATE.bg_tasks): t.cancel()
    with contextlib.suppress(Exception):
        await asyncio.gather(*list(STATE.bg_tasks), return_exceptions=True)
    STATE.bg_tasks.clear()
    await _close_sse_listeners()
    return PlainTextResponse("stopped")

@app.post("/shutdown", response_class=PlainTextResponse)
async def shutdown_server():
    STATE.shutting_down = True
    await stop_pipeline()
    asyncio.create_task(_exit_after_delay())
    return PlainTextResponse("shutting down")

async def _exit_after_delay():
    await asyncio.sleep(0.2)
    os._exit(0)

@app.get("/selftest/audio")
async def selftest_audio():
    try:
        idx = pick_input_device(AUDIO_DEVICE_PREF)
        dev = sd.query_devices()[idx]; sr = SAMPLE_RATE; ch = 1; dur=1.0
        sd.default.device = (idx, None)
        rec = sd.rec(int(dur*sr), samplerate=sr, channels=ch, dtype='float32'); sd.wait()
        x = rec[:,0]; peak=float(np.max(np.abs(x))); rms=float(np.sqrt(np.mean(x*x)))
        return JSONResponse({"device_used_index": idx, "device_used_name": dev.get("name"), "sr": sr, "duration_s": dur, "peak": round(peak,4), "rms": round(rms,4)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/debug/last-prompt")
async def debug_last_prompt():
    return {"last_prompt": STATE.last_prompt, "last_llm_error": STATE.last_llm_error, "model": OLLAMA_MODEL}

@app.get("/health/ollama")
async def health_ollama():
    try:
        async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_short()) as client:
            r = await client.get(_ollama_url("/api/tags")); r.raise_for_status()
            data = r.json()
        return {"ok": True, "models": [m.get("name") for m in data.get("models", [])]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/health/comfyui")
async def health_comfyui():
    try:
        async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_short()) as client:
            ok = await _comfy_available(client)
            return {"ok": ok}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ---- Zusatz: Diagnose-/Bedien-APIs für Ollama und Plan ----
@app.post("/api/ollama/generate")
async def api_ollama_generate(req: OllamaGenerateRequest):
    if await _ollama_available() is False:
        return JSONResponse({"error": "ollama_unavailable"}, status_code=503)
    body = {"model": req.model or OLLAMA_MODEL, "prompt": req.prompt, "stream": bool(req.stream), "options": req.options or {}}
    async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_normal()) as client:
        try:
            data = await _post_with_retries(client, _ollama_url("/api/generate"), body, timeout=float(OLLAMA_TIMEOUT_SEC))
            return {"response": data.get("response","")}
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
            # Ollama chat returns {"message": {"content": "..."}}
            msg = (data.get("message") or {}).get("content","") if isinstance(data, dict) else ""
            return {"response": msg}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/plan")
async def api_plan(req: PlanRequest):
    # Erzeugt einen kompakten Bild-Prompt aus freiem Text
    if await _ollama_available() is False:
        return JSONResponse({"error": "ollama_unavailable"}, status_code=503)
    prompt_text = _build_neutral_prompt_text(req.text, lang=WHISPER_LANGUAGE or "de")
    body = {"model": OLLAMA_MODEL, "prompt": prompt_text, "stream": False, "options": _ollama_options_for_prompt()}
    async with httpx.AsyncClient(limits=_httpx_limits(), timeout=_timeout_normal()) as client:
        try:
            data = await _post_with_retries(client, _ollama_url("/api/generate"), body, timeout=float(OLLAMA_TIMEOUT_SEC))
            out = (data.get("response") or "").strip()
            if out:
                STATE.last_prompt = out
                await broadcast("llm_prompt", out)
            return {"prompt": out}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/ollama/warmup")
async def api_ollama_warmup():
    try:
        await _silent_ollama_warmup()
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

# ---- Main entry ----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=False)
