#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Tuple

import httpx
import numpy as np
import sounddevice as sd
from fastapi import FastAPI, Response, Request
from fastapi.responses import HTMLResponse, FileResponse, PlainTextResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict, field_validator

# === Optional: WebRTC VAD with RMS fallback ===
try:
    import webrtcvad  # type: ignore
    WEBRTCVAD_AVAILABLE = True
except Exception as e:  # noqa: BLE001
    print(f"[WARN] could not import webrtcvad: {e}. Falling back to RMS-VAD.")
    webrtcvad = None  # type: ignore
    WEBRTCVAD_AVAILABLE = False

# === Whisper (pywhispercpp) ===
WHISPER_AVAILABLE = True
try:
    from pywhispercpp.model import Model as WhisperModel
except Exception as e:
    print(f"[WARN] could not import pywhispercpp: {e}. Whisper disabled.")
    WhisperModel = None  # type: ignore
    WHISPER_AVAILABLE = False

# -------------------------------
# Configuration (from .env / environment)
# -------------------------------

def _env_str(key: str, default: str) -> str:
    return os.getenv(key, default).strip()

def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except Exception:
        return default

def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except Exception:
        return default

# Audio
AUDIO_DEVICE_RAW = _env_str("APP_AUDIO_DEVICE", "")
AUDIO_DEVICE_PREF: Optional[str | int] = None
if AUDIO_DEVICE_RAW != "":
    AUDIO_DEVICE_PREF = int(AUDIO_DEVICE_RAW) if AUDIO_DEVICE_RAW.isdigit() else AUDIO_DEVICE_RAW

SAMPLE_RATE_ENV = _env_int("APP_SAMPLE_RATE", 48000)
FRAME_MS = _env_int("APP_FRAME_DURATION_MS", 20)

# VAD
DISABLE_VAD = _env_int("APP_DISABLE_VAD", 1) == 1
VAD_AGGR = _env_int("APP_VAD_AGGRESSIVENESS", 0)
RMS_VAD_THRESHOLD = _env_float("APP_RMS_VAD_THRESHOLD", 0.012)
MAX_SILENCE_MS = _env_int("APP_MAX_SILENCE_MS", 300)

# Snapshot interval for pushing a transcription window to Whisper
SNAPSHOT_SEC = _env_float("APP_SNAPSHOT_SEC", 6.0)

# Whisper
WHISPER_MODEL_PATH = _env_str("APP_WHISPER_MODEL_PATH", "")
WHISPER_LANGUAGE = _env_str("APP_WHISPER_LANGUAGE", "de")
WHISPER_THREADS = _env_int("APP_WHISPER_THREADS", 4)
WHISPER_TEMPERATURE = _env_float("APP_WHISPER_TEMPERATURE", 0.0)

# Ollama
OLLAMA_HOST = _env_str("APP_OLLAMA_HOST", "127.0.0.1")
OLLAMA_PORT = _env_int("APP_OLLAMA_PORT", 11434)
OLLAMA_MODEL = _env_str("APP_OLLAMA_MODEL", "phi3:mini")
OLLAMA_TEMPERATURE = _env_float("APP_OLLAMA_TEMPERATURE", 0.2)

# Latency-friendly defaults
OLLAMA_NUM_CTX = _env_int("APP_OLLAMA_NUM_CTX", 2048)
OLLAMA_NUM_PREDICT = _env_int("APP_OLLAMA_NUM_PREDICT", 320)
OLLAMA_TOP_K = _env_int("APP_OLLAMA_TOP_K", 40)
OLLAMA_TOP_P = _env_float("APP_OLLAMA_TOP_P", 0.9)
OLLAMA_REPEAT_PENALTY = _env_float("APP_OLLAMA_REPEAT_PENALTY", 1.1)
OLLAMA_TIMEOUT_SEC = _env_float("APP_OLLAMA_TIMEOUT_SEC", 60.0)
OLLAMA_MAX_RETRIES = _env_int("APP_OLLAMA_MAX_RETRIES", 3)
OLLAMA_RETRY_BASE_DELAY = _env_float("APP_OLLAMA_RETRY_BASE_DELAY", 0.5)

# ComfyUI
COMFY_HOST = _env_str("APP_COMFY_HOST", "127.0.0.1")
COMFY_PORT = _env_int("APP_COMFY_PORT", 8188)
DISABLE_COMFYUI = _env_int("APP_DISABLE_COMFYUI", 0) == 1  # quick way to disable ComfyUI

# Output directory for generated images
OUTPUT_DIR = Path(_env_str("APP_OUTPUT_DIR", "./outputs/images"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def assert_local(host: str) -> None:
    # Security: strictly require localhost binding for privacy
    if host != "127.0.0.1":
        raise AssertionError(f"Only localhost allowed, got {host}")

assert_local(OLLAMA_HOST)
assert_local(COMFY_HOST)

# -------------------------------
# Pydantic models
# -------------------------------

class OllamaGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    prompt: str
    stream: bool = False
    options: dict = Field(default_factory=dict)

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

# -------------------------------
# Audio helpers
# -------------------------------

def _list_input_devices() -> list[dict]:
    return list(sd.query_devices())

def _prefer_to_index(prefer: Optional[str | int]) -> int:
    """
    Input device selection priority:
    1) exact name match
    2) substring match
    3) device containing 'pulse'
    4) first device with max_input_channels > 0
    """
    devs = _list_input_devices()
    if not devs:
        raise RuntimeError("No audio devices found (sd.query_devices empty).")

    if prefer is not None:
        if isinstance(prefer, int):
            if 0 <= prefer < len(devs) and devs[prefer].get("max_input_channels", 0) > 0:
                return prefer
        else:
            name = prefer.strip().lower()
            for i, d in enumerate(devs):
                if d.get("name", "").lower() == name and d.get("max_input_channels", 0) > 0:
                    return i
            for i, d in enumerate(devs):
                if name in d.get("name", "").lower() and d.get("max_input_channels", 0) > 0:
                    return i

    for i, d in enumerate(devs):
        if "pulse" in d.get("name", "").lower() and d.get("max_input_channels", 0) > 0:
            return i

    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0:
            return i

    raise RuntimeError("No input device with max_input_channels>0 found.")

def pick_input_device_index(prefer: Optional[str | int]) -> tuple[int, dict]:
    idx = _prefer_to_index(prefer)
    dev = sd.query_devices()[idx]
    return idx, dev

def to_int16(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16, copy=False)

def rms_vad(frame: np.ndarray, rms_threshold: float = 0.01) -> bool:
    if frame.size == 0:
        return False
    rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float32), dtype=np.float64)))
    return rms >= rms_threshold

def resample_to_16k(samples: np.ndarray, sr: int) -> np.ndarray:
    # Lightweight linear resampling to 16 kHz (sufficient for Whisper)
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

def seconds_in_buffer(frame_count: int) -> float:
    return (frame_count * FRAME_MS) / 1000.0

# -------------------------------
# Whisper (pywhispercpp)
# -------------------------------

_WHISPER_MODEL: Optional[WhisperModel] = None

def init_whisper_model() -> None:
    # Initialize Whisper model once on startup
    global _WHISPER_MODEL
    if not WHISPER_AVAILABLE:
        print("[WHISPER] pywhispercpp not available.")
        return
    if _WHISPER_MODEL is not None:
        return
    if not WHISPER_MODEL_PATH:
        print("[WHISPER] APP_WHISPER_MODEL_PATH not set – transcription disabled.")
        return
    model_path = Path(WHISPER_MODEL_PATH)
    if not model_path.is_file():
        print(f"[WHISPER] model not found: {model_path} – transcription disabled.")
        return

    try:
        _WHISPER_MODEL = WhisperModel(
            str(model_path),
            n_threads=WHISPER_THREADS,
            print_progress=False,
            print_realtime=False,
            language=WHISPER_LANGUAGE if WHISPER_LANGUAGE else None,
            translate=False,
            temperature=WHISPER_TEMPERATURE,
        )
        print(f"[WHISPER] model loaded: {model_path.name}, threads={WHISPER_THREADS}, lang={WHISPER_LANGUAGE}")
    except Exception as e:
        print(f"[WHISPER] initialization failed: {e}")
        _WHISPER_MODEL = None

def _strip_metadata_like_segments(text: str) -> str:
    # Remove metadata-like bracketed segments (e.g., "[t0=..., text=...]") from certain pywhispercpp builds
    if not text:
        return ""
    import re
    text = re.sub(r"$$.*?$$", " ", text)
    return " ".join(text.split()).strip()

def transcribe_chunk_with_whisper(samples: np.ndarray, sr: int) -> str:
    # Transcribe a given PCM float32 sample array (single channel)
    if not WHISPER_AVAILABLE or _WHISPER_MODEL is None:
        return ""
    if samples.size == 0 or float(np.max(np.abs(samples))) < 0.01:
        return ""
    if sr != 16000:
        samples = resample_to_16k(samples, sr)
        if samples.size == 0:
            return ""
    try:
        if hasattr(_WHISPER_MODEL, "transcribe_float32"):
            txt = _WHISPER_MODEL.transcribe_float32(samples)
        elif hasattr(_WHISPER_MODEL, "transcribe"):
            txt = _WHISPER_MODEL.transcribe(samples)
        else:
            txt = _WHISPER_MODEL.transcribe_pcm16(to_int16(samples))
        if isinstance(txt, dict):
            txt = txt.get("text") or ""
        elif not isinstance(txt, str):
            txt = str(txt)
        return _strip_metadata_like_segments(txt)
    except Exception as e:
        print(f"[WHISPER] transcription failed: {e}")
        return ""

# -------------------------------
# HTTP helpers (robust error logging + retries)
# -------------------------------

async def http_post_json(client: httpx.AsyncClient, url: str, json: dict, timeout: float = 30.0) -> dict:
    try:
        r = await client.post(url, json=json, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as e:
        body = e.response.text if getattr(e, "response", None) is not None else None
        print(f"[HTTP] POST {url} failed: {e} body={body!r}")
        raise

async def http_get_json(client: httpx.AsyncClient, url: str, timeout: float = 15.0) -> dict:
    try:
        r = await client.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as e:
        body = e.response.text if getattr(e, "response", None) is not None else None
        print(f"[HTTP] GET {url} failed: {e} body={body!r}")
        raise

# -------------------------------
# LLM Prompt (Ollama)
# -------------------------------

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
    # Exponential backoff for transient errors
    delay = OLLAMA_RETRY_BASE_DELAY
    last_exc = None
    for attempt in range(1, OLLAMA_MAX_RETRIES + 1):
        try:
            resp = await client.post(url, json=body, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError, httpx.HTTPStatusError) as e:
            last_exc = e
            status = getattr(e, "response", None).status_code if getattr(e, "response", None) else None
            retryable = status in (429, 500, 502, 503) or isinstance(e, (httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError, httpx.ConnectError))
            print(f"[OLLAMA] attempt {attempt} failed (status={status}): {e}")
            if not STATE.running:
                break  # do not retry during shutdown
            if attempt >= OLLAMA_MAX_RETRIES or not retryable:
                break
            await asyncio.sleep(delay)
            delay *= 2.0
    raise RuntimeError(f"Ollama request failed after {OLLAMA_MAX_RETRIES} attempts: {last_exc}")

def _make_ollama_prompt(user_text: str) -> str:
    sys_prompt = (
        "You are an assistant that reformulates a description into a concise, robust prompt for image generation "
        "(subject, environment, style, lighting, color mood, composition). Avoid placeholders. Keep it short."
    )
    return f"<<SYS>>{sys_prompt}<</SYS>>\n\nInput:\n{user_text.strip()}\n\nPrompt:"

async def ollama_generate_prompt(client: httpx.AsyncClient, user_text: str) -> str:
    body = {
        "model": OLLAMA_MODEL,
        "prompt": _make_ollama_prompt(user_text),
        "stream": False,
        "options": _ollama_options_for_prompt(),
    }
    url = _ollama_url("/api/generate")
    data = await _post_with_retries(client, url, body, timeout=OLLAMA_TIMEOUT_SEC)
    return (data.get("response") or "").strip()

# -------------------------------
# ComfyUI (optional image generation)
# -------------------------------

def _comfy_url(path: str) -> str:
    return f"http://{COMFY_HOST}:{COMFY_PORT}{path}"

async def _comfy_available(client: httpx.AsyncClient) -> bool:
    # Simple availability probe via /queue or /history
    try:
        _ = await http_get_json(client, _comfy_url("/queue"))
        return True
    except Exception:
        try:
            _ = await http_get_json(client, _comfy_url("/history"))
            return True
        except Exception:
            return False

def build_comfy_prompt_from_text(text: str) -> ComfyPromptRequest:
    # Placeholder workflow payload. Adjust to your ComfyUI setup.
    prompt_payload = {
        "3": {
            "inputs": {"text": text},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "PromptEncoder"},
        },
    }
    return ComfyPromptRequest(prompt=prompt_payload)

async def comfyui_run_and_wait(
    client: httpx.AsyncClient,
    req: ComfyPromptRequest,
    poll_interval: float = 1.0,
) -> Tuple[str, Optional[str]]:
    resp = await http_post_json(client, _comfy_url("/prompt"), req.model_dump())
    pr = ComfyPromptResponse.model_validate(resp)
    prompt_id = pr.prompt_id
    url_hist = _comfy_url(f"/history/{prompt_id}")
    print(f"[COMFY] prompt id={prompt_id} submitted, waiting for result...")
    for _ in range(600):  # ~10 minutes
        await asyncio.sleep(poll_interval)
        hist = await http_get_json(client, url_hist)
        outputs = hist.get(prompt_id, {}).get("outputs", {})
        for _node_id, node_out in outputs.items():
            imgs = node_out.get("images") or []
            if imgs:
                img = imgs[0]
                filename = img.get("filename")
                subfolder = img.get("subfolder") or ""
                rel_path = f"{subfolder}/{filename}" if subfolder else filename
                return prompt_id, rel_path
    return prompt_id, None

# -------------------------------
# Pipeline state
# -------------------------------

@dataclass
class PipelineState:
    running: bool = False
    task: Optional[asyncio.Task] = None
    listeners: List[asyncio.Queue] = field(default_factory=list)
    transcript_buffer: List[bytes] = field(default_factory=list)  # legacy; kept for status ticks
    actual_sr: int = 16000
    device_used_index: Optional[int] = None
    device_used_name: Optional[str] = None
    last_prompt: Optional[str] = None
    last_llm_error: Optional[str] = None
    last_llm_run_ts: float = 0.0  # debounce LLM calls

STATE = PipelineState()

# -------------------------------
# SSE events
# -------------------------------

def sse_format(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"

async def broadcast(event: str, data: str) -> None:
    for q in list(STATE.listeners):
        try:
            await q.put(sse_format(event, data))
        except Exception:
            pass

async def sse_stream() -> AsyncGenerator[bytes, None]:
    q: asyncio.Queue[str] = asyncio.Queue()
    STATE.listeners.append(q)
    try:
        await q.put(sse_format("status", "connected"))
        while True:
            msg = await q.get()
            yield msg.encode("utf-8")
    except asyncio.CancelledError:
        pass
    finally:
        if q in STATE.listeners:
            STATE.listeners.remove(q)

# -------------------------------
# Audio/transcription loop with segmented endpointing
# -------------------------------

def _device_name_override_for_alsa(idx: int, name: str) -> Optional[str]:
    # On some Linux/ALSA combos it's more reliable to use explicit hw:* identifier
    if "ALC293" in name and "(hw:0,0)" in name:
        return "hw:0,0"
    return None

def _open_input_stream(desired_sr: int, frame_ms: int, prefer: Optional[str | int]):
    # Try preferred device and a list of common sample rates
    dev_idx, dev_info = pick_input_device_index(prefer)
    dev_name = dev_info.get("name", f"idx:{dev_idx}")
    dev_param: int | str = dev_idx
    alsa_name = _device_name_override_for_alsa(dev_idx, dev_name)
    if alsa_name:
        dev_param = alsa_name

    for sr in [desired_sr, 48000, 44100, 32000, 16000]:
        frame_samples = int(sr * frame_ms / 1000)
        try:
            stream = sd.InputStream(
                samplerate=sr,
                channels=1,
                dtype="float32",
                device=dev_param,
                blocksize=frame_samples,
                callback=None,
                latency="low",
            )
            print(f"[AUDIO] open OK: device={dev_param!r}, name={dev_name!r}, samplerate={sr}, blocksize={frame_samples}")
            return stream, sr, frame_samples, dev_idx, dev_name, dev_param
        except Exception as e:
            print(f"[AUDIO] open FAIL: device={dev_param!r}, name={dev_name!r}, samplerate={sr}: {e}")
            continue
    raise RuntimeError(f"No supported samplerate for device={dev_param!r} ({dev_name!r}).")

async def audio_transcription_loop() -> None:
    # Open input stream
    try:
        stream, actual_sr, frame_samples, dev_idx, dev_name, dev_param = _open_input_stream(
            SAMPLE_RATE_ENV, FRAME_MS, AUDIO_DEVICE_PREF
        )
    except Exception as e:
        await broadcast("status", f"audio_open_failed: {e}")
        print(f"[AUDIO] failed to open input stream: {e}")
        return

    STATE.actual_sr = actual_sr
    STATE.device_used_index = dev_idx
    STATE.device_used_name = dev_name
    # If WebRTC VAD is available and VAD is enabled, you can wire it here; we use RMS as default
    vad = webrtcvad.Vad(VAD_AGGR) if (WEBRTCVAD_AVAILABLE and not DISABLE_VAD) else None

    audio_frames: List[np.ndarray] = []
    last_snapshot = time.time()
    last_tick = time.time()
    total_frames = 0

    # New: segmented endpointing buffers
    current_segment_frames: List[np.ndarray] = []
    silence_ms = 0
    endpoint_silence_ms = 400   # segment ends after this much silence
    max_segment_sec = 10.0      # hard cap per segment
    max_segment_frames = int((max_segment_sec * 1000) / FRAME_MS)

    def sd_callback(indata, frames, timeinfo, status):
        # Collect frames from sounddevice callback
        mono = indata[:, 0].copy()
        audio_frames.append(mono)

    stream.callback = sd_callback

    try:
        sd.default.device = (dev_param, None)
        sd.default.samplerate = actual_sr
        sd.default.channels = 1
    except Exception as e:
        print(f"[AUDIO] could not set sd.default.*: {e}")

    print(f"[CFG] device_pref={AUDIO_DEVICE_PREF!r}, device_used idx:{dev_idx} name:{dev_name!r}, param={dev_param!r}, desired_sr={SAMPLE_RATE_ENV}, actual_sr={actual_sr}, frame_ms={FRAME_MS}, snapshot_sec={SNAPSHOT_SEC}, disable_vad={DISABLE_VAD}")

    await broadcast("status", f"recording_start sr={actual_sr}")
    await broadcast("status", f"device_used idx={dev_idx} name={dev_name}")
    await broadcast("status", "tick warmup")

    stream.start()

    # Give callback a short priming period
    try:
        priming_deadline = time.time() + 0.6
        while time.time() < priming_deadline and not audio_frames:
            await asyncio.sleep(FRAME_MS / 1000.0)
    except asyncio.CancelledError:
        stream.stop(); stream.close()
        await broadcast("status", "recording_stop")
        return

    # Fallback to polling if callback did not produce frames
    use_polling = False
    if not audio_frames:
        use_polling = True
        print("[AUDIO] callback produced no frames; switching to polling read() loop")

    try:
        while STATE.running:
            await asyncio.sleep(FRAME_MS / 1000.0)

            now = time.time()
            if now - last_tick >= 1.0:
                last_tick = now
                # Legacy: transcript_buffer is no longer used for content; keep a short indicator
                buf_sec = seconds_in_buffer(len(STATE.transcript_buffer))
                await broadcast("status", f"tick frames={total_frames} seg_frames={len(current_segment_frames)} (~{buf_sec:.1f}s) sr={actual_sr}")

            # Read frame if in polling mode
            if use_polling:
                try:
                    data, ov = stream.read(frame_samples)
                    frame = data[:, 0].copy()
                    audio_frames.append(frame)
                except Exception as e:
                    await broadcast("status", f"audio_read_error: {e}")
                    continue

            if not audio_frames:
                continue

            # Normalize frame length and clamp range
            frame = audio_frames.pop(0)
            total_frames += 1
            if len(frame) != frame_samples:
                if len(frame) < frame_samples:
                    frame = np.pad(frame, (0, frame_samples - len(frame)))
                else:
                    frame = frame[:frame_samples]
            frame = np.clip(frame, -1.0, 1.0)

            # Endpointing: simple RMS-based speech gate per frame
            if DISABLE_VAD:
                is_speech = True
            else:
                is_speech = rms_vad(frame, rms_threshold=RMS_VAD_THRESHOLD)

            current_segment_frames.append(frame)
            if len(current_segment_frames) > max_segment_frames:
                is_segment_end = True
            else:
                if is_speech:
                    silence_ms = 0
                else:
                    silence_ms += FRAME_MS
                is_segment_end = (silence_ms >= endpoint_silence_ms)

            # Periodic snapshot OR segment end → run Whisper
            do_snapshot = (now - last_snapshot >= SNAPSHOT_SEC)
            if do_snapshot or is_segment_end:
                last_snapshot = now
                seg = np.frombuffer(b"".join([f.tobytes() for f in current_segment_frames]), dtype=np.float32)
                if seg.size > 0 and float(np.max(np.abs(seg))) >= 0.01:
                    txt = transcribe_chunk_with_whisper(seg, actual_sr)
                    if txt:
                        print(f"[WHISPER] text: {txt}")
                        await broadcast("transcript", txt)
                        # LLM debounce: minimum gap between runs
                        if time.time() - STATE.last_llm_run_ts >= 3.0:
                            STATE.last_llm_run_ts = time.time()
                            asyncio.create_task(run_llm_and_optionally_image(txt))
                    else:
                        await broadcast("status", "whisper_empty(no_text)")
                # Reset on segment end; also reset silence counter
                if is_segment_end:
                    current_segment_frames.clear()
                    silence_ms = 0

    except asyncio.CancelledError:
        pass
    except Exception as e:
        await broadcast("status", f"error_audio: {e}")
        raise
    finally:
        with contextlib.suppress(Exception):
            stream.stop()
            stream.close()
        await broadcast("status", "recording_stop")

# -------------------------------
# LLM + optional image generation
# -------------------------------

async def run_llm_and_optionally_image(text: str) -> None:
    async with httpx.AsyncClient() as client:
        # Health probe for Ollama before first generate (helps with clear status if server is down)
        try:
            _ = await http_get_json(client, f"http://127.0.0.1:{OLLAMA_PORT}/api/tags")
        except Exception as e:
            await broadcast("status", f"ollama_unavailable: {e}")
            return

        try:
            img_prompt = await ollama_generate_prompt(client, text)
            if img_prompt:
                STATE.last_prompt = img_prompt
                print(f"[LLM] prompt: {img_prompt}")
                await broadcast("llm_prompt", img_prompt)
            else:
                STATE.last_llm_error = "llm_empty"
                await broadcast("status", "llm_empty")
                return

            await broadcast("status", "llm_ok")

            # Optional ComfyUI integration (can be disabled via APP_DISABLE_COMFYUI=1)
            if DISABLE_COMFYUI:
                await broadcast("status", "comfy_disabled")
                return

            try:
                if not await _comfy_available(client):
                    await broadcast("status", "comfy_unavailable")
                    return
            except Exception as e:
                await broadcast("status", f"comfy_unavailable: {e}")
                return

            try:
                req = build_comfy_prompt_from_text(img_prompt)
                pid, rel_img_path = await comfyui_run_and_wait(client, req)
                if rel_img_path:
                    await broadcast("image", rel_img_path)
                else:
                    await broadcast("status", "comfy_timeout")
            except Exception as e:
                await broadcast("status", f"comfy_error: {e}")
        except Exception as e:
            STATE.last_llm_error = f"pipeline_error: {e}"
            print("[LLM] pipeline_error:", e)
            await broadcast("status", f"pipeline_error: {e}")

# -------------------------------
# FastAPI app
# -------------------------------

app = FastAPI()

@app.on_event("startup")
async def _startup_init_models() -> None:
    init_whisper_model()

# Serve generated images under /static (local only)
app.mount("/static", StaticFiles(directory=str(OUTPUT_DIR), html=False), name="static")

INDEX_HTML = """<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <title>Vorlesen → Bilder (Lokal)</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body { font-family: system-ui, sans-serif; margin: 0; padding: 1rem; background: #0b1020; color: #e8ecf1; }
    header { display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; margin-bottom: 0.75rem; }
    button { background: #1f6feb; color: white; border: 0; padding: 0.6rem 1rem; border-radius: 8px; cursor: pointer; }
    button.stop { background: #c53b3b; }
    #status { opacity: 0.9; font-size: 0.9rem; }

    #live { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 0.75rem; }
    .panel { background: #141a2e; border-radius: 10px; overflow: hidden; box-shadow: 0 0 0 1px rgba(255,255,255,0.06) inset; }
    .panel-title { font-size: 0.9rem; font-weight: 600; padding: 0.5rem 0.75rem; color: #c7d1df; border-bottom: 1px solid rgba(255,255,255,0.06); }
    .panel-body { padding: 0.6rem 0.75rem; min-height: 52px; white-space: pre-wrap; word-break: break-word; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }

    #grid { margin-top: 0.5rem; display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }
    .card { background: #141a2e; border-radius: 10px; overflow: hidden; box-shadow: 0 0 0 1px rgba(255,255,255,0.06) inset; }
    .card img { width: 100%; display: block; }
    .cap { padding: 0.5rem 0.75rem; font-size: 0.85rem; color: #c7d1df; }
  </style>
</head>
<body>
  <header>
    <button id="start">Start</button>
    <button id="stop" class="stop">Stop</button>
    <div id="status">Ready.</div>
  </header>

  <main>
    <section id="live">
      <div class="panel">
        <div class="panel-title">Transcript</div>
        <div id="transcript" class="panel-body mono"></div>
      </div>
      <div class="panel">
        <div class="panel-title">Prompt</div>
        <div id="prompt" class="panel-body"></div>
      </div>
    </section>

    <h4>Images</h4>
    <div id="grid"></div>
  </main>

  <script>
    const statusEl = document.getElementById('status');
    const promptEl = document.getElementById('prompt');
    const transcriptEl = document.getElementById('transcript');
    const grid = document.getElementById('grid');
    let evtSrc = null;

    function setStatus(msg) { statusEl.textContent = msg; }
    function setPrompt(p) { promptEl.textContent = p || ''; }
    function setTranscript(t) { transcriptEl.textContent = t || ''; }

    async function start() {
      if (evtSrc) evtSrc.close();
      evtSrc = new EventSource('/events');

      evtSrc.addEventListener('status', e => setStatus(e.data));
      evtSrc.addEventListener('transcript', e => setTranscript(e.data));
      evtSrc.addEventListener('llm_prompt', e => setPrompt(e.data));

      evtSrc.addEventListener('image', e => {
        const rel = e.data;
        const src = '/static/' + rel;
        const card = document.createElement('div');
        card.className = 'card';
        const img = document.createElement('img');
        img.src = src + '?t=' + Date.now();
        const cap = document.createElement('div'); cap.className = 'cap';
        cap.textContent = new Date().toLocaleTimeString();
        card.appendChild(img); card.appendChild(cap);
        grid.prepend(card);
      });

      await new Promise(res => {
        const check = () => {
          if (evtSrc && evtSrc.readyState === 1) res();
          else setTimeout(check, 50);
        };
        check();
      });

      await fetch('/start', { method:'POST' });
    }

    async function stop() {
      await fetch('/stop', { method:'POST' });
      if (evtSrc) { evtSrc.close(); evtSrc = null; }
      setStatus('stopped');
    }

    document.getElementById('start').addEventListener('click', start);
    document.getElementById('stop').addEventListener('click', stop);
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(INDEX_HTML)

@app.post("/start", response_class=PlainTextResponse)
async def start_pipeline():
    if STATE.running:
        return PlainTextResponse("already running", status_code=200)
    STATE.transcript_buffer.clear()
    STATE.running = True
    STATE.task = asyncio.create_task(audio_transcription_loop())
    return PlainTextResponse("started")

@app.post("/stop", response_class=PlainTextResponse)
async def stop_pipeline():
    if not STATE.running:
        return PlainTextResponse("not running", status_code=200)
    STATE.running = False
    if STATE.task:
        STATE.task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await STATE.task
        STATE.task = None
    return PlainTextResponse("stopped")

@app.get("/events")
async def events(request: Request):
    async def gen():
        async for chunk in sse_stream():
            yield chunk
    return StreamingResponse(gen(), media_type="text/event-stream")

@app.get("/images/{path:path}")
async def get_image(path: str):
    fp = OUTPUT_DIR / path
    if fp.is_file():
        return FileResponse(str(fp))
    return Response(status_code=404, content="Not found")

# Device inspection
@app.get("/devices")
async def list_devices():
    try:
        devs = sd.query_devices()
        hostapis = sd.query_hostapis()
        out = []
        for i, d in enumerate(devs):
            out.append({
                "index": i,
                "name": d.get("name"),
                "max_input_channels": d.get("max_input_channels"),
                "max_output_channels": d.get("max_output_channels"),
                "default_samplerate": d.get("default_samplerate"),
                "hostapi_index": d.get("hostapi"),
                "hostapi_name": hostapis[d.get("hostapi")]["name"] if d.get("hostapi") is not None else None,
            })
        return JSONResponse({"devices": out})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Audio self-test
@app.get("/selftest/audio")
async def selftest_audio():
    try:
        idx, dev = pick_input_device_index(AUDIO_DEVICE_PREF)
        sr = SAMPLE_RATE_ENV
        ch = 1
        dur = 3.0
        dev_param: int | str = idx
        alsa_name = "hw:0,0" if "(hw:0,0)" in (dev.get("name") or "") else None
        if alsa_name:
            dev_param = alsa_name
        sd.default.device = (dev_param, None)
        rec = sd.rec(int(dur*sr), samplerate=sr, channels=ch, dtype='float32')
        sd.wait()
        x = rec[:,0]
        peak = float(np.max(np.abs(x)))
        rms = float(np.sqrt(np.mean(x*x)))
        return JSONResponse({"device_used_index": idx, "device_used_name": dev.get("name"), "sr": sr, "duration_s": dur, "peak": round(peak,4), "rms": round(rms,4)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Debug + Health
@app.get("/debug/last-prompt")
async def debug_last_prompt():
    return {"last_prompt": STATE.last_prompt, "last_llm_error": STATE.last_llm_error, "model": OLLAMA_MODEL}

@app.get("/health/ollama")
async def health_ollama():
    try:
        async with httpx.AsyncClient() as client:
            data = await http_get_json(client, f"http://127.0.0.1:11434/api/tags")
        return {"ok": True, "models": [m.get("name") for m in data.get("models", [])]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/health/comfyui")
async def health_comfyui():
    try:
        async with httpx.AsyncClient() as client:
            ok = await _comfy_available(client)
            return {"ok": ok}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# Warmup Ollama (loads model to RAM)
@app.post("/warmup/ollama")
async def warmup_ollama():
    try:
        async with httpx.AsyncClient() as client:
            body = {
                "model": OLLAMA_MODEL,
                "prompt": "Say hello.",
                "stream": False,
                "options": {
                    "temperature": OLLAMA_TEMPERATURE,
                    "num_ctx": max(512, min(OLLAMA_NUM_CTX, 2048)),
                    "num_predict": 8,
                },
            }
            data = await _post_with_retries(client, _ollama_url("/api/generate"), body, timeout=min(OLLAMA_TIMEOUT_SEC, 30.0))
            return {"ok": True, "response": (data.get("response") or "").strip()}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=502)

# Optional: clean SIGINT/SIGTERM handling to stop background tasks
def _graceful_shutdown(*_):
    if STATE.running:
        STATE.running = False
    if STATE.task:
        try:
            STATE.task.cancel()
        except Exception:
            pass

try:
    signal.signal(signal.SIGINT, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)
except Exception:
    pass
