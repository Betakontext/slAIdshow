# transcription_backend.py
#
# Drop-in replacement with:
# - Preserves existing public API, env keys, status structure, capability logic
# - Primary backend: faster-whisper (CTranslate2), fallback: pywhispercpp.model.Model
# - Keeps your established audio preparation and cleaning utilities
# - Adds a low-latency RealtimeTranscriber:
#     * RMS-based VAD + debounce to detect end-of-utterance
#     * Deterministic decode ticker for partials (low jitter)
#     * Tail-window decodes for stability (avoid reinterpreting old text)
#     * Partial dedupe + growth + meaningful-light + punct-only skip
#     * Adaptive finalize delay (sentence-end vs no-sentence-end)
#     * Optional file recording and file-transcription fallback if realtime-final is empty
# - All operational messages remain in English

from __future__ import annotations

import asyncio
import contextlib
import io
import os
import re
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np


# -----------------------
# --- Realtime merging and sentence heuristics utilities ---
# -----------------------

SENTENCE_END_RE = re.compile(r"[\.!\?…]\s*$")


def _merge_overlap(prev: str, new: str, min_overlap: int = 5, max_overlap: int = 18) -> str:
    """
    Merge 'new' into 'prev' with de-duplication by overlapping tail/head.
    Returns the combined string.
    """
    prev = prev or ""
    new = new or ""
    if not prev:
        return new
    # Normalize spaces
    prev_n = re.sub(r"\s+", " ", prev).strip()
    new_n = re.sub(r"\s+", " ", new).strip()
    if not new_n:
        return prev_n
    # Try largest overlap between end(prev) and start(new)
    max_k = min(max_overlap, len(prev_n), len(new_n))
    for k in range(max_k, min_overlap - 1, -1):
        if prev_n.endswith(new_n[:k]):
            return (prev_n + new_n[k:]).strip()
    # Try the reverse (rare but cheap)
    if new_n.endswith(prev_n[:max_k]):
        return (prev_n + " " + new_n).strip()
    return (prev_n + " " + new_n).strip()


def _is_meaningful(text: str, min_chars: int = 12, min_words: int = 3) -> bool:
    text_n = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text_n) < min_chars:
        return False
    if len(text_n.split()) < min_words:
        return False
    return True


def _is_punct_only_change(prev_norm: str, new_norm: str) -> bool:
    """
    Return True if the only difference between prev and new is a terminal punctuation change.
    This reduces UI "flicker" when Whisper only toggles a trailing '.' or '…' etc.
    """
    if prev_norm == new_norm:
        return False
    prev_core = re.sub(r"[\.!\?…]+\s*$", "", prev_norm)
    new_core = re.sub(r"[\.!\?…]+\s*$", "", new_norm)
    return prev_core == new_core


# -----------------------
# Environment helpers
# -----------------------

def _env_str(key: str, default: str = "") -> str:
    return (os.getenv(key, default) or "").strip()


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


def _env_bool(key: str, default: bool = False) -> bool:
    default_text = "1" if default else "0"
    value = (os.getenv(key, default_text) or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


# -----------------------
# Regex helpers
# -----------------------

TEXT_FIELD_RE = re.compile(r"text\s*=\s*(.+?)(?:,|$)")
META_RE = re.compile(
    r"\b(musik|music|applaus|applause|lachen|laugh|geräusch|noise|"
    r"husten|cough|klatschen|klingel|ring|summen|hmm+|pause)\b",
    re.IGNORECASE,
)


# -----------------------
# Configuration dataclass
# -----------------------

@dataclass(frozen=True)
class TranscriptionConfig:
    """Configuration loaded once from environment variables."""

    requested_backend: str
    language: str
    threads: int
    temperature: float
    min_seconds: float
    min_peak: float

    # Optional Whisper-level thresholds (used by faster-whisper)
    no_speech_threshold: Optional[float]
    logprob_threshold: Optional[float]
    condition_on_previous_text: bool

    pywhispercpp_model_path: str

    faster_whisper_model: str
    faster_whisper_device: str
    faster_whisper_compute_type: str
    faster_whisper_cpu_threads: int
    faster_whisper_num_workers: int
    faster_whisper_beam_size: int
    faster_whisper_vad_filter: bool
    faster_whisper_download_root: Optional[str]

    # Realtime parameters (new; defaults tuned for low latency)
    rt_slice_sec: float
    rt_window_sec: float
    rt_end_debounce_ms: int
    vad_enabled: bool
    vad_rms_threshold: float
    rt_fallback_record: bool
    rt_fallback_dir: str

    @classmethod
    def from_environment(cls) -> "TranscriptionConfig":
        requested_backend = _env_str("APP_WHISPER_BACKEND", "auto").lower()
        # Preserve legacy names
        if requested_backend not in {"auto", "faster_whisper", "pywhispercpp"}:
            print(
                "[TRANSCRIBE] Invalid APP_WHISPER_BACKEND="
                f"{requested_backend!r}; using 'auto'."
            )
            requested_backend = "auto"

        faster_device = _env_str("APP_FASTER_WHISPER_DEVICE", "auto").lower()
        if faster_device not in {"auto", "cuda", "cpu"}:
            print(
                "[TRANSCRIBE] Invalid APP_FASTER_WHISPER_DEVICE="
                f"{faster_device!r}; using 'auto'."
            )
            faster_device = "auto"

        faster_compute_type = _env_str("APP_FASTER_WHISPER_COMPUTE_TYPE", "auto").lower()

        download_root_raw = _env_str("APP_FASTER_WHISPER_DOWNLOAD_ROOT", "")
        download_root = str(Path(download_root_raw).resolve()) if download_root_raw else None

        # Optional decode controls; None means "do not pass" to keep library defaults
        no_speech_threshold_env = _env_str("APP_WHISPER_NO_SPEECH_THRESHOLD", "")
        no_speech_threshold: Optional[float] = None
        if no_speech_threshold_env:
            try:
                no_speech_threshold = float(no_speech_threshold_env)
            except Exception:
                print("[TRANSCRIBE] Ignoring invalid APP_WHISPER_NO_SPEECH_THRESHOLD")

        logprob_threshold_env = _env_str("APP_WHISPER_LOGPROB_THRESHOLD", "")
        logprob_threshold: Optional[float] = None
        if logprob_threshold_env:
            try:
                logprob_threshold = float(logprob_threshold_env)
            except Exception:
                print("[TRANSCRIBE] Ignoring invalid APP_WHISPER_LOGPROB_THRESHOLD")

        # Realtime params with sensible defaults
        rt_slice_sec = _env_float("APP_RT_SLICE_SEC", 3.0)
        rt_window_sec = _env_float("APP_RT_WINDOW_SEC", 30.0)
        rt_end_debounce_ms = _env_int("APP_RT_END_DEBOUNCE_MS", 550)
        vad_enabled = not _env_bool("APP_DISABLE_VAD", True)  # default True (enabled), inverted flag
        vad_rms_threshold = _env_float("APP_RMS_VAD_THRESHOLD", 0.015)
        rt_fallback_record = _env_bool("APP_RT_FALLBACK_RECORD", True)
        rt_fallback_dir = _env_str("APP_RT_FALLBACK_DIR", "./outputs/voice_fallback")

        return cls(
            requested_backend=requested_backend,
            language=(_env_str("APP_WHISPER_LANGUAGE", "").strip() or None),
            threads=max(1, _env_int("APP_WHISPER_THREADS", 2)),
            temperature=max(0.0, _env_float("APP_WHISPER_TEMPERATURE", 0.0)),
            min_seconds=max(0.0, _env_float("APP_WHISPER_MIN_SEC", 0.35)),
            min_peak=max(0.0, _env_float("APP_WHISPER_MIN_PEAK", 0.0009)),
            no_speech_threshold=no_speech_threshold,
            logprob_threshold=logprob_threshold,
            condition_on_previous_text=_env_bool("APP_WHISPER_CONDITION_ON_PREVIOUS_TEXT", False),
            pywhispercpp_model_path=_env_str("APP_WHISPER_MODEL_PATH", ""),
            faster_whisper_model=_env_str("APP_FASTER_WHISPER_MODEL", "small"),
            faster_whisper_device=faster_device,
            faster_whisper_compute_type=faster_compute_type,
            faster_whisper_cpu_threads=max(
                1,
                _env_int(
                    "APP_FASTER_WHISPER_CPU_THREADS",
                    _env_int("APP_WHISPER_THREADS", 2),
                ),
            ),
            faster_whisper_num_workers=max(
                1,
                _env_int("APP_FASTER_WHISPER_NUM_WORKERS", 1),
            ),
            faster_whisper_beam_size=max(
                1,
                _env_int("APP_FASTER_WHISPER_BEAM_SIZE", 1),
            ),
            faster_whisper_vad_filter=_env_bool(
                "APP_FASTER_WHISPER_VAD_FILTER",
                False,
            ),
            faster_whisper_download_root=download_root,
            # Realtime params
            rt_slice_sec=rt_slice_sec,
            rt_window_sec=rt_window_sec,
            rt_end_debounce_ms=rt_end_debounce_ms,
            vad_enabled=vad_enabled,
            vad_rms_threshold=vad_rms_threshold,
            rt_fallback_record=rt_fallback_record,
            rt_fallback_dir=rt_fallback_dir,
        )


# -----------------------
# Status dataclass
# -----------------------

@dataclass
class TranscriptionStatus:
    """Runtime-visible state for /config and operational troubleshooting."""

    requested_backend: str
    selected_backend: str = "uninitialized"
    selected_device: Optional[str] = None
    selected_compute_type: Optional[str] = None
    model: Optional[str] = None
    language: Optional[str] = None
    ready: bool = False
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    initialization_error: Optional[str] = None
    supported_cuda_compute_types: List[str] = None
    supported_cpu_compute_types: List[str] = None
    initialized_at_unix: Optional[float] = None
    last_transcription_ms: Optional[float] = None
    last_transcription_error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.supported_cuda_compute_types is None:
            self.supported_cuda_compute_types = []
        if self.supported_cpu_compute_types is None:
            self.supported_cpu_compute_types = []

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------
# Backend base
# -----------------------

class BaseTranscriptionBackend:
    """Common synchronous interface used by app.py through asyncio.to_thread."""

    backend_name = "base"

    def __init__(self, config: TranscriptionConfig) -> None:
        self.config = config

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        raise NotImplementedError


# -----------------------
# Faster-Whisper backend
# -----------------------

class FasterWhisperBackend(BaseTranscriptionBackend):
    """CTranslate2-backed faster-whisper implementation."""

    backend_name = "faster_whisper"

    def __init__(
        self,
        config: TranscriptionConfig,
        device: str,
        compute_type: str,
    ) -> None:
        super().__init__(config)

        from faster_whisper import WhisperModel  # type: ignore

        model_kwargs: Dict[str, Any] = {
            "device": device,
            "compute_type": compute_type,
            "cpu_threads": config.faster_whisper_cpu_threads,
            "num_workers": config.faster_whisper_num_workers,
        }

        if config.faster_whisper_download_root:
            Path(config.faster_whisper_download_root).mkdir(
                parents=True,
                exist_ok=True,
            )
            model_kwargs["download_root"] = config.faster_whisper_download_root

        self.device = device
        self.compute_type = compute_type
        self.model_name = config.faster_whisper_model
        self.model = WhisperModel(config.faster_whisper_model, **model_kwargs)

        print(
            "[TRANSCRIBE] Faster-Whisper init: "
            f"device={self.device} compute_type={self.compute_type} "
            f"model={self.model_name} cpu_threads={config.faster_whisper_cpu_threads} "
            f"workers={config.faster_whisper_num_workers} "
            f"beam_size={config.faster_whisper_beam_size} "
            f"vad_filter={int(config.faster_whisper_vad_filter)} "
            f"lang={config.language} temp={config.temperature:.2f} "
            f"min_sec={config.min_seconds:.2f} min_peak={config.min_peak:.4f} "
            f"no_speech_thr={config.no_speech_threshold} "
            f"logprob_thr={config.logprob_threshold} "
            f"cond_prev={int(config.condition_on_previous_text)}"
        )

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        audio = _prepare_audio(
            samples=samples,
            sample_rate=sample_rate,
            min_seconds=self.config.min_seconds,
        )
        if audio.size == 0:
            return ""

        # Determine effective language at runtime:
        # Priority: (no explicit per-call override here) config.language > APP_INPUT_LOCALE_HINT > None
        hint_lang = (os.getenv("APP_INPUT_LOCALE_HINT", "") or "").strip().lower() or None
        conf_lang = self.config.language or None
        eff_lang = conf_lang or hint_lang or None

        # Build kwargs conditionally to avoid overriding library defaults with None
        kwargs: Dict[str, Any] = dict(
            language=eff_lang,  # None enables auto-detect
            beam_size=self.config.faster_whisper_beam_size,
            temperature=self.config.temperature,
            vad_filter=self.config.faster_whisper_vad_filter,
            condition_on_previous_text=self.config.condition_on_previous_text,
            word_timestamps=False,
            without_timestamps=True,
        )

        if self.config.no_speech_threshold is not None:
            kwargs["no_speech_threshold"] = self.config.no_speech_threshold
        if self.config.logprob_threshold is not None:
            kwargs["log_prob_threshold"] = self.config.logprob_threshold

        segments, _info = self.model.transcribe(audio, **kwargs)

        text_parts: List[str] = []
        for segment in segments:
            text = getattr(segment, "text", "")
            if isinstance(text, str) and text.strip():
                text_parts.append(text.strip())

        return clean_transcript(" ".join(text_parts))


# -----------------------
# pywhispercpp backend
# -----------------------

class PyWhisperCppBackend(BaseTranscriptionBackend):
    """whisper.cpp-based fallback implementation using pywhispercpp.model.Model."""

    backend_name = "pywhispercpp"

    def __init__(self, config: TranscriptionConfig) -> None:
        super().__init__(config)

        model_path = Path(config.pywhispercpp_model_path)
        if not config.pywhispercpp_model_path or not model_path.is_file():
            raise FileNotFoundError(
                "pywhispercpp model is unavailable. Set APP_WHISPER_MODEL_PATH "
                "to an existing ggml model file."
            )

        from pywhispercpp.model import Model as WhisperModel  # type: ignore

        self.model_path = model_path.resolve()
        self.model = WhisperModel(
            str(self.model_path),
            n_threads=config.threads,
            print_progress=False,
            print_realtime=False,
            # Force auto-detect to avoid re-init on runtime hint changes:
            language=None,
            translate=False,
            temperature=config.temperature,
        )

        print(
            "[TRANSCRIBE] pywhispercpp init: "
            f"model={self.model_path.name} threads={config.threads} "
            f"lang={self.config.language} temp={self.config.temperature:.2f} "
            f"min_sec={self.config.min_seconds:.2f} min_peak={self.config.min_peak:.4f}"
        )

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        audio = _prepare_audio(
            samples=samples,
            sample_rate=sample_rate,
            min_seconds=self.config.min_seconds,
        )
        if audio.size == 0:
            return ""

        # Prefer float32 path if available
        if hasattr(self.model, "transcribe_float32"):
            raw = self.model.transcribe_float32(audio)
        elif hasattr(self.model, "transcribe"):
            raw = self.model.transcribe(audio)
        else:
            raw = self.model.transcribe_pcm16(_to_int16(audio))

        return clean_transcript(_parse_pywhispercpp_output(raw))


# -----------------------
# Manager and initialization
# -----------------------

class TranscriptionManager:
    """
    Owns one selected backend for the process lifetime.

    The manager deliberately does not switch implementations during an active
    recording session. A stable backend avoids model reloading, inconsistent
    latency, and surprise CUDA/VRAM contention while ComfyUI is generating.
    """

    def __init__(self) -> None:
        self.config = TranscriptionConfig.from_environment()
        self.backend: Optional[BaseTranscriptionBackend] = None
        self.status = TranscriptionStatus(
            requested_backend=self.config.requested_backend,
            language=self.config.language,
        )

    def initialize(self) -> None:
        """Initialize the best permitted backend according to configuration."""

        if self.status.ready:
            return

        self.config = TranscriptionConfig.from_environment()
        self.status = TranscriptionStatus(
            requested_backend=self.config.requested_backend,
            language=self.config.language,
        )
        attempts: List[str] = []

        if self.config.requested_backend == "pywhispercpp":
            self._initialize_pywhispercpp_or_fail(attempts)
            return

        if self.config.requested_backend == "faster_whisper":
            self._initialize_explicit_faster_whisper_or_fail(attempts)
            return

        self._initialize_auto(attempts)

    def _initialize_auto(self, attempts: List[str]) -> None:
        """Try faster-whisper CUDA, then CPU, then pywhispercpp."""

        faster_error: Optional[str] = None

        try:
            if self.config.faster_whisper_device in {"auto", "cuda"}:
                self._initialize_faster_whisper_cuda()
                return
        except Exception as exc:
            faster_error = f"CUDA faster-whisper unavailable: {exc}"
            attempts.append(faster_error)
            print(f"[TRANSCRIBE] {faster_error}")

        try:
            if self.config.faster_whisper_device in {"auto", "cpu"}:
                self._initialize_faster_whisper_cpu()
                if faster_error:
                    self.status.fallback_used = True
                    self.status.fallback_reason = faster_error
                return
        except Exception as exc:
            cpu_error = f"CPU faster-whisper unavailable: {exc}"
            attempts.append(cpu_error)
            print(f"[TRANSCRIBE] {cpu_error}")

        try:
            self._initialize_pywhispercpp()
            self.status.fallback_used = True
            self.status.fallback_reason = (
                "; ".join(attempts)
                if attempts
                else "faster-whisper could not be initialized"
            )
            return
        except Exception as exc:
            fallback_error = f"pywhispercpp fallback unavailable: {exc}"
            attempts.append(fallback_error)
            print(f"[TRANSCRIBE] {fallback_error}")

        self.backend = None
        self.status.selected_backend = "unavailable"
        self.status.ready = False
        self.status.initialization_error = " | ".join(attempts)
        print(
            "[TRANSCRIBE] No transcription backend is available. "
            f"Details: {self.status.initialization_error}"
        )

    def _initialize_explicit_faster_whisper_or_fail(
        self,
        attempts: List[str],
    ) -> None:
        """
        Explicit faster_whisper never silently switches to pywhispercpp.

        With device=auto, CUDA is attempted first and CPU/int8 is a valid
        faster-whisper fallback.
        """

        try:
            if self.config.faster_whisper_device in {"auto", "cuda"}:
                self._initialize_faster_whisper_cuda()
                return
        except Exception as exc:
            message = f"CUDA faster-whisper unavailable: {exc}"
            attempts.append(message)
            print(f"[TRANSCRIBE] {message}")

        try:
            if self.config.faster_whisper_device in {"auto", "cpu"}:
                self._initialize_faster_whisper_cpu()
                if attempts:
                    self.status.fallback_used = True
                    self.status.fallback_reason = "; ".join(attempts)
                return
        except Exception as exc:
            message = f"CPU faster-whisper unavailable: {exc}"
            attempts.append(message)
            print(f"[TRANSCRIBE] {message}")

        self.backend = None
        self.status.selected_backend = "unavailable"
        self.status.ready = False
        self.status.initialization_error = " | ".join(attempts)
        print(
            "[TRANSCRIBE] Explicit faster-whisper selection failed. "
            f"Details: {self.status.initialization_error}"
        )

    def _initialize_faster_whisper_cuda(self) -> None:
        supported = _get_supported_compute_types("cuda")
        self.status.supported_cuda_compute_types = sorted(supported)

        if not supported:
            raise RuntimeError(
                "CTranslate2 reports no usable CUDA compute types. "
                "CUDA/cuDNN may be missing, incompatible, or unavailable."
            )

        compute_type = _select_compute_type(
            requested=self.config.faster_whisper_compute_type,
            supported=supported,
            preferred=["float16", "int8_float16", "int8", "float32"],
        )

        if compute_type is None:
            raise RuntimeError(
                "No requested or compatible CUDA compute type was found. "
                f"Supported: {sorted(supported)}"
            )

        backend = FasterWhisperBackend(
            config=self.config,
            device="cuda",
            compute_type=compute_type,
        )
        self._set_ready_faster_whisper(backend)

    def _initialize_faster_whisper_cpu(self) -> None:
        supported = _get_supported_compute_types("cpu")
        self.status.supported_cpu_compute_types = sorted(supported)

        if not supported:
            raise RuntimeError("CTranslate2 reports no usable CPU compute types.")

        compute_type = _select_compute_type(
            requested=self.config.faster_whisper_compute_type,
            supported=supported,
            preferred=["int8", "int8_float32", "float32"],
        )

        if compute_type is None:
            raise RuntimeError(
                "No requested or compatible CPU compute type was found. "
                f"Supported: {sorted(supported)}"
            )

        backend = FasterWhisperBackend(
            config=self.config,
            device="cpu",
            compute_type=compute_type,
        )
        self._set_ready_faster_whisper(backend)

    def _initialize_pywhispercpp(self) -> None:
        backend = PyWhisperCppBackend(config=self.config)
        self.backend = backend
        self.status.selected_backend = backend.backend_name
        self.status.selected_device = "cpu"
        self.status.selected_compute_type = None
        self.status.model = str(backend.model_path)
        self.status.ready = True
        self.status.initialized_at_unix = time.time()
        self.status.initialization_error = None
        print(
            "[TRANSCRIBE] Ready: backend=pywhispercpp "
            f"model={backend.model_path.name} threads={self.config.threads} "
            f"language={self.config.language}"
        )

    def _set_ready_faster_whisper(
        self,
        backend: FasterWhisperBackend,
    ) -> None:
        self.backend = backend
        self.status.selected_backend = backend.backend_name
        self.status.selected_device = backend.device
        self.status.selected_compute_type = backend.compute_type
        self.status.model = backend.model_name
        self.status.ready = True
        self.status.initialized_at_unix = time.time()
        self.status.initialization_error = None
        print(
            "[TRANSCRIBE] Ready: backend=faster_whisper "
            f"device={backend.device} compute_type={backend.compute_type} "
            f"model={backend.model_name} language={self.config.language}"
        )

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        """Transcribe one audio segment and record operational timing."""

        if not self.status.ready or self.backend is None:
            return ""

        if samples.size == 0:
            return ""

        peak = float(np.max(np.abs(samples)))
        if peak < self.config.min_peak:
            print(
                "[TRANSCRIBE] below_min_peak "
                f"peak={peak:.4f} threshold={self.config.min_peak:.4f}"
            )
            return ""

        started = time.perf_counter()
        try:
            text = self.backend.transcribe(samples, sample_rate)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.status.last_transcription_ms = round(elapsed_ms, 2)
            self.status.last_transcription_error = None

            if text:
                print(
                    "[TRANSCRIBE] "
                    f"backend={self.status.selected_backend} "
                    f"device={self.status.selected_device} "
                    f"elapsed_ms={elapsed_ms:.1f} text={text}"
                )
            else:
                print(
                    "[TRANSCRIBE] "
                    f"backend={self.status.selected_backend} "
                    f"elapsed_ms={elapsed_ms:.1f} raw_to_empty"
                )

            return text
        except KeyboardInterrupt:
            return ""
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.status.last_transcription_ms = round(elapsed_ms, 2)
            self.status.last_transcription_error = str(exc)
            print(
                "[TRANSCRIBE] transcription failed: "
                f"{exc}\n{traceback.format_exc()}"
            )
            return ""

    def get_status(self) -> Dict[str, Any]:
        return self.status.to_dict()


# -----------------------
# Capability helpers
# -----------------------

def _get_supported_compute_types(device: str) -> Set[str]:
    """
    Query CTranslate2 capabilities without relying on torch.cuda.is_available().

    Importing CTranslate2 can succeed even if CUDA is not usable. The later
    WhisperModel constructor remains the final device readiness check.
    """

    try:
        import ctranslate2  # type: ignore

        supported = ctranslate2.get_supported_compute_types(device)
        return {str(item).lower() for item in supported}
    except Exception as exc:
        print(
            "[TRANSCRIBE] CTranslate2 capability query failed "
            f"for device={device}: {exc}"
        )
        return set()


def _select_compute_type(
    requested: str,
    supported: Set[str],
    preferred: List[str],
) -> Optional[str]:
    """Return the requested type when valid, otherwise use a safe preference."""

    normalized_requested = (requested or "auto").strip().lower()

    if normalized_requested != "auto":
        if normalized_requested in supported:
            return normalized_requested

        print(
            "[TRANSCRIBE] Requested compute type "
            f"{normalized_requested!r} is unsupported; "
            f"available={sorted(supported)}."
        )
        return None

    for candidate in preferred:
        if candidate in supported:
            return candidate

    return None


# -----------------------
# Audio preparation (preserved)
# -----------------------

def _prepare_audio(
    samples: np.ndarray,
    sample_rate: int,
    min_seconds: float,
) -> np.ndarray:
    """Convert arbitrary mono float samples to padded 16 kHz float32 audio."""

    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return np.array([], dtype=np.float32)

    minimum_frames = int(max(0.0, min_seconds) * max(1, sample_rate))
    if audio.size < minimum_frames:
        audio = np.pad(
            audio,
            (0, minimum_frames - audio.size),
            mode="constant",
        )

    if sample_rate != 16000:
        audio = _resample_to_16k(audio, sample_rate)

    return np.ascontiguousarray(audio, dtype=np.float32)


def _resample_to_16k(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Dependency-free linear resampling for short live microphone segments."""

    if sample_rate == 16000:
        return samples.astype(np.float32, copy=False)

    if sample_rate <= 0 or samples.size == 0:
        return np.array([], dtype=np.float32)

    target_length = int(samples.shape[0] * (16000.0 / float(sample_rate)))
    if target_length <= 0:
        return np.array([], dtype=np.float32)

    source_positions = np.linspace(
        0.0,
        1.0,
        num=samples.shape[0],
        endpoint=False,
        dtype=np.float64,
    )
    target_positions = np.linspace(
        0.0,
        1.0,
        num=target_length,
        endpoint=False,
        dtype=np.float64,
    )

    return np.interp(
        target_positions,
        source_positions,
        samples.astype(np.float64, copy=False),
    ).astype(np.float32, copy=False)


def _to_int16(samples: np.ndarray) -> np.ndarray:
    """Convert normalized float audio for old pywhispercpp fallback APIs."""

    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16, copy=False)


# -----------------------
# Output parsing/cleaning (preserved)
# -----------------------

def _parse_pywhispercpp_output(raw: object) -> str:
    """Normalize several pywhispercpp return shapes into a text string."""

    if raw is None:
        return ""

    if isinstance(raw, dict):
        direct_text = raw.get("text")
        if isinstance(direct_text, str):
            return direct_text

        segments = raw.get("segments")
        if isinstance(segments, list):
            return " ".join(
                str(segment.get("text", "")).strip()
                for segment in segments
                if isinstance(segment, dict)
            ).strip()

        return ""

    text = str(raw).strip()
    if not text or text == "[]":
        return ""

    if text.startswith("[") and "text=" in text:
        parts = TEXT_FIELD_RE.findall(text)
        if parts:
            cleaned: List[str] = []
            for part in parts:
                value = part.strip()
                if (
                    len(value) >= 2
                    and value[0] == value[-1]
                    and value[0] in "\"'"
                ):
                    value = value[1:-1]
                if value.strip():
                    cleaned.append(value.strip())
            return " ".join(cleaned).strip()

    return text


def clean_transcript(raw: str) -> str:
    """
    Remove empty, whitespace-only, and obvious non-speech fillers.
    Keep short but plausible sentence starts (avoid over-filtering).
    """

    if not raw:
        return ""

    text = " ".join(raw.split()).strip()
    if not text:
        return ""

    # Filter single-word trivial fillers only (more permissive than before)
    single = text.lower()
    if len(text.split()) == 1 and single in {
        "ja",
        "und",
        "also",
        "äh",
        "oh",
        "ähm",
        "hm",
        "hmm",
    }:
        return ""

    # Remove meta/fx-only very short lines like "(Applaus)"
    if META_RE.search(text) and len(text.split()) <= 3:
        return ""

    return text


# -----------------------
# Public API (preserved)
# -----------------------

_MANAGER: Optional[TranscriptionManager] = None


def get_transcription_manager() -> TranscriptionManager:
    """Return the process-wide manager without initializing model weights."""
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = TranscriptionManager()
    return _MANAGER


def init_transcription_backend() -> TranscriptionManager:
    """Initialize the configured backend once during FastAPI lifespan startup."""
    manager = get_transcription_manager()
    manager.initialize()
    return manager


def transcribe_chunk(samples: np.ndarray, sample_rate: int) -> str:
    """Application-facing synchronous transcription function."""
    return get_transcription_manager().transcribe(samples, sample_rate)


def get_transcription_status() -> Dict[str, Any]:
    """Application-facing status payload for API/UI diagnostics."""
    return get_transcription_manager().get_status()


# -----------------------
# Realtime Transcriber (new)
# -----------------------

RealtimeEvent = Dict[str, Any]
RealtimeCallback = Callable[[RealtimeEvent], None]


class RealtimeTranscriber:
    """
    Low-latency realtime transcriber that reuses the already-initialized backend via
    TranscriptionManager. Feed frames (float32 mono), receive 'partial' and 'final' events.

    Events:
      - {"type":"partial", "text": str, "process_ms": int, "record_ms": int}
      - {"type":"final", "text": str, "process_ms": int, "record_ms": int, "fallback_used": bool}
    """

    def __init__(self, sample_rate: int, config: Optional[TranscriptionConfig] = None) -> None:
        self.sample_rate = int(sample_rate)
        self.manager = get_transcription_manager()
        self.config = config or self.manager.config

        # Rolling buffer and queue
        self._queue: "asyncio.Queue[Optional[np.ndarray]]" = asyncio.Queue(maxsize=128)
        self._subs: List[RealtimeCallback] = []
        self._task: Optional[asyncio.Task] = None
        self._stop_evt = asyncio.Event()

        # Timers
        self._started_ms: int = 0
        self._last_voice_ms: int = 0

        # Rolling audio
        self._rolling = np.zeros(0, dtype=np.float32)

        # Recording for file fallback
        self._record_bytes: Optional[io.BytesIO] = None
        self._record_path: Optional[Path] = None

        # Running merged text and finalize debounce
        self._cur_utt: str = ""
        self._finalize_task: Optional[asyncio.Task] = None

        # Base finalize delay (kept for backward-compat but adaptive delays will be used)
        self._finalize_delay_sec: float = _env_float("APP_FINALIZE_BASE_DELAY_SEC", 0.35)

        # Final "meaningfulness" thresholds
        self._min_chars: int = _env_int("APP_TEXT_MIN_CHARS", 12)
        self._min_words: int = _env_int("APP_TEXT_MIN_WORDS", 3)

        # Partial emit debouncing and tail-window decode
        self._last_partial_emit_ms: int = 0
        self._last_emitted_partial_norm: str = ""
        self._min_partial_emit_interval_ms: int = _env_int("APP_MIN_PARTIAL_EMIT_MS", 600)
        self._partial_min_growth_chars: int = _env_int("APP_PARTIAL_MIN_GROWTH_CHARS", 3)
        self._partial_min_growth_words: int = _env_int("APP_PARTIAL_MIN_GROWTH_WORDS", 1)
        self._partial_min_chars: int = _env_int("APP_PARTIAL_MIN_CHARS", 10)  # meaningful-light
        self._partial_min_words: int = _env_int("APP_PARTIAL_MIN_WORDS", 2)   # meaningful-light

        # Decode ticker for partials (deterministic cadence, avoids bursty re-decodes)
        self._min_partial_decode_interval_ms: int = _env_int("APP_MIN_PARTIAL_DECODE_MS", 350)

        # Tail decode windows
        self._tail_decode_sec: float = _env_float("APP_RT_TAIL_DECODE_SEC", 10.0)
        self._tail_min_sec: float = _env_float("APP_RT_TAIL_MIN_SEC", 6.0)   # reserved for future dynamic tail
        self._tail_max_sec: float = _env_float("APP_RT_TAIL_MAX_SEC", 12.0)  # reserved for future dynamic tail

        # Adaptive finalize delays depending on sentence-end detection
        self._finalize_base_delay_sec: float = _env_float("APP_FINALIZE_BASE_DELAY_SEC", 0.35)
        self._finalize_nosent_delay_sec: float = _env_float("APP_FINALIZE_NOSENT_DELAY_SEC", 0.85)

        # File fallback minimum record time (ms) to avoid over-triggering on tiny utterances
        self._fallback_min_record_ms: int = _env_int("APP_RT_FALLBACK_MIN_RECORD_MS", 1500)

        # Prepare fallback dir if needed
        if self.config.rt_fallback_record:
            Path(self.config.rt_fallback_dir).mkdir(parents=True, exist_ok=True)

    def subscribe(self, cb: RealtimeCallback) -> None:
        self._subs.append(cb)

    async def start(self) -> None:
        if self._task is not None:
            return
        # Ensure backend initialized lazily
        self.manager.initialize()
        self._started_ms = _now_ms()
        self._last_voice_ms = self._started_ms
        if self.config.rt_fallback_record:
            self._record_bytes = io.BytesIO()
        self._task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop_evt.set()
        with contextlib.suppress(Exception):
            self._queue.put_nowait(None)
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

        # Cancel pending finalize task
        if self._finalize_task and not self._finalize_task.done():
            self._finalize_task.cancel()
            self._finalize_task = None

        # Flush a meaningful remainder if present
        if _is_meaningful(self._cur_utt, min_chars=self._min_chars, min_words=self._min_words):
            self._emit({
                "type": "final",
                "text": self._cur_utt.strip(),
                "process_ms": 0,
                "record_ms": _now_ms() - self._started_ms if self._started_ms else 0,
                "fallback_used": False,
            })
        self._cur_utt = ""

    async def feed(self, frames: np.ndarray) -> None:
        if self._task is None:
            await self.start()
        a = _float32_mono(frames)
        if a.size == 0:
            return
        try:
            self._queue.put_nowait(a)
        except asyncio.QueueFull:
            # Drop the oldest to keep latency low
            with contextlib.suppress(Exception):
                _ = self._queue.get_nowait()
            with contextlib.suppress(Exception):
                self._queue.put_nowait(a)

    # -----------------------
    # Internal worker
    # -----------------------

    async def _worker(self) -> None:
        sr = self.sample_rate
        slice_len = max(1, int(sr * float(self.config.rt_slice_sec)))
        max_keep = max(1, int(sr * float(self.config.rt_window_sec)))

        buf = self._rolling
        last_decode_ms = _now_ms()

        while not self._stop_evt.is_set():
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=0.25)
            except asyncio.TimeoutError:
                item = None

            if isinstance(item, np.ndarray):
                buf = np.concatenate([buf, item], axis=0)

                if self.config.rt_fallback_record and self._record_bytes is not None:
                    self._record_bytes.write(item.astype(np.float32, copy=False).tobytes())

                if self.config.vad_enabled:
                    if _rms(item) >= float(self.config.vad_rms_threshold):
                        self._last_voice_ms = _now_ms()

            # Truncate rolling buffer (keep up to 2x max_keep to avoid growth bursts)
            if buf.size > (max_keep * 2):
                buf = buf[-max_keep:]

            now = _now_ms()
            elapsed_ms = now - self._started_ms
            since_last_voice = now - self._last_voice_ms

            # Deterministic partial decode ticker
            should_decode = False
            if buf.size >= slice_len:
                if (now - last_decode_ms) >= self._min_partial_decode_interval_ms:
                    should_decode = True

            # End-of-utterance detection via silence debounce
            should_finalize = since_last_voice >= int(self.config.rt_end_debounce_ms)

            if should_decode:
                # Use only the last tail window for decoding (stable partials)
                tail_frames = int(self.sample_rate * max(2.5, min(self._tail_decode_sec, self.config.rt_window_sec)))
                audio_for_decode = buf[-tail_frames:] if buf.size > tail_frames else buf
                txt, proc_ms = await self._decode_partial(audio_for_decode)
                last_decode_ms = _now_ms()
                if txt:
                    merged = _merge_overlap(self._cur_utt, txt, min_overlap=5, max_overlap=18)
                    # Debounce and dedupe partials
                    now_ms = _now_ms()
                    merged_norm = re.sub(r"\s+", " ", merged).strip().lower()

                    # Require growth
                    growth_chars = len(merged_norm) - len(self._last_emitted_partial_norm)
                    growth_words = max(0, len(merged_norm.split()) - len(self._last_emitted_partial_norm.split()))
                    interval_ok = (now_ms - self._last_partial_emit_ms) >= self._min_partial_emit_interval_ms
                    growth_ok = (growth_chars >= self._partial_min_growth_chars) or (growth_words >= self._partial_min_growth_words)
                    changed_ok = merged_norm != self._last_emitted_partial_norm

                    # Light meaningfulness on partials to reduce 1-word flicker
                    meaningful_light = (len(merged_norm) >= self._partial_min_chars) or (len(merged_norm.split()) >= self._partial_min_words)

                    # Skip punct-only terminal changes (reduce flicker)
                    if _is_punct_only_change(self._last_emitted_partial_norm, merged_norm):
                        changed_ok = False

                    self._cur_utt = merged
                    if interval_ok and changed_ok and growth_ok and meaningful_light:
                        self._emit({
                            "type": "partial",
                            "text": self._cur_utt,
                            "process_ms": proc_ms,
                            "record_ms": elapsed_ms,
                        })
                        self._last_partial_emit_ms = now_ms
                        self._last_emitted_partial_norm = merged_norm

            if should_finalize and buf.size > 0:
                # Use a slightly longer tail for finalize than for partial
                tail_frames = int(self.sample_rate * max(3.0, min(self._tail_decode_sec, self.config.rt_window_sec)))
                audio_for_decode = buf[-tail_frames:] if buf.size > tail_frames else buf
                candidate_txt, proc_ms = await self._decode_partial(audio_for_decode, finalize=True)
                candidate_txt = clean_transcript(candidate_txt)

                fallback_used = False

                # If empty AND recording was sufficiently long, try file fallback
                if (not candidate_txt) and self.config.rt_fallback_record and (elapsed_ms >= self._fallback_min_record_ms):
                    with contextlib.suppress(Exception):
                        fp = await self._flush_recording_wav(sr)
                        if fp is not None:
                            fb_txt = await transcribe_file_for_fallback(fp)
                            fb_txt = clean_transcript(fb_txt)
                            if fb_txt:
                                candidate_txt = fb_txt
                                fallback_used = True

                # Merge into running utterance (dedupe)
                if candidate_txt:
                    self._cur_utt = _merge_overlap(self._cur_utt, candidate_txt, min_overlap=5, max_overlap=18)

                # Adaptive finalize scheduling
                if self._cur_utt:
                    if SENTENCE_END_RE.search(self._cur_utt):
                        # If a clear sentence end exists, shorten delay
                        delay = max(0.2, self._finalize_base_delay_sec * 0.6)
                    else:
                        # No obvious sentence end, wait a bit longer
                        delay = self._finalize_nosent_delay_sec

                    self._schedule_finalize(delay=delay,
                                            process_ms=proc_ms,
                                            record_ms=elapsed_ms,
                                            fallback_used=fallback_used)

                # Reset raw rolling buffer for next turn of speech
                buf = np.zeros(0, dtype=np.float32)
                self._last_voice_ms = _now_ms()
                if self.config.rt_fallback_record:
                    self._record_bytes = io.BytesIO()
                    self._record_path = None

        # Optionally flush on shutdown (non-critical)
        with contextlib.suppress(Exception):
            await self._flush_recording_wav(sr)

    async def _decode_partial(self, audio: np.ndarray, finalize: bool = False) -> Tuple[str, int]:
        """
        Decode current rolling window using the already-selected backend.
        Returns (text, process_ms).
        """
        start_ms = _now_ms()

        backend = self.manager.backend
        if backend is None:
            # Should not happen; ensure initialized
            self.manager.initialize()
            backend = self.manager.backend
        txt = ""
        try:
            # Keep same low-latency decode path for both partial and finalize
            txt = backend.transcribe(audio, sample_rate=self.sample_rate) if backend else ""
        except Exception as e:
            print(f"[RT] decode error: {e}")

        end_ms = _now_ms()
        return clean_transcript(txt), (end_ms - start_ms)

    def _schedule_finalize(self, delay: float, process_ms: int, record_ms: int, fallback_used: bool) -> None:
        """
        Debounced scheduling of final emit. Cancels any pending finalize task.
        """
        # Cancel any previous finalize
        if self._finalize_task and not self._finalize_task.done():
            self._finalize_task.cancel()
        loop = asyncio.get_running_loop()
        self._finalize_task = loop.create_task(
            self._delayed_finalize(delay, process_ms, record_ms, fallback_used)
        )

    async def _delayed_finalize(self, delay: float, process_ms: int, record_ms: int, fallback_used: bool) -> None:
        """
        Wait a short delay; if the accumulated utterance is still meaningful, emit final.
        """
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        text = re.sub(r"\s+", " ", (self._cur_utt or "")).strip()
        if not _is_meaningful(text, min_chars=self._min_chars, min_words=self._min_words):
            # Keep as partial; do not finalize meaningless fragments
            return
        # Emit final
        self._emit({
            "type": "final",
            "text": text,
            "process_ms": process_ms,
            "record_ms": record_ms,
            "fallback_used": bool(fallback_used),
        })
        # Reset current utterance buffer and partial-debounce trackers
        self._cur_utt = ""
        self._last_emitted_partial_norm = ""
        self._last_partial_emit_ms = _now_ms()

    def _emit(self, evt: RealtimeEvent) -> None:
        for cb in list(self._subs):
            try:
                cb(evt)
            except Exception as e:
                print(f"[RT] subscriber error: {e}")

    async def _flush_recording_wav(self, sr: int) -> Optional[Path]:
        """
        Write in-memory float32 mono PCM to a WAV file for fallback transcription.
        """
        if not self.config.rt_fallback_record or self._record_bytes is None:
            return None

        try:
            raw = self._record_bytes.getvalue()
            if not raw:
                return None

            import wave
            out_dir = Path(self.config.rt_fallback_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            p = out_dir / f"rt_{int(time.time())}.wav"
            with wave.open(str(p), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(4)  # float32
                wf.setframerate(int(sr))
                wf.writeframes(raw)
            self._record_path = p
            return p
        except Exception as e:
            print(f"[RT] writing fallback WAV failed: {e}")
            return None


# -----------------------
# Helpers for realtime
# -----------------------

def _now_ms() -> int:
    return int(time.time() * 1000)


def _float32_mono(arr: np.ndarray) -> np.ndarray:
    if arr is None:
        return np.zeros(0, dtype=np.float32)
    a = np.asarray(arr)
    if a.dtype != np.float32:
        a = a.astype(np.float32, copy=False)
    if a.ndim == 2 and a.shape[1] > 1:
        a = a[:, 0]
    return a


def _rms(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame, dtype=np.float32), dtype=np.float64)))


# -----------------------
# File transcription fallback (reuses existing backends)
# -----------------------

async def transcribe_file_for_fallback(path: Union[str, Path]) -> str:
    """
    Transcribe an audio file using the already-initialized backend.
    For faster-whisper we can pass a path. For pywhispercpp, we decode to float32 16k first.
    """
    manager = get_transcription_manager()
    manager.initialize()
    backend = manager.backend
    if backend is None:
        return ""

    # If faster-whisper backend is active, try path directly via its .transcribe by loading file
    if isinstance(backend, FasterWhisperBackend):
        try:
            # Determine effective language at runtime also for file fallback
            hint_lang = (os.getenv("APP_INPUT_LOCALE_HINT", "") or "").strip().lower() or None
            conf_lang = manager.config.language or None
            eff_lang = conf_lang or hint_lang or None

            kwargs: Dict[str, Any] = dict(
                language=eff_lang,  # None enables auto-detect
                beam_size=manager.config.faster_whisper_beam_size,
                temperature=manager.config.temperature,
                vad_filter=manager.config.faster_whisper_vad_filter,
                condition_on_previous_text=manager.config.condition_on_previous_text,
                word_timestamps=False,
                without_timestamps=True,
            )
            if manager.config.no_speech_threshold is not None:
                kwargs["no_speech_threshold"] = manager.config.no_speech_threshold
            if manager.config.logprob_threshold is not None:
                kwargs["log_prob_threshold"] = manager.config.logprob_threshold

            segments, _info = backend.model.transcribe(str(path), **kwargs)  # type: ignore

            parts: List[str] = []
            for seg in segments:
                t = getattr(seg, "text", "")
                if isinstance(t, str) and t.strip():
                    parts.append(t.strip())
            return clean_transcript(" ".join(parts))
        except Exception as e:
            print(f"[FALLBACK] faster-whisper file transcribe failed: {e}")
            return ""

    # For pywhispercpp backend: decode file to float32 mono 16k and pass into backend.transcribe
    try:
        data, sr = _load_audio_as_f32_mono(path, target_sr=16000)
        return backend.transcribe(data, sr)  # type: ignore
    except Exception as e:
        print(f"[FALLBACK] pywhispercpp file transcribe failed: {e}")
        return ""


def _load_audio_as_f32_mono(path: Union[str, Path], target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load arbitrary audio file and return float32 mono resampled to target_sr.
    Uses soundfile; keeps dependencies minimal.
    """
    import soundfile as sf  # type: ignore

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if data.shape[1] > 1:
        data = data[:, 0:1]
    data = data[:, 0]
    if sr != target_sr:
        ratio = float(target_sr) / float(sr)
        x_idx = np.arange(0, data.size) * ratio
        data = np.interp(x_idx, np.arange(data.size), data).astype(np.float32)
        sr = target_sr
    return data, sr
