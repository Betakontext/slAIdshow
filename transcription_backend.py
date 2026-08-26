"""
Selectable local speech-to-text backends for slAIdshow.

Supported backends:
- faster_whisper: Preferred backend using CTranslate2.
- pywhispercpp: Lightweight fallback using whisper.cpp Python bindings.
- auto: faster-whisper CUDA -> faster-whisper CPU/int8 -> pywhispercpp.

All comments and operational messages in this module are intentionally English.
"""

from __future__ import annotations

import os
import re
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np


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

    @classmethod
    def from_environment(cls) -> "TranscriptionConfig":
        requested_backend = _env_str("APP_WHISPER_BACKEND", "auto").lower()
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

        faster_compute_type = _env_str(
            "APP_FASTER_WHISPER_COMPUTE_TYPE",
            "auto",
        ).lower()

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

        return cls(
            requested_backend=requested_backend,
            language=_env_str("APP_WHISPER_LANGUAGE", "de"),
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

        # Log effective parameters for troubleshooting
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

        # Build kwargs conditionally to avoid overriding library defaults with None
        kwargs: Dict[str, Any] = dict(
            language=self.config.language or None,
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
    """Existing whisper.cpp-based fallback implementation."""

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
            language=config.language or None,
            translate=False,
            temperature=config.temperature,
        )

        print(
            "[TRANSCRIBE] pywhispercpp init: "
            f"model={self.model_path.name} threads={config.threads} "
            f"lang={config.language} temp={config.temperature:.2f} "
            f"min_sec={config.min_seconds:.2f} min_peak={config.min_peak:.4f}"
        )

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        audio = _prepare_audio(
            samples=samples,
            sample_rate=sample_rate,
            min_seconds=self.config.min_seconds,
        )
        if audio.size == 0:
            return ""

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

    def _initialize_pywhispercpp_or_fail(self, attempts: List[str]) -> None:
        try:
            self._initialize_pywhispercpp()
        except Exception as exc:
            message = f"pywhispercpp initialization failed: {exc}"
            attempts.append(message)
            self.backend = None
            self.status.selected_backend = "unavailable"
            self.status.ready = False
            self.status.initialization_error = " | ".join(attempts)
            print(f"[TRANSCRIBE] {message}")

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
            raise RuntimeError(
                "CTranslate2 reports no usable CPU compute types."
            )

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
# Audio preparation
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
# Output parsing/cleaning
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
# Public API
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
