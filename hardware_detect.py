# hardware_detect.py

"""Discover one NVIDIA GPU and atomically persist its hardware profile."""

from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import logging
import os
import platform
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

OUTPUT_PATH = Path("/app/config/hardware.json")
_GIB = 1024**3
LOGGER = logging.getLogger(__name__)


class HardwareDetectionError(RuntimeError):
    """Report unavailable hardware or unsuccessful persistence."""


@dataclass(frozen=True)
class HardwareProfile:
    """Describe one GPU without aggregating memory across devices."""

    vram_gb: float
    gpu_name: str
    gpu_count: int
    cpu_ram_gb: float
    os: str
    is_apple_silicon: bool
    gpu_index: int
    gpu_uuid: str
    vram_total_bytes: int
    vram_free_bytes: int
    vram_used_bytes: int


def detect_cpu_ram() -> float:
    """Return physical RAM in GiB, or zero when unavailable."""
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024 / _GIB
    except (OSError, ValueError, IndexError):
        pass

    try:
        return (
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / _GIB
        )
    except (AttributeError, OSError, ValueError):
        pass

    if platform.system() == "Windows":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_uint32),
                ("load", ctypes.c_uint32),
                ("total_physical", ctypes.c_uint64),
                ("available_physical", ctypes.c_uint64),
                ("total_page_file", ctypes.c_uint64),
                ("available_page_file", ctypes.c_uint64),
                ("total_virtual", ctypes.c_uint64),
                ("available_virtual", ctypes.c_uint64),
                ("available_extended_virtual", ctypes.c_uint64),
            ]

        try:
            state = MemoryStatus()
            state.length = ctypes.sizeof(state)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
                return state.total_physical / _GIB
        except (AttributeError, OSError):
            pass

    return 0.0


def detect_apple_silicon() -> bool:
    """Identify Apple Silicon without treating unified memory as VRAM."""
    return platform.system() == "Darwin" and platform.machine().lower() in {
        "arm64",
        "aarch64",
    }


def _text(value: object) -> str:
    """Normalize NVML versions returning either bytes or text."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def detect_nvidia() -> list[HardwareProfile]:
    """Read GPU counters and close successfully initialized NVML."""
    try:
        nvml = importlib.import_module("pynvml")
    except ImportError as exc:
        raise HardwareDetectionError(
            "NVIDIA-Erkennung nicht möglich: Paket 'pynvml' installieren."
        ) from exc

    initialized = False
    try:
        try:
            nvml.nvmlInit()
            initialized = True
        except Exception as exc:
            raise HardwareDetectionError(
                "NVML konnte nicht initialisiert werden. "
                "NVIDIA-Treiber und Container-GPU-Zugriff prüfen."
            ) from exc

        count = nvml.nvmlDeviceGetCount()
        ram = detect_cpu_ram()
        profiles: list[HardwareProfile] = []

        for index in range(count):
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
            memory = nvml.nvmlDeviceGetMemoryInfo(handle)
            total = int(memory.total)
            free = int(memory.free)
            used = int(memory.used)

            if total <= 0 or min(free, used) < 0:
                raise HardwareDetectionError(
                    f"GPU {index} liefert ungültige VRAM-Werte."
                )
            if free > total or used > total or free + used > total:
                raise HardwareDetectionError(
                    f"GPU {index} liefert widersprüchliche VRAM-Werte."
                )

            profiles.append(
                HardwareProfile(
                    vram_gb=total / _GIB,
                    gpu_name=_text(nvml.nvmlDeviceGetName(handle)),
                    gpu_count=count,
                    cpu_ram_gb=ram,
                    os=platform.system(),
                    is_apple_silicon=False,
                    gpu_index=index,
                    gpu_uuid=_text(nvml.nvmlDeviceGetUUID(handle)),
                    vram_total_bytes=total,
                    vram_free_bytes=free,
                    vram_used_bytes=used,
                )
            )

        return profiles
    except HardwareDetectionError:
        raise
    except Exception as exc:
        raise HardwareDetectionError(
            "NVML-Daten konnten nicht gelesen werden."
        ) from exc
    finally:
        if initialized:
            try:
                nvml.nvmlShutdown()
            except Exception as exc:  # noqa: BLE001
                # NVML cleanup must not discard a successful snapshot.
                LOGGER.warning("NVML-Shutdown fehlgeschlagen: %s", exc)


def _requested_index(gpu_index: int | None) -> int | None:
    """Validate explicit selection before touching GPU hardware."""
    if gpu_index is not None:
        if type(gpu_index) is not int or gpu_index < 0:
            raise HardwareDetectionError(
                "GPU-Index muss eine nichtnegative ganze Zahl sein."
            )
        return gpu_index

    raw = os.environ.get("GPU_INDEX")
    if raw is None:
        return None
    if not re.fullmatch(r"(?:0|[1-9][0-9]*)", raw):
        raise HardwareDetectionError(
            "GPU_INDEX muss eine nichtnegative ganze Zahl sein."
        )
    try:
        return int(raw)
    except ValueError as exc:
        raise HardwareDetectionError("GPU_INDEX ist zu groß.") from exc


def _write_json_atomic(data: dict[str, Any], path: Path) -> None:
    """Preserve an existing report if writing or replacement fails."""
    temporary: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary, path)
        temporary = None
    except (OSError, ValueError, TypeError) as exc:
        raise HardwareDetectionError(
            f"Hardware-Datei konnte nicht sicher geschrieben werden: {path}"
        ) from exc
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def detect(
    gpu_index: int | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Select one GPU and persist exact byte counters."""
    requested = _requested_index(gpu_index)

    if platform.system() == "Darwin":
        label = "Apple Silicon" if detect_apple_silicon() else "macOS"
        raise HardwareDetectionError(
            f"{label} wird nicht unterstützt; NVIDIA-NVML ist erforderlich."
        )

    profiles = detect_nvidia()
    if not profiles:
        raise HardwareDetectionError(
            "Keine NVIDIA-GPU gefunden. Treiber und NVML-Zugriff prüfen."
        )
    if requested is None and len(profiles) > 1:
        raise HardwareDetectionError(
            "Mehrere NVIDIA-GPUs gefunden: GPU_INDEX oder --gpu-index angeben."
        )

    selected = (
        profiles[0]
        if requested is None
        else next(
            (profile for profile in profiles if profile.gpu_index == requested),
            None,
        )
    )
    if selected is None:
        raise HardwareDetectionError(
            f"GPU-Index {requested} ist nicht verfügbar."
        )

    if selected.vram_total_bytes < 6 * _GIB:
        print(
            f"Warnung: GPU {selected.gpu_index} hat weniger als 6 GiB VRAM.",
            file=sys.stderr,
        )

    result: dict[str, Any] = {
        "schema_version": 1,
        "memory_unit": "GiB",
        **asdict(selected),
    }
    target = Path(
        output_path
        if output_path is not None
        else os.environ.get("HARDWARE_PATH", str(OUTPUT_PATH))
    )
    _write_json_atomic(result, target)
    return result


def main(argv: list[str] | None = None) -> int:
    """Run NVIDIA discovery without downloading or loading models."""
    parser = argparse.ArgumentParser(
        description="Ermittelt die NVIDIA-Hardware für slAIdshow."
    )
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        profile = detect(args.gpu_index, args.output)
    except HardwareDetectionError as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1

    target = args.output or Path(
        os.environ.get("HARDWARE_PATH", str(OUTPUT_PATH))
    )
    print(
        f"Hardware-Datei geschrieben: {target} "
        f"(GPU {profile['gpu_index']}: {profile['gpu_name']})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

