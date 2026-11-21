# config_generator.py
#
# Generate local NVIDIA configuration from hardware.json with tier-specific
# STT/LLM/Image settings and scheduler mode. Writes a single YAML file.
# No runtime validation, no model downloads, no network calls.

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import yaml

GIB = 1024**3
SCHEMA_VERSION = 1

# Tier thresholds (non-aggregated, single GPU)
TIER_MINIMUMS: dict[str, int] = {
    "6gb": 6 * GIB,
    "8gb": 8 * GIB,
    "16gb": 16 * GIB,
    "24gb": 24 * GIB,
}

DEFAULT_HARDWARE_PATH = Path("/app/config/hardware.json")
DEFAULT_CONFIG_PATH = Path("/app/config/generated.yaml")


class ConfigurationError(ValueError):
    """Report invalid input or unsuccessful configuration generation."""


def _integer(data: dict[str, Any], key: str, minimum: int = 0) -> int:
    """Get an integer >= minimum from a dict, otherwise raise."""
    value = data.get(key)
    if type(value) is not int or value < minimum:
        raise ConfigurationError(
            f"Hardware field '{key}' must be an integer >= {minimum}."
        )
    return value


def validate_hardware(data: object) -> dict[str, Any]:
    """Validate and normalize the report without replacing its metadata."""
    if not isinstance(data, dict):
        raise ConfigurationError("Hardware file must contain a JSON object.")

    version = data.get("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ConfigurationError("Hardware schema version missing or unsupported.")
    if data.get("memory_unit") != "GiB":
        raise ConfigurationError("Hardware field 'memory_unit' must be 'GiB'.")

    apple_silicon = data.get("is_apple_silicon")
    if type(apple_silicon) is not bool:
        raise ConfigurationError("Hardware field 'is_apple_silicon' must be boolean.")
    if apple_silicon:
        raise ConfigurationError("Apple Silicon is not supported for this local profile.")

    index = _integer(data, "gpu_index")
    count = _integer(data, "gpu_count", 1)
    if index >= count:
        raise ConfigurationError("GPU index is out of bounds for detected GPUs.")

    identity: dict[str, str] = {}
    for key in ("gpu_uuid", "gpu_name"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError(f"Hardware field '{key}' must be a non-empty string.")
        identity[key] = value.strip()

    total = _integer(data, "vram_total_bytes", 1)
    free = _integer(data, "vram_free_bytes")
    used = _integer(data, "vram_used_bytes")
    if free > total or used > total or free + used > total:
        raise ConfigurationError("Hardware file contains inconsistent VRAM values.")

    return {
        "schema_version": version,
        "memory_unit": "GiB",
        "is_apple_silicon": apple_silicon,
        "gpu_index": index,
        "gpu_count": count,
        **identity,
        "vram_total_bytes": total,
        "vram_free_bytes": free,
        "vram_used_bytes": used,
    }


def select_tier(total_bytes: int, override: str | None = None) -> str:
    """Select by single-device capacity; allow smaller or equal profile overrides."""
    if type(total_bytes) is not int or total_bytes < TIER_MINIMUMS["6gb"]:
        raise ConfigurationError(
            "Local GPU image profiles require at least 6 GiB VRAM on one NVIDIA GPU; "
            "configure CPU/remote separately."
        )

    detected = "6gb"
    for name, minimum in TIER_MINIMUMS.items():
        if total_bytes >= minimum:
            detected = name

    if override is None or override == "":
        return detected
    if not isinstance(override, str) or override not in TIER_MINIMUMS:
        raise ConfigurationError("SLAIDSHOW_TIER must be 6gb, 8gb, 16gb or 24gb.")
    if TIER_MINIMUMS[override] > total_bytes:
        raise ConfigurationError(
            f"Profile '{override}' exceeds the capacity of the selected GPU; "
            f"auto-detected: '{detected}'."
        )
    # Allow smaller profile overrides.
    return override


def _tier_presets(tier: str) -> Dict[str, Any]:
    """
    Return tier-specific runtime presets without performing any validation
    or downloads. These are conservative but usable defaults aligned to
    the VRAM tiers in the project brief.
    """
    # Common image backend base URL (local ComfyUI service by default)
    image_base_url = "http://127.0.0.1:8188"
    ollama_base_url = "http://127.0.0.1:11434"

    if tier == "6gb":
        # 6–8 GiB: CPU STT, small quant LLM, SD1.5 + LCM @512, sequential
        return {
            "stt": {
                "backend": "faster-whisper",
                "device": "cpu",
                "compute_type": "int8",
                "model": "small-int8",  # symbolic; app may map to local cache
                "model_selection_required": False,
            },
            "llm": {
                "backend": "ollama",
                "base_url": ollama_base_url,
                "model": "qwen3:1.7b-q4",
                "selection_status": "selected_by_tier",
                "license_verified": False,
                "artifact_verified": False,
            },
            "image": {
                "backend": "comfyui",
                "base_url": image_base_url,
                "intended_model_family": "sd1.5",
                "intended_acceleration": "lcm",
                "width": 512,
                "height": 512,
                "steps": 4,
                "workflow_path": None,
                "checkpoint_path": None,
                "lcm_artifact_path": None,
                "validated": False,
            },
            "scheduler": {
                "enabled": True,
                "mode": "sequential",
                "parallel_allowed": False,
            },
        }

    if tier == "8gb":
        # 8–12 GiB: GPU STT (cuda), SD1.5+LCM @768, sequential
        return {
            "stt": {
                "backend": "faster-whisper",
                "device": "cuda",
                "compute_type": "float16",  # explicit fp16 for stability
                "model": "medium-fp16",     # symbolic
                "model_selection_required": False,
            },
            "llm": {
                "backend": "ollama",
                "base_url": ollama_base_url,
                "model": "qwen3:4b-q4",
                "selection_status": "selected_by_tier",
                "license_verified": False,
                "artifact_verified": False,
            },
            "image": {
                "backend": "comfyui",
                "base_url": image_base_url,
                "intended_model_family": "sd1.5",
                "intended_acceleration": "lcm",
                "width": 768,
                "height": 768,
                "steps": 4,
                "workflow_path": None,
                "checkpoint_path": None,
                "lcm_artifact_path": None,
                "validated": False,
            },
            "scheduler": {
                "enabled": True,
                "mode": "sequential",
                "parallel_allowed": False,  # <12 GiB stays sequential
            },
        }

    if tier == "16gb":
        # 12–16 GiB effective: allow parallel; FLUX.2 small 4B FP8; larger LLM
        return {
            "stt": {
                "backend": "faster-whisper",
                "device": "cuda",
                "compute_type": "float16",
                "model": "medium-fp16",
                "model_selection_required": False,
            },
            "llm": {
                "backend": "ollama",
                "base_url": ollama_base_url,
                "model": "gemma4:26b-a4b-q4",
                "selection_status": "selected_by_tier",
                "license_verified": False,
                "artifact_verified": False,
            },
            "image": {
                "backend": "comfyui",
                "base_url": image_base_url,
                "intended_model_family": "flux.2-small",
                "intended_acceleration": "fp8",
                "width": 1024,
                "height": 1024,
                "steps": 6,
                "workflow_path": None,
                "checkpoint_path": None,
                "lcm_artifact_path": None,
                "validated": False,
            },
            "scheduler": {
                "enabled": True,
                "mode": "parallel",
                "parallel_allowed": True,  # ≥12 GiB per spec
            },
        }

    if tier == "24gb":
        # 16+ GiB: FLUX.2 small 4B + optional Qwen-Image 2.1; biggest LLM
        return {
            "stt": {
                "backend": "faster-whisper",
                "device": "cuda",
                "compute_type": "float16",
                "model": "large-v2-fp16",  # symbolic
                "model_selection_required": False,
            },
            "llm": {
                "backend": "ollama",
                "base_url": ollama_base_url,
                "model": "qwen3.8:27b-q4",
                "selection_status": "selected_by_tier",
                "license_verified": False,
                "artifact_verified": False,
            },
            "image": {
                "backend": "comfyui",
                "base_url": image_base_url,
                "intended_model_family": "flux.2-small+qwenimage2.1",
                "intended_acceleration": "fp8",
                "width": 1024,
                "height": 1024,
                "steps": 6,
                "workflow_path": None,
                "checkpoint_path": None,
                "lcm_artifact_path": None,
                "validated": False,
            },
            "scheduler": {
                "enabled": True,
                "mode": "parallel",
                "parallel_allowed": True,
            },
        }

    # Fallback (should not occur): use safest 6gb
    return _tier_presets("6gb")


def build_config(hardware: dict[str, Any], tier: str) -> dict[str, Any]:
    """Compose a tier-specific configuration without contacting services."""
    validated = validate_hardware(hardware)
    selected = select_tier(validated["vram_total_bytes"], tier)
    hardware_snapshot = {
        key: value
        for key, value in validated.items()
        if key not in {"schema_version", "memory_unit"}
    }

    presets = _tier_presets(selected)
    # Conservative scheduler reserve
    reserve_bytes = 512 * 1024**2
    parallel_allowed = bool(presets["scheduler"]["parallel_allowed"])
    mode = "parallel" if parallel_allowed else "sequential"

    return {
        "schema_version": SCHEMA_VERSION,
        "tier": selected,
        "hardware": {
            **hardware_snapshot,
            "memory_unit": "bytes",
            "memory_values_are_snapshot": True,
            "gpu_index_scope": "hardware_detection_environment",
            "runtime_gpu_uuid_verification_required": True,
        },
        "readiness": {
            "runtime_integration_required": True,
            "model_validation_required": True,
            "workflow_validation_required": True,
            "gpu_acceptance_test_required": True,
        },
        "runtime": {
            "offline": True,
            "cloud_enabled": False,
            "remote_backends_enabled": False,
            "frontend_assets": "local",
            "event_transport": "sse",
        },
        "stt": presets["stt"],
        "llm": presets["llm"],
        "image": presets["image"],
        "scheduler": {
            "enabled": True,
            "mode": mode,
            "parallel_allowed": parallel_allowed,
            "require_confirmed_unload": not parallel_allowed,
            "on_unload_failure": "block_next_gpu_phase",
            "transition_timeout_seconds": 60,
            "vram_reserve_bytes": reserve_bytes,
            "reserve_requires_hardware_validation": True,
            "measure_free_vram_before_each_phase": not parallel_allowed,
            "phase_budget_validation_required": not parallel_allowed,
        },
        "downloads": {
            # Downloads remain disabled until artifact/license verification exists.
            "enabled": False,
            "reason": "model_artifacts_and_licenses_require_validation",
            "artifacts": [],
        },
        "paths": {
            "models": "/app/models",
            "outputs": "/app/outputs",
            "recordings": "/app/recordings",
        },
    }


def _path(
    explicit: str | Path | None,
    environment_key: str,
    default: Path,
) -> Path:
    """Reject empty paths instead of interpreting them as the current directory."""
    raw = (
        explicit
        if explicit is not None
        else os.environ.get(environment_key, str(default))
    )
    if not isinstance(raw, (str, Path)):
        raise ConfigurationError(f"{environment_key}: invalid file path.")
    if isinstance(raw, str) and not raw.strip():
        raise ConfigurationError(f"{environment_key} must not be empty.")
    try:
        return Path(raw).expanduser()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ConfigurationError(
            f"{environment_key}: failed to resolve path: {exc}"
        ) from exc


def _validate_paths(source: Path, target: Path) -> None:
    """Protect the hardware report; ensure distinct files and no symlinks."""
    try:
        identical = source.resolve() == target.resolve()
        if not identical and source.exists() and target.exists():
            identical = source.samefile(target)
        if identical:
            raise ConfigurationError("Hardware file and config file must not be identical.")
        if target.is_symlink():
            raise ConfigurationError("Config file must not be a symbolic link.")
    except (OSError, RuntimeError, ValueError) as exc:
        if isinstance(exc, ConfigurationError):
            raise
        raise ConfigurationError(f"Failed to check file paths: {exc}") from exc


def _write_yaml_atomic(
    config: dict[str, Any],
    target: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write YAML atomically; preserve an existing file unless overwrite=True."""
    temporary: str | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            raise ConfigurationError("Config file must not be a symbolic link.")
        if target.exists() and not overwrite:
            raise ConfigurationError(
                f"Config already exists: {target}; use --force to replace."
            )

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())

        if overwrite:
            os.replace(temporary, target)
            temporary = None
        else:
            # Atomic publish using a hard link; fail closed if not supported.
            try:
                os.link(temporary, target)
            except FileExistsError as exc:
                raise ConfigurationError(
                    f"Config already exists: {target}; use --force to replace."
                ) from exc
    except (OSError, ValueError, yaml.YAMLError) as exc:
        if isinstance(exc, ConfigurationError):
            raise
        raise ConfigurationError(
            f"Failed to persist configuration safely: {target}: {exc}"
        ) from exc
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def generate(
    hardware_path: str | Path | None = None,
    output_path: str | Path | None = None,
    tier: str | None = None,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Read hardware and persist configuration without contacting services."""
    if type(overwrite) is not bool:
        raise ConfigurationError("overwrite must be boolean.")

    source = _path(hardware_path, "HARDWARE_PATH", DEFAULT_HARDWARE_PATH)
    target = _path(output_path, "CONFIG_PATH", DEFAULT_CONFIG_PATH)
    _validate_paths(source, target)

    try:
        with source.open(encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ConfigurationError(f"Failed to read hardware file: {source}: {exc}") from exc

    hardware = validate_hardware(raw)
    override = tier if tier is not None else os.environ.get("SLAIDSHOW_TIER")
    selected = select_tier(hardware["vram_total_bytes"], override)
    config = build_config(hardware, selected)
    _write_yaml_atomic(config, target, overwrite=overwrite)
    return config


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: generate tier-specific config; replace only with --force."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a tier-specific local NVIDIA configuration; "
            "existing files are only replaced with --force."
        )
    )
    parser.add_argument("--hardware", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--tier",
        choices=tuple(TIER_MINIMUMS),
        help="Takes precedence over SLAIDSHOW_TIER when provided.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing configuration file atomically.",
    )
    args = parser.parse_args(argv)

    try:
        target = _path(args.output, "CONFIG_PATH", DEFAULT_CONFIG_PATH)
        config = generate(
            args.hardware,
            target,
            args.tier,
            overwrite=args.force,
        )
    except ConfigurationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Config written: {target} "
        f"(tier {config['tier']}, STT={config['stt']['backend']}:{config['stt']['device']}, "
        f"scheduler={config['scheduler']['mode']}).",
        file=sys.stderr,
    )
    print(
        "Note: Models/workflows are not downloaded; Ollama and ComfyUI must be provided separately. "
        "The application should load and apply this YAML at startup.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
