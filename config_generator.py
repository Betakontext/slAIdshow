# config_generator.py

"""Generate conservative local-NVIDIA configuration from hardware.json.

Existing configurations require explicit replacement with --force.
This generator does not validate runtime integration or model compatibility.
CPU-only and remote operation require a separate configuration path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

GIB = 1024**3
SCHEMA_VERSION = 1
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
    value = data.get(key)
    if type(value) is not int or value < minimum:
        raise ConfigurationError(
            f"Hardware-Feld '{key}' muss eine ganze Zahl ab {minimum} sein."
        )
    return value


def validate_hardware(data: object) -> dict[str, Any]:
    """Validate and normalize the report without replacing its metadata."""
    if not isinstance(data, dict):
        raise ConfigurationError("Hardware-Datei muss ein JSON-Objekt enthalten.")

    version = data.get("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ConfigurationError(
            "Hardware-Schema-Version fehlt oder wird nicht unterstützt."
        )
    if data.get("memory_unit") != "GiB":
        raise ConfigurationError("Hardware-Feld 'memory_unit' muss 'GiB' sein.")

    apple_silicon = data.get("is_apple_silicon")
    if type(apple_silicon) is not bool:
        raise ConfigurationError(
            "Hardware-Feld 'is_apple_silicon' muss ein Boolean sein."
        )
    if apple_silicon:
        raise ConfigurationError(
            "Dieses lokale GPU-Profil unterstützt Apple Silicon nicht."
        )

    index = _integer(data, "gpu_index")
    count = _integer(data, "gpu_count", 1)
    if index >= count:
        raise ConfigurationError("GPU-Index liegt außerhalb der erkannten GPUs.")

    identity: dict[str, str] = {}
    for key in ("gpu_uuid", "gpu_name"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError(
                f"Hardware-Feld '{key}' muss eine nichtleere Zeichenkette sein."
            )
        identity[key] = value.strip()

    total = _integer(data, "vram_total_bytes", 1)
    free = _integer(data, "vram_free_bytes")
    used = _integer(data, "vram_used_bytes")
    if free > total or used > total or free + used > total:
        raise ConfigurationError(
            "Hardware-Datei enthält widersprüchliche VRAM-Werte."
        )

    # A gap may represent reserved memory; equality is not required.
    # Exact byte counters are authoritative, not rounded GiB values.
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
    """Select by single-device capacity; allow smaller profile overrides."""
    if type(total_bytes) is not int or total_bytes < TIER_MINIMUMS["6gb"]:
        raise ConfigurationError(
            "Lokale GPU-Bildprofile benötigen mindestens 6 GiB VRAM auf "
            "einer einzelnen NVIDIA-GPU; CPU-/Remote-Betrieb separat "
            "konfigurieren."
        )

    detected = "6gb"
    for name, minimum in TIER_MINIMUMS.items():
        if total_bytes >= minimum:
            detected = name

    if override is None:
        return detected
    if not isinstance(override, str) or override not in TIER_MINIMUMS:
        raise ConfigurationError(
            "SLAIDSHOW_TIER muss 6gb, 8gb, 16gb oder 24gb sein."
        )
    if TIER_MINIMUMS[override] > total_bytes:
        raise ConfigurationError(
            f"Profil '{override}' übersteigt die Kapazität der ausgewählten "
            f"GPU; automatisch erkannt: '{detected}'."
        )
    return override


def build_config(hardware: dict[str, Any], tier: str) -> dict[str, Any]:
    """Use the conservative baseline until larger combinations are accepted."""
    validated = validate_hardware(hardware)
    selected = select_tier(validated["vram_total_bytes"], tier)
    hardware_snapshot = {
        key: value
        for key, value in validated.items()
        if key not in {"schema_version", "memory_unit"}
    }

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
        "stt": {
            "backend": "faster-whisper",
            "device": "cpu",
            "compute_type": "int8",
            "model": None,
            "model_selection_required": True,
        },
        "llm": {
            "backend": "ollama",
            "base_url": "http://127.0.0.1:11434",
            "model": "gemma3:1b",
            "selection_status": "existing_default_pending_validation",
            "license_verified": False,
            "artifact_verified": False,
        },
        "image": {
            "backend": "comfyui",
            "base_url": "http://127.0.0.1:8188",
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
            "require_confirmed_unload": True,
            "on_unload_failure": "block_next_gpu_phase",
            "transition_timeout_seconds": 60,
            "vram_reserve_bytes": 512 * 1024**2,
            "reserve_requires_hardware_validation": True,
            "measure_free_vram_before_each_phase": True,
            "phase_budget_validation_required": True,
        },
        "downloads": {
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
        raise ConfigurationError(f"{environment_key}: ungültiger Dateipfad.")
    if isinstance(raw, str) and not raw.strip():
        raise ConfigurationError(f"{environment_key} darf nicht leer sein.")
    try:
        return Path(raw).expanduser()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ConfigurationError(
            f"{environment_key}: Dateipfad konnte nicht aufgelöst werden: {exc}"
        ) from exc


def _validate_paths(source: Path, target: Path) -> None:
    """Protect the hardware report, including symlink and hard-link aliases."""
    try:
        identical = source.resolve() == target.resolve()
        if not identical and source.exists() and target.exists():
            identical = source.samefile(target)
        if identical:
            raise ConfigurationError(
                "Hardware-Datei und Konfigurationsdatei "
                "dürfen nicht identisch sein."
            )
        if target.is_symlink():
            raise ConfigurationError(
                "Die Konfigurationsdatei darf kein symbolischer Link sein."
            )
    except (OSError, RuntimeError, ValueError) as exc:
        if isinstance(exc, ConfigurationError):
            raise
        raise ConfigurationError(
            f"Dateipfade konnten nicht geprüft werden: {exc}"
        ) from exc


def _write_yaml_atomic(
    config: dict[str, Any],
    target: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Publish a complete file, preserving existing files unless authorized."""
    temporary: str | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            raise ConfigurationError(
                "Die Konfigurationsdatei darf kein symbolischer Link sein."
            )
        if target.exists() and not overwrite:
            raise ConfigurationError(
                f"Konfiguration existiert bereits: {target}; "
                "zum bewussten Ersetzen --force verwenden."
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
            yaml.safe_dump(
                config,
                handle,
                allow_unicode=True,
                sort_keys=False,
            )
            handle.flush()
            os.fsync(handle.fileno())

        if overwrite:
            os.replace(temporary, target)
            temporary = None
        else:
            # Atomic no-clobber publication prevents concurrent regeneration.
            # Fail safely if the filesystem does not support hard links.
            try:
                os.link(temporary, target)
            except FileExistsError as exc:
                raise ConfigurationError(
                    f"Konfiguration existiert bereits: {target}; "
                    "zum bewussten Ersetzen --force verwenden."
                ) from exc
    except (OSError, ValueError, yaml.YAMLError) as exc:
        if isinstance(exc, ConfigurationError):
            raise
        raise ConfigurationError(
            f"Konfiguration konnte nicht sicher gespeichert werden: "
            f"{target}: {exc}"
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
        raise ConfigurationError("overwrite muss ein Boolean sein.")

    source = _path(hardware_path, "HARDWARE_PATH", DEFAULT_HARDWARE_PATH)
    target = _path(output_path, "CONFIG_PATH", DEFAULT_CONFIG_PATH)
    _validate_paths(source, target)

    try:
        with source.open(encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ConfigurationError(
            f"Hardware-Datei konnte nicht gelesen werden: {source}: {exc}"
        ) from exc

    hardware = validate_hardware(raw)
    override = tier if tier is not None else os.environ.get("SLAIDSHOW_TIER")
    selected = select_tier(hardware["vram_total_bytes"], override)
    config = build_config(hardware, selected)
    _write_yaml_atomic(config, target, overwrite=overwrite)
    return config


def main(argv: list[str] | None = None) -> int:
    """Provide a standalone configuration generation command."""
    parser = argparse.ArgumentParser(
        description=(
            "Erzeugt eine konservative lokale NVIDIA-Konfiguration; "
            "bestehende Dateien werden nur mit --force ersetzt."
        )
    )
    parser.add_argument("--hardware", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--tier",
        choices=tuple(TIER_MINIMUMS),
        help="Hat Vorrang vor SLAIDSHOW_TIER.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bestehende Konfiguration vollständig ersetzen.",
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
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1

    print(
        f"Konfiguration gespeichert: {target} "
        f"(Profil {config['tier']}, CPU-STT, sequenzielle GPU-Nutzung).",
        file=sys.stderr,
    )
    print(
        "Hinweis: Alle Tiers verwenden vorerst dieselbe konservative Basis; "
        "Laufzeitintegration, Dienst-Endpunkte, Modelle, Workflows und "
        "Hardware-Abnahme müssen noch geprüft werden. "
        "Downloads sind deaktiviert.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


