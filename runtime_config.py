# runtime_config.py
"""Load conservative generated configuration without initializing backends.

This module has no import-time filesystem, GPU, or network side effects.
Schema version 1 accepts only the current conservative generator output.
Application integration must happen after dotenv loading and before importing
backend modules that read environment variables.

Loading configuration does not enforce runtime policy or establish readiness.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from config_generator import (
    DEFAULT_CONFIG_PATH,
    SCHEMA_VERSION,
    ConfigurationError,
    build_config,
    select_tier,
    validate_hardware,
)

MAX_CONFIG_BYTES = 256 * 1024
HARDWARE_REPORT_FIELDS = (
    "is_apple_silicon",
    "gpu_index",
    "gpu_count",
    "gpu_uuid",
    "gpu_name",
    "vram_total_bytes",
    "vram_free_bytes",
    "vram_used_bytes",
)


class RuntimeConfigurationError(ValueError):
    """Report missing, malformed, or unsupported runtime configuration."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate mapping keys instead of silently overwriting them."""


def _construct_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
) -> dict[str, Any]:
    """Require unique string keys in every YAML mapping."""
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str):
            raise RuntimeConfigurationError(
                "YAML-Schlüssel müssen Zeichenketten sein."
            )
        if key in result:
            # Do not echo untrusted keys or potential secrets.
            raise RuntimeConfigurationError(
                "Konfiguration enthält einen doppelten YAML-Schlüssel."
            )
        result[key] = loader.construct_object(value_node, deep=True)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def _match_baseline(
    actual: object,
    expected: object,
    path: str = "config",
) -> None:
    """Check exact types and values without exposing configuration values."""
    if type(actual) is not type(expected):
        raise RuntimeConfigurationError(
            f"Ungültiger Datentyp bei '{path}'."
        )

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise RuntimeConfigurationError(
                f"Ungültiges Objekt bei '{path}'."
            )
        if actual.keys() != expected.keys():
            raise RuntimeConfigurationError(
                f"Fehlende oder unbekannte Felder bei '{path}'."
            )
        # Only trusted baseline keys are included in diagnostic paths.
        for key, value in expected.items():
            _match_baseline(actual[key], value, f"{path}.{key}")
        return

    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise RuntimeConfigurationError(
                f"Nicht freigegebene Liste bei '{path}'."
            )
        for index, value in enumerate(expected):
            _match_baseline(actual[index], value, f"{path}[{index}]")
        return

    if actual != expected:
        raise RuntimeConfigurationError(
            f"Nicht freigegebener Wert bei '{path}'; "
            "Konfiguration mit config_generator.py neu erzeugen."
        )


def validate_config(data: object) -> dict[str, Any]:
    """Accept only the current conservative configuration contract."""
    if not isinstance(data, dict):
        raise RuntimeConfigurationError(
            "Konfiguration muss ein YAML-Objekt enthalten."
        )

    version = data.get("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        raise RuntimeConfigurationError(
            "Konfigurations-Schema-Version fehlt oder wird nicht unterstützt."
        )

    hardware = data.get("hardware")
    if not isinstance(hardware, dict):
        raise RuntimeConfigurationError("Hardware-Konfiguration fehlt.")

    tier = data.get("tier")
    if not isinstance(tier, str):
        raise RuntimeConfigurationError("Konfigurationsprofil fehlt.")

    if hardware.get("memory_unit") != "bytes":
        raise RuntimeConfigurationError(
            "Hardware-Konfiguration muss Byte-Zähler deklarieren."
        )
    if hardware.get("is_apple_silicon") is not False:
        raise RuntimeConfigurationError(
            "Dieses lokale GPU-Profil benötigt "
            "'is_apple_silicon: false'."
        )

    # Adapt the generated schema to the detector schema explicitly.
    # Preserve all supplied hardware facts, especially platform information.
    # The detector's GiB label accompanies authoritative byte counters.
    report = {
        "schema_version": version,
        "memory_unit": "GiB",
        **{key: hardware.get(key) for key in HARDWARE_REPORT_FIELDS},
    }

    try:
        normalized_hardware = validate_hardware(report)
        selected = select_tier(
            normalized_hardware["vram_total_bytes"],
            tier,
        )
        expected = build_config(normalized_hardware, selected)
    except ConfigurationError as exc:
        # Generator errors may include untrusted override values.
        raise RuntimeConfigurationError(
            "Hardware-Konfiguration oder Profil ist ungültig; "
            "Konfiguration mit config_generator.py neu erzeugen."
        ) from exc

    # Reject edits that could bypass the conservative memory/offline policy.
    # Model selections and deployment variants need an explicit contract update.
    _match_baseline(data, expected)
    return copy.deepcopy(expected)


@dataclass(frozen=True)
class LoadedConfiguration:
    """Hold validated data without claiming backend readiness.

    Use as_dict() for application initialization. The private dictionary is
    an internal implementation detail, not a security boundary.
    """

    path: Path
    _data: dict[str, Any]

    @property
    def tier(self) -> str:
        return str(self._data["tier"])

    def as_dict(self) -> dict[str, Any]:
        """Return a detached copy for application initialization."""
        return copy.deepcopy(self._data)

    def status(self) -> dict[str, Any]:
        """Expose policy status without URLs, paths, prompts, or credentials."""
        return {
            "loaded": True,
            "schema_version": self._data["schema_version"],
            "tier": self.tier,
            "offline_policy": self._data["runtime"]["offline"],
            "stt_device_policy": self._data["stt"]["device"],
            "scheduler_mode_policy": self._data["scheduler"]["mode"],
            "parallel_allowed": self._data["scheduler"]["parallel_allowed"],
            "image_size_policy": {
                "width": self._data["image"]["width"],
                "height": self._data["image"]["height"],
            },
            "downloads_enabled": self._data["downloads"]["enabled"],
            "readiness": copy.deepcopy(self._data["readiness"]),
            "runtime_policy_enforced": False,
            "hardware_snapshot_revalidated": False,
            "backend_readiness_checked": False,
        }


def _config_path(path: str | Path | None) -> Path:
    """Resolve explicit or environment configuration paths without I/O."""
    raw_path = (
        path
        if path is not None
        else os.environ.get("CONFIG_PATH", str(DEFAULT_CONFIG_PATH))
    )
    if not isinstance(raw_path, (str, Path)):
        raise RuntimeConfigurationError(
            "CONFIG_PATH muss eine Zeichenkette oder ein Path sein."
        )
    if isinstance(raw_path, str) and not raw_path.strip():
        raise RuntimeConfigurationError("CONFIG_PATH darf nicht leer sein.")

    try:
        source = Path(raw_path).expanduser()
        if "\x00" in str(source):
            raise RuntimeConfigurationError(
                "CONFIG_PATH enthält ein ungültiges Nullzeichen."
            )
        return source
    except (OSError, RuntimeError, ValueError) as exc:
        if isinstance(exc, RuntimeConfigurationError):
            raise
        raise RuntimeConfigurationError(
            "CONFIG_PATH konnte nicht aufgelöst werden."
        ) from exc


def load_config(
    path: str | Path | None = None,
) -> LoadedConfiguration:
    """Load required configuration without silently mutating it."""
    source = _config_path(path)

    try:
        with source.open("rb") as handle:
            content = handle.read(MAX_CONFIG_BYTES + 1)
    except (OSError, ValueError) as exc:
        raise RuntimeConfigurationError(
            "Konfiguration konnte nicht gelesen werden; "
            "CONFIG_PATH und Dateiberechtigungen prüfen."
        ) from exc

    if len(content) > MAX_CONFIG_BYTES:
        raise RuntimeConfigurationError(
            "Konfigurationsdatei überschreitet 256 KiB."
        )

    try:
        data = yaml.load(
            content.decode("utf-8"),
            Loader=_UniqueKeyLoader,
        )
    except RuntimeConfigurationError:
        raise
    except (UnicodeError, yaml.YAMLError, RecursionError, ValueError) as exc:
        raise RuntimeConfigurationError(
            "Konfiguration enthält ungültiges oder zu stark "
            "verschachteltes UTF-8/YAML."
        ) from exc

    try:
        validated = validate_config(data)
    except RecursionError as exc:
        raise RuntimeConfigurationError(
            "Konfiguration enthält eine nicht unterstützte Verschachtelung."
        ) from exc

    # Overrides belong to generation, not hidden runtime mutation.
    override = os.environ.get("SLAIDSHOW_TIER")
    if override is not None and override != validated["tier"]:
        raise RuntimeConfigurationError(
            "SLAIDSHOW_TIER widerspricht der gespeicherten Konfiguration; "
            "config_generator.py mit dem gewünschten Profil und --force "
            "bewusst erneut ausführen."
        )

    return LoadedConfiguration(path=source, _data=validated)


def main(argv: list[str] | None = None) -> int:
    """Validate persisted YAML without loading models or contacting servers."""
    parser = argparse.ArgumentParser(
        description="Prüft die persistierte slAIdshow-Konfiguration."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML-Pfad; alternativ CONFIG_PATH oder /app/config/generated.yaml.",
    )
    args = parser.parse_args(argv)

    try:
        configuration = load_config(args.config)
    except RuntimeConfigurationError as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            configuration.status(),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
