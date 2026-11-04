# runtime_config.py
"""Load conservative generated configuration without initializing backends.

This module has no import-time filesystem, GPU, or network side effects.
Schema version 1 currently accepts only the conservative generator output.
Application integration must happen after dotenv loading and before importing
backend modules that read environment variables.
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
            raise RuntimeConfigurationError(
                f"Doppelter YAML-Schlüssel: '{key}'."
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
    """Check exact types and values without exposing values in errors."""
    if type(actual) is not type(expected):
        raise RuntimeConfigurationError(
            f"Ungültiger Datentyp bei '{path}'."
        )

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise RuntimeConfigurationError(f"Ungültiges Objekt bei '{path}'.")
        if actual.keys() != expected.keys():
            raise RuntimeConfigurationError(
                f"Fehlende oder unbekannte Felder bei '{path}'."
            )
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

    try:
        normalized_hardware = validate_hardware(
            {
                **hardware,
                "schema_version": SCHEMA_VERSION,
                "memory_unit": "GiB",
                "is_apple_silicon": False,
            }
        )
        selected = select_tier(
            normalized_hardware["vram_total_bytes"],
            tier,
        )
        expected = build_config(normalized_hardware, selected)
    except ConfigurationError as exc:
        raise RuntimeConfigurationError(str(exc)) from exc

    # Reject edits that could bypass the conservative memory/offline policy.
    # Validated model selections require an explicit future schema extension.
    _match_baseline(data, expected)
    return copy.deepcopy(expected)


@dataclass(frozen=True)
class LoadedConfiguration:
    """Hold validated data without claiming backend readiness."""

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
        }


def load_config(
    path: str | Path | None = None,
) -> LoadedConfiguration:
    """Load required configuration; never silently fall back to environment."""
    raw_path = (
        path
        if path is not None
        else os.environ.get("CONFIG_PATH", str(DEFAULT_CONFIG_PATH))
    )
    if not str(raw_path).strip():
        raise RuntimeConfigurationError("CONFIG_PATH darf nicht leer sein.")

    source = Path(raw_path).expanduser()

    try:
        with source.open("rb") as handle:
            content = handle.read(MAX_CONFIG_BYTES + 1)
    except OSError as exc:
        raise RuntimeConfigurationError(
            f"Konfiguration konnte nicht gelesen werden: {source}"
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
    except (UnicodeError, yaml.YAMLError, RecursionError) as exc:
        raise RuntimeConfigurationError(
            "Konfiguration enthält ungültiges oder zu stark "
            "verschachteltes UTF-8/YAML."
        ) from exc

    validated = validate_config(data)

    # Overrides belong to generation, not hidden runtime mutation.
    override = os.environ.get("SLAIDSHOW_TIER")
    if override is not None and override != validated["tier"]:
        raise RuntimeConfigurationError(
            "SLAIDSHOW_TIER widerspricht der gespeicherten Konfiguration; "
            "config_generator.py mit dem gewünschten Profil erneut ausführen."
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
