# tests/test_config_generator.py

"""Test configuration generation without GPU access or network calls."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

import config_generator as generator


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate tests from local configuration overrides."""
    for name in ("SLAIDSHOW_TIER", "HARDWARE_PATH", "CONFIG_PATH"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def hardware() -> dict[str, Any]:
    """Provide a valid report for one selected 6 GiB GPU."""
    return {
        "schema_version": 1,
        "memory_unit": "GiB",
        "is_apple_silicon": False,
        "gpu_index": 0,
        "gpu_count": 1,
        "gpu_uuid": "GPU-test-0001",
        "gpu_name": "Mock NVIDIA GPU",
        "vram_total_bytes": 6 * generator.GIB,
        "vram_free_bytes": 5 * generator.GIB,
        "vram_used_bytes": generator.GIB,
    }


@pytest.fixture
def paths(
    tmp_path: Path, hardware: dict[str, Any]
) -> tuple[Path, Path]:
    source = tmp_path / "hardware.json"
    source.write_text(json.dumps(hardware), encoding="utf-8")
    return source, tmp_path / "config" / "generated.yaml"


@pytest.mark.parametrize(
    ("total_bytes", "expected"),
    [
        (6 * generator.GIB, "6gb"),
        (8 * generator.GIB - 1, "6gb"),
        (8 * generator.GIB, "8gb"),
        (12 * generator.GIB, "8gb"),
        (16 * generator.GIB - 1, "8gb"),
        (16 * generator.GIB, "16gb"),
        (24 * generator.GIB - 1, "16gb"),
        (24 * generator.GIB, "24gb"),
        (48 * generator.GIB, "24gb"),
    ],
)
def test_tier_boundaries(total_bytes: int, expected: str) -> None:
    assert generator.select_tier(total_bytes) == expected


@pytest.mark.parametrize(
    "total_bytes",
    [0, -1, 6 * generator.GIB - 1, True, 6.0, "6"],
)
def test_invalid_capacity_is_rejected(total_bytes: Any) -> None:
    with pytest.raises(generator.ConfigurationError):
        generator.select_tier(total_bytes)


@pytest.mark.parametrize("override", ["", "12gb", "6GB", " 6gb", 6])
def test_invalid_override_is_rejected(override: Any) -> None:
    with pytest.raises(generator.ConfigurationError):
        generator.select_tier(24 * generator.GIB, override)


def test_smaller_override_is_allowed() -> None:
    assert generator.select_tier(24 * generator.GIB, "6gb") == "6gb"


def test_larger_override_is_rejected() -> None:
    with pytest.raises(generator.ConfigurationError, match="übersteigt"):
        generator.select_tier(6 * generator.GIB, "8gb")


def test_hardware_validation_does_not_mutate_input(
    hardware: dict[str, Any],
) -> None:
    original = copy.deepcopy(hardware)
    result = generator.validate_hardware(hardware)

    assert hardware == original
    assert result["vram_total_bytes"] == 6 * generator.GIB
    assert result["gpu_uuid"] == hardware["gpu_uuid"]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("memory_unit", "GB"),
        ("is_apple_silicon", True),
        ("gpu_index", -1),
        ("gpu_index", True),
        ("gpu_count", 0),
        ("gpu_uuid", ""),
        ("gpu_name", " "),
        ("vram_total_bytes", 0),
        ("vram_free_bytes", -1),
        ("vram_used_bytes", False),
    ],
)
def test_invalid_hardware_field_is_rejected(
    hardware: dict[str, Any], key: str, value: Any
) -> None:
    hardware[key] = value
    with pytest.raises(generator.ConfigurationError):
        generator.validate_hardware(hardware)


@pytest.mark.parametrize(
    "key",
    [
        "schema_version",
        "memory_unit",
        "is_apple_silicon",
        "gpu_index",
        "gpu_count",
        "gpu_uuid",
        "gpu_name",
        "vram_total_bytes",
        "vram_free_bytes",
        "vram_used_bytes",
    ],
)
def test_missing_hardware_field_is_rejected(
    hardware: dict[str, Any], key: str
) -> None:
    del hardware[key]
    with pytest.raises(generator.ConfigurationError):
        generator.validate_hardware(hardware)


def test_invalid_gpu_index_is_rejected(hardware: dict[str, Any]) -> None:
    hardware["gpu_index"] = hardware["gpu_count"]
    with pytest.raises(generator.ConfigurationError):
        generator.validate_hardware(hardware)


def test_inconsistent_memory_is_rejected(hardware: dict[str, Any]) -> None:
    hardware["vram_free_bytes"] = hardware["vram_total_bytes"]
    with pytest.raises(generator.ConfigurationError):
        generator.validate_hardware(hardware)


def test_multiple_gpu_memory_is_not_added(
    paths: tuple[Path, Path], hardware: dict[str, Any]
) -> None:
    source, target = paths
    hardware["gpu_count"] = 4
    source.write_text(json.dumps(hardware), encoding="utf-8")

    config = generator.generate(source, target)

    assert config["tier"] == "6gb"
    assert config["hardware"]["vram_total_bytes"] == 6 * generator.GIB


def test_low_free_memory_does_not_change_capacity_tier(
    paths: tuple[Path, Path], hardware: dict[str, Any]
) -> None:
    source, target = paths
    hardware["vram_free_bytes"] = 0
    hardware["vram_used_bytes"] = hardware["vram_total_bytes"]
    source.write_text(json.dumps(hardware), encoding="utf-8")

    config = generator.generate(source, target)

    assert config["tier"] == "6gb"
    assert config["scheduler"]["measure_free_vram_before_each_phase"] is True
    assert config["scheduler"]["phase_budget_validation_required"] is True


def test_safe_configuration_round_trip(paths: tuple[Path, Path]) -> None:
    source, target = paths
    config = generator.generate(source, target)

    assert yaml.safe_load(target.read_text(encoding="utf-8")) == config
    assert config["schema_version"] == 1
    assert config["stt"]["device"] == "cpu"
    assert config["scheduler"]["mode"] == "sequential"
    assert config["scheduler"]["parallel_allowed"] is False
    assert config["scheduler"]["require_confirmed_unload"] is True
    assert config["scheduler"]["on_unload_failure"] == "block_next_gpu_phase"
    assert config["image"]["width"] == 512
    assert config["image"]["height"] == 512
    assert config["runtime"]["offline"] is True
    assert config["runtime"]["cloud_enabled"] is False
    assert config["runtime"]["remote_backends_enabled"] is False
    assert config["downloads"]["enabled"] is False
    assert config["downloads"]["artifacts"] == []
    assert all(config["readiness"].values())


def test_environment_paths_are_used(
    paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    source, target = paths
    monkeypatch.setenv("HARDWARE_PATH", str(source))
    monkeypatch.setenv("CONFIG_PATH", str(target))

    generator.generate()

    assert target.is_file()


def test_explicit_tier_overrides_environment(
    paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    source, target = paths
    monkeypatch.setenv("SLAIDSHOW_TIER", "24gb")

    assert generator.generate(source, target, tier="6gb")["tier"] == "6gb"


def test_invalid_environment_override_preserves_existing_config(
    paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    source, target = paths
    target.parent.mkdir()
    target.write_text("previous configuration\n", encoding="utf-8")
    monkeypatch.setenv("SLAIDSHOW_TIER", "24gb")

    with pytest.raises(generator.ConfigurationError):
        generator.generate(source, target)

    assert target.read_text(encoding="utf-8") == "previous configuration\n"


@pytest.mark.parametrize("content", ["{", "[]", "null", '{"schema_version": 1}'])
def test_invalid_json_report_is_rejected(
    paths: tuple[Path, Path], content: str
) -> None:
    source, target = paths
    source.write_text(content, encoding="utf-8")

    with pytest.raises(generator.ConfigurationError):
        generator.generate(source, target)

    assert not target.exists()


def test_missing_hardware_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(generator.ConfigurationError):
        generator.generate(tmp_path / "missing.json", tmp_path / "generated.yaml")


def test_source_cannot_be_overwritten(paths: tuple[Path, Path]) -> None:
    source, _ = paths
    previous = source.read_bytes()

    with pytest.raises(generator.ConfigurationError, match="identisch"):
        generator.generate(source, source)

    assert source.read_bytes() == previous


def test_failed_replace_preserves_existing_config_and_removes_temporary_file(
    paths: tuple[Path, Path],
) -> None:
    source, target = paths
    target.parent.mkdir()
    target.write_bytes(b"previous configuration\n")

    with (
        patch.object(generator.os, "replace", side_effect=OSError("mock failure")),
        pytest.raises(generator.ConfigurationError),
    ):
        generator.generate(source, target)

    assert target.read_bytes() == b"previous configuration\n"
    assert list(target.parent.glob(f".{target.name}.*.tmp")) == []


def test_failed_serialization_preserves_existing_config(
    paths: tuple[Path, Path],
) -> None:
    source, target = paths
    target.parent.mkdir()
    target.write_bytes(b"previous configuration\n")

    with (
        patch.object(generator.yaml, "safe_dump", side_effect=yaml.YAMLError("mock")),
        pytest.raises(generator.ConfigurationError),
    ):
        generator.generate(source, target)

    assert target.read_bytes() == b"previous configuration\n"
    assert list(target.parent.glob(f".{target.name}.*.tmp")) == []


def test_cli_success(
    paths: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    source, target = paths

    result = generator.main(
        ["--hardware", str(source), "--output", str(target), "--tier", "6gb"]
    )

    assert result == 0
    assert target.is_file()
    assert "Profil 6gb" in capsys.readouterr().err


def test_cli_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = generator.main(
        [
            "--hardware",
            str(tmp_path / "missing.json"),
            "--output",
            str(tmp_path / "generated.yaml"),
        ]
    )

    assert result == 1
    assert "Fehler:" in capsys.readouterr().err
    assert not (tmp_path / "generated.yaml").exists()
