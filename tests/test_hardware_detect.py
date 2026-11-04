# tests/test_harware_detect.py
"""Test NVIDIA discovery with mocked NVML and isolated filesystem writes."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

import hardware_detect as detector

GIB = 1024**3


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid platform and environment dependencies."""
    monkeypatch.delenv("GPU_INDEX", raising=False)
    monkeypatch.delenv("HARDWARE_PATH", raising=False)
    monkeypatch.setattr(detector.platform, "system", lambda: "Linux")
    monkeypatch.setattr(detector, "detect_cpu_ram", lambda: 32.0)


@pytest.fixture
def nvml(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Provide a fake NVML module without importing pynvml."""
    module = MagicMock()
    module.nvmlDeviceGetCount.return_value = 1
    module.nvmlDeviceGetHandleByIndex.side_effect = lambda index: index
    module.nvmlDeviceGetMemoryInfo.return_value = SimpleNamespace(
        total=6 * GIB,
        free=5 * GIB,
        used=GIB,
    )
    module.nvmlDeviceGetName.return_value = b"Mock NVIDIA GPU"
    module.nvmlDeviceGetUUID.return_value = b"GPU-mock-0001"
    monkeypatch.setattr(
        detector.importlib, "import_module", lambda name: module
    )
    return module


def test_detect_preserves_exact_memory_and_writes_json(
    nvml: MagicMock, tmp_path: Path
) -> None:
    target = tmp_path / "config" / "hardware.json"

    result = detector.detect(output_path=target)

    assert json.loads(target.read_text(encoding="utf-8")) == result
    assert result["schema_version"] == 1
    assert result["memory_unit"] == "GiB"
    assert result["gpu_index"] == 0
    assert result["gpu_uuid"] == "GPU-mock-0001"
    assert result["vram_total_bytes"] == 6 * GIB
    assert result["vram_free_bytes"] == 5 * GIB
    assert result["vram_used_bytes"] == GIB
    assert result["vram_gb"] == 6.0
    assert result["cpu_ram_gb"] == 32.0
    nvml.nvmlShutdown.assert_called_once()


@pytest.mark.parametrize("name", [b"Mock GPU", "Mock GPU"])
def test_gpu_name_accepts_bytes_and_text(
    nvml: MagicMock, name: bytes | str
) -> None:
    nvml.nvmlDeviceGetName.return_value = name
    assert detector.detect_nvidia()[0].gpu_name == "Mock GPU"


def test_missing_pynvml_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_module(name: str) -> Any:
        raise ImportError("mock missing module")

    monkeypatch.setattr(detector.importlib, "import_module", missing_module)

    with pytest.raises(detector.HardwareDetectionError, match="pynvml"):
        detector.detect_nvidia()


def test_initialization_failure_does_not_shutdown(nvml: MagicMock) -> None:
    nvml.nvmlInit.side_effect = RuntimeError("mock driver failure")

    with pytest.raises(detector.HardwareDetectionError, match="initialisiert"):
        detector.detect_nvidia()

    nvml.nvmlShutdown.assert_not_called()


def test_read_failure_still_shuts_down(nvml: MagicMock) -> None:
    nvml.nvmlDeviceGetMemoryInfo.side_effect = RuntimeError("mock read failure")

    with pytest.raises(detector.HardwareDetectionError, match="NVML-Daten"):
        detector.detect_nvidia()

    nvml.nvmlShutdown.assert_called_once()


def test_shutdown_failure_does_not_discard_result(nvml: MagicMock) -> None:
    nvml.nvmlShutdown.side_effect = RuntimeError("mock shutdown failure")
    assert len(detector.detect_nvidia()) == 1


def test_no_gpu_does_not_write_file(
    nvml: MagicMock, tmp_path: Path
) -> None:
    nvml.nvmlDeviceGetCount.return_value = 0
    target = tmp_path / "hardware.json"

    with pytest.raises(detector.HardwareDetectionError, match="Keine NVIDIA"):
        detector.detect(output_path=target)

    assert not target.exists()
    nvml.nvmlShutdown.assert_called_once()


def test_multiple_gpus_require_explicit_selection(
    nvml: MagicMock, tmp_path: Path
) -> None:
    nvml.nvmlDeviceGetCount.return_value = 2

    with pytest.raises(detector.HardwareDetectionError, match="Mehrere"):
        detector.detect(output_path=tmp_path / "hardware.json")


def test_selected_gpu_memory_is_not_added(
    nvml: MagicMock, tmp_path: Path
) -> None:
    nvml.nvmlDeviceGetCount.return_value = 2

    result = detector.detect(gpu_index=1, output_path=tmp_path / "hardware.json")

    assert result["gpu_index"] == 1
    assert result["gpu_count"] == 2
    assert result["vram_total_bytes"] == 6 * GIB


def test_environment_selects_gpu(
    nvml: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nvml.nvmlDeviceGetCount.return_value = 2
    monkeypatch.setenv("GPU_INDEX", "1")

    result = detector.detect(output_path=tmp_path / "hardware.json")

    assert result["gpu_index"] == 1


def test_explicit_index_overrides_environment(
    nvml: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nvml.nvmlDeviceGetCount.return_value = 2
    monkeypatch.setenv("GPU_INDEX", "invalid")

    result = detector.detect(gpu_index=0, output_path=tmp_path / "hardware.json")

    assert result["gpu_index"] == 0


@pytest.mark.parametrize("index", [-1, True, 1.5, "0"])
def test_invalid_explicit_index_is_rejected(index: Any) -> None:
    with pytest.raises(detector.HardwareDetectionError, match="GPU-Index"):
        detector.detect(gpu_index=index)


@pytest.mark.parametrize("value", ["", "-1", "1.5", " 0", "01", "true"])
def test_invalid_environment_index_is_rejected(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GPU_INDEX", value)

    with pytest.raises(detector.HardwareDetectionError, match="GPU_INDEX"):
        detector.detect()


def test_unavailable_index_is_rejected(
    nvml: MagicMock, tmp_path: Path
) -> None:
    with pytest.raises(detector.HardwareDetectionError, match="nicht verfügbar"):
        detector.detect(gpu_index=4, output_path=tmp_path / "hardware.json")


def test_small_gpu_is_reported_with_warning(
    nvml: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    nvml.nvmlDeviceGetMemoryInfo.return_value = SimpleNamespace(
        total=4 * GIB, free=3 * GIB, used=GIB
    )

    result = detector.detect(output_path=tmp_path / "hardware.json")

    assert result["vram_total_bytes"] == 4 * GIB
    assert "weniger als 6 GiB" in capsys.readouterr().err


@pytest.mark.parametrize("machine", ["arm64", "x86_64"])
def test_macos_is_rejected_before_nvml_access(
    nvml: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    machine: str,
) -> None:
    monkeypatch.setattr(detector.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(detector.platform, "machine", lambda: machine)

    with pytest.raises(detector.HardwareDetectionError, match="nicht unterstützt"):
        detector.detect()

    nvml.nvmlInit.assert_not_called()


def test_hardware_path_environment_is_used(
    nvml: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "hardware.json"
    monkeypatch.setenv("HARDWARE_PATH", str(target))

    detector.detect()

    assert target.is_file()


def test_failed_replace_preserves_previous_report(
    nvml: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "hardware.json"
    target.write_bytes(b"previous report\n")

    def fail_replace(source: Any, destination: Any) -> None:
        raise OSError("mock replace failure")

    monkeypatch.setattr(detector.os, "replace", fail_replace)

    with pytest.raises(detector.HardwareDetectionError, match="geschrieben"):
        detector.detect(output_path=target)

    assert target.read_bytes() == b"previous report\n"
    assert list(tmp_path.glob(".hardware.json.*")) == []


def test_cli_success(
    nvml: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "hardware.json"

    assert detector.main(["--output", str(target)]) == 0
    assert target.is_file()
    assert "Hardware-Datei geschrieben" in capsys.readouterr().err


def test_cli_failure(
    nvml: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    nvml.nvmlDeviceGetCount.return_value = 0
    target = tmp_path / "hardware.json"

    assert detector.main(["--output", str(target)]) == 1
    assert not target.exists()
    assert "Fehler:" in capsys.readouterr().err

