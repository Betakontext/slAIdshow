# model_downloader.py
"""Validate download policy without fetching unapproved model artifacts.

Schema version 1 permits only disabled downloads. Supporting real downloads
requires an explicitly validated artifact contract and a schema extension.

This module performs no filesystem, network, or subprocess operations at import.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from runtime_config import (
    RuntimeConfigurationError,
    load_config,
    validate_config,
)

LOGGER = logging.getLogger(__name__)


class ModelDownloadError(RuntimeError):
    """Report invalid policy or unsupported artifact downloads."""


def download_models(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the complete configuration before considering any download.

    Return a policy result, not a claim that models exist or are usable.
    Never invoke a model server, shell command, or download client.
    """
    try:
        validated = validate_config(config)
    except (RuntimeConfigurationError, RecursionError) as exc:
        raise ModelDownloadError(
            "Download-Konfiguration ist ungültig oder nicht freigegeben; "
            "mit config_generator.py neu erzeugen."
        ) from exc

    downloads = validated["downloads"]

    # Fail closed if a future schema enables downloads without implementing
    # verified artifacts, pinned sources, licenses, and atomic persistence here.
    if downloads["enabled"] is not False:
        raise ModelDownloadError(
            "Aktive Downloads werden von dieser Downloader-Version "
            "nicht unterstützt. Es wurden keine Downloads gestartet."
        )

    if downloads["artifacts"] != []:
        raise ModelDownloadError(
            "Deaktivierte Downloads dürfen keine Artefakte enthalten."
        )

    LOGGER.info(
        "Downloads für Profil %s deaktiviert; keine Modelle geladen.",
        validated["tier"],
    )

    return {
        "tier": validated["tier"],
        "downloads_enabled": False,
        "outcome": "skipped_by_policy",
        "reason": downloads["reason"],
        "downloaded_artifacts": [],
        "model_presence_checked": False,
        "models_ready": False,
    }


def main(argv: list[str] | None = None) -> int:
    """Load persisted configuration and report the enforced download policy."""
    parser = argparse.ArgumentParser(
        description=(
            "Prüft die slAIdshow-Downloadrichtlinie. "
            "Schema-Version 1 erlaubt keine Modell-Downloads."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "YAML-Pfad; alternativ CONFIG_PATH "
            "oder /app/config/generated.yaml."
        ),
    )
    args = parser.parse_args(argv)

    try:
        configuration = load_config(args.config)
        result = download_models(configuration.as_dict())
    except (RuntimeConfigurationError, ModelDownloadError) as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        "Downloads sind gemäß Konfiguration deaktiviert. "
        "Es wurden keine Netzwerkzugriffe oder Download-Befehle ausgeführt. "
        "Modellverfügbarkeit und Show-Startfähigkeit sind nicht bestätigt.",
        file=sys.stderr,
    )

    # A disabled download policy was successfully enforced. This exit status
    # must not be interpreted as backend or model readiness.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

