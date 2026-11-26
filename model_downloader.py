# SPDX-FileCopyrightText: 2026 Christoph Medicus
# SPDX-License-Identifier: MIT

# model_downloader.py

"""
Validate download policy without fetching unapproved model artifacts.

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
            "Download configuration is invalid or not approved; "
            "regenerate with config_generator.py."
        ) from exc

    downloads = validated["downloads"]

    # Fail closed if a future schema enables downloads without implementing
    # verified artifacts, pinned sources, licenses, and atomic persistence here.
    if downloads.get("enabled") is not False:
        raise ModelDownloadError(
            "Active downloads are not supported by this downloader version. "
            "No downloads were started."
        )

    if downloads.get("artifacts") not in ([], None):
        raise ModelDownloadError(
            "Disabled downloads must not specify any artifacts."
        )

    LOGGER.info(
        "Downloads disabled for tier %s; no models were fetched.",
        validated["tier"],
    )

    return {
        "tier": validated["tier"],
        "downloads_enabled": False,
        "outcome": "skipped_by_policy",
        "reason": downloads.get("reason", "downloads_disabled_by_schema_v1"),
        "downloaded_artifacts": [],
        "model_presence_checked": False,
        "models_ready": False,
    }


def main(argv: list[str] | None = None) -> int:
    """Load persisted configuration and report the enforced download policy."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate the slAIdshow download policy. "
            "Schema version 1 does not allow model downloads."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "YAML path; alternatively use CONFIG_PATH "
            "or /app/config/generated.yaml."
        ),
    )
    args = parser.parse_args(argv)

    try:
        configuration = load_config(args.config)
        result = download_models(configuration.as_dict())
    except (RuntimeConfigurationError, ModelDownloadError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        "Downloads are disabled per configuration. "
        "No network access or download commands were executed. "
        "Model availability and show readiness are not confirmed.",
        file=sys.stderr,
    )

    # A disabled download policy was successfully enforced. This exit status
    # must not be interpreted as backend or model readiness.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
