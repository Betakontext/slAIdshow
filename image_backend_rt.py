from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Import your last functioning backend module without changes
# Rename it to image_backend_base.py or adjust this import accordingly.
from image_backend_base import (
    LocalComfyBackend,
    PollinationsBackend,
    BackendEnv,
    ComfyConfig,
    PollinationsConfig,
)

from runtime_settings import RuntimeSettings


def _env_bool01(k: str, d: int = 0) -> bool:
    v = (os.getenv(k, str(d)) or "").strip().lower()
    return v in {"1", "true", "yes", "on"}


def build_image_backend_rt(
    backend_override: Optional[str] = None,
    allow_cloud_override: Optional[bool] = None,
):
    """
    Runtime-aware factory:
    Priority:
      1) per-request override (if given)
      2) RuntimeSettings (set via UI)
      3) Environment defaults
    """
    env = BackendEnv()
    out_dir: Path = env.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rs = RuntimeSettings.get()
    selected = (backend_override or rs.image_backend or env.image_backend).lower()
    allow_cloud = (
        (allow_cloud_override is True)
        or rs.allow_cloud
        or _env_bool01("ALLOW_CLOUD_IMAGE_BACKEND", 0)
    )

    if selected == "comfyui":
        cfg = ComfyConfig()
        # Respect runtime size defaults if no explicit width/height is passed later.
        # Width/height are handled inside LocalComfyBackend.generate using cfg defaults,
        # so push runtime defaults into cfg to keep a single source of truth.
        cfg.width = rs.image_width or cfg.width
        cfg.height = rs.image_height or cfg.height
        return LocalComfyBackend(out_dir=out_dir, cfg=cfg)

    if selected == "pollinations":
        cfg = PollinationsConfig()
        cfg.allow_cloud = allow_cloud  # enable only if runtime/env permits
        return PollinationsBackend(out_dir=out_dir, cfg=cfg)

    raise RuntimeError(f"Unsupported IMAGE_BACKEND={selected}")
