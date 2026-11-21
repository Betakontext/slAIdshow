# vram_scheduler.py


import time
import torch
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from slaidshow.config_generator import TIERS


class GpuState(Enum):
    IDLE = "idle"
    LLM_ACTIVE = "llm_active"
    IMAGE_ACTIVE = "image_active"
    TRANSITIONING = "transitioning"


@dataclass
class SchedulerConfig:
    vram_budget_mb: int
    parallel: bool
    transition_timeout_s: float = 5.0


class VramScheduler:
    """
    Steuert die GPU-Zuweisung zwischen LLM und ComfyUI.
    Bei parallel=True (≥12GB) ist dies ein No-Op.
    """

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.state = GpuState.IDLE
        self._llm_loaded = False
        self._image_loaded = False

    @property
    def is_parallel(self) -> bool:
        return self.config.parallel

    async def acquire_llm(self) -> None:
        """GPU für LLM-Inferenz reservieren."""
        if self.is_parallel:
            return
        self._ensure_image_unloaded()
        self.state = GpuState.LLM_ACTIVE
        self._llm_loaded = True

    async def release_llm(self) -> None:
        """LLM-VRAM freigeben."""
        if self.is_parallel:
            return
        self._llm_loaded = False
        torch.cuda.empty_cache()
        self.state = GpuState.IDLE

    async def acquire_image(self) -> None:
        """GPU für Bildgenerierung reservieren."""
        if self.is_parallel:
            return
        self._ensure_llm_unloaded()
        self.state = GpuState.IMAGE_ACTIVE
        self._image_loaded = True

    async def release_image(self) -> None:
        """ComfyUI-VRAM freigeben."""
        if self.is_parallel:
            return
        self._image_loaded = False
        torch.cuda.empty_cache()
        self.state = GpuState.IDLE

    def _ensure_llm_unloaded(self) -> None:
        if self._llm_loaded:
            self.release_llm_sync()

    def _ensure_image_unloaded(self) -> None:
        if self._image_loaded:
            self.release_image_sync()

    def release_llm_sync(self) -> None:
        self._llm_loaded = False
        torch.cuda.empty_cache()

    def release_image_sync(self) -> None:
        self._image_loaded = False
        torch.cuda.empty_cache()

    def get_status(self) -> dict:
        return {
            "state": self.state.value,
            "parallel": self.config.parallel,
            "llm_loaded": self._llm_loaded,
            "image_loaded": self._image_loaded,
        }
