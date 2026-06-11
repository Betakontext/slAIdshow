from __future__ import annotations

import threading
from dataclasses import dataclass

@dataclass
class _RuntimeState:
    image_backend: str = "comfyui"  # comfyui | pollinations
    allow_cloud: bool = False
    image_width: int = 512
    image_height: int = 512

class RuntimeSettings:
    """Thread-safe singleton for UI-controlled runtime settings."""
    _state: _RuntimeState | None = None
    _lock = threading.RLock()

    @classmethod
    def init(cls, image_backend: str = "comfyui", allow_cloud: bool = False, width: int = 512, height: int = 512) -> None:
        with cls._lock:
            cls._state = _RuntimeState(
                image_backend=image_backend,
                allow_cloud=allow_cloud,
                image_width=width,
                image_height=height,
            )

    @classmethod
    def get(cls) -> _RuntimeState:
        with cls._lock:
            if cls._state is None:
                cls._state = _RuntimeState()
            s = cls._state
            # return a copy to prevent external mutation
            return _RuntimeState(s.image_backend, s.allow_cloud, s.image_width, s.image_height)

    @classmethod
    def set_backend(cls, backend: str | None = None, allow_cloud: bool | None = None) -> _RuntimeState:
        with cls._lock:
            if cls._state is None:
                cls._state = _RuntimeState()
            if backend is not None:
                cls._state.image_backend = backend.lower()
            if allow_cloud is not None:
                cls._state.allow_cloud = bool(allow_cloud)
            return cls.get()

    @classmethod
    def set_size(cls, width: int | None = None, height: int | None = None) -> _RuntimeState:
        with cls._lock:
            if cls._state is None:
                cls._state = _RuntimeState()
            if width is not None:
                cls._state.image_width = int(width)
            if height is not None:
                cls._state.image_height = int(height)
            return cls.get()
