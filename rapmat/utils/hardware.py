"""Hardware (CPU / CUDA) cached detection.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareInfo:
    torch_installed: bool
    cuda: bool
    device_name: str | None = None


_lock = threading.Lock()
_cache: HardwareInfo | None = None


def detect() -> HardwareInfo:
    """Detect CPU/CUDA, cache the result. Blocking."""
    global _cache
    with _lock:
        if _cache is None:
            _cache = _detect()
        return _cache


def cached() -> HardwareInfo | None:
    """Return the detected info if ready, else ``None``. Never blocks."""
    return _cache


def _detect() -> HardwareInfo:
    try:
        import torch
    except ImportError:
        return HardwareInfo(torch_installed=False, cuda=False)

    if torch.cuda.is_available():
        name = (
            torch.cuda.get_device_name(0)
            if torch.cuda.device_count() > 0
            else "Unknown GPU"
        )
        return HardwareInfo(torch_installed=True, cuda=True, device_name=name)
    return HardwareInfo(torch_installed=True, cuda=False)


def header_markup(info: HardwareInfo | None) -> list[tuple[str, str]]:
    """Markup for the right-aligned header badge."""
    if info is None:
        return [("cpu_tag", " ⏳ ")]
    if not info.torch_installed:
        return [("cpu_tag", " ❌  NO TORCH ")]
    if info.cuda:
        return [("cuda_tag", " ⚡  CUDA ")]
    return [("cpu_tag", " 🖥️  CPU ")]


def home_label(info: HardwareInfo | None) -> str:
    """Longer hardware line for the home screen's DB-info panel."""
    if info is None:
        return "⏳  detecting..."
    if not info.torch_installed:
        return "🖥️  (Torch not installed)"
    if info.cuda:
        return f"⚡  CUDA ({info.device_name})"
    return "🖥️  CPU (No GPU detected)"
