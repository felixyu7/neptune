"""Lazy JIT loader for the C++ CPU FPS kernels.

The sources under csrc/ compile on first CPU use via
torch.utils.cpp_extension.load (cached in ~/.cache/torch_extensions, or
$TORCH_EXTENSIONS_DIR). No compiler at runtime is fine: get_module()
returns None after warning once, and callers fall back to the pure-torch
reference implementation.
"""
from __future__ import annotations

import threading
import warnings
from pathlib import Path

_lock = threading.Lock()
_module = None
_state = "unloaded"  # unloaded | ready | failed


def _build():
    from torch.utils.cpp_extension import load

    csrc = Path(__file__).resolve().parent / "csrc"
    return load(
        name="neptune_fps_cpu",
        sources=[str(csrc / "fps_cpu.cpp"), str(csrc / "binding_cpu.cpp")],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def get_module():
    """The compiled CPU extension, or None if it cannot be built."""
    global _module, _state
    if _state == "ready":
        return _module
    if _state == "failed":
        return None
    with _lock:
        if _state == "unloaded":
            try:
                _module = _build()
                _state = "ready"
            except Exception as exc:
                _state = "failed"
                warnings.warn(
                    "neptune.fps: C++ CPU extension could not be built "
                    f"({exc}); falling back to the slower pure-PyTorch "
                    "implementation. Install a C++ compiler for fast CPU FPS.",
                    RuntimeWarning,
                    stacklevel=3,
                )
    return _module if _state == "ready" else None
