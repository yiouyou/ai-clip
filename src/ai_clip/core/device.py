"""CPU/GPU detection so the same code runs on a laptop or a GPU box."""

from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def has_cuda() -> bool:
    try:
        import torch  # noqa: PLC0415

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def whisper_runtime(model_size: str, compute_type: str) -> tuple[str, str]:
    """Resolve (device, compute_type) for faster-whisper.

    compute_type="auto" -> float16 on CUDA, int8 on CPU (best CPU throughput).
    """
    device = "cuda" if has_cuda() else "cpu"
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type
