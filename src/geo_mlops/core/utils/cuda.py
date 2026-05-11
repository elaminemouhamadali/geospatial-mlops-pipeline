from __future__ import annotations
from typing import Optional, Dict, Any
import torch


def _cuda_free_mem_bytes(device: torch.device) -> Optional[int]:
    """Return free CUDA bytes if available, else None."""
    if device.type != "cuda":
        return None
    free_b, total_b = torch.cuda.mem_get_info(device)
    return int(free_b)


def _heuristic_initial_bs(free_bytes: Optional[int], max_bs: int) -> int:
    """
    Very simple heuristic based on free VRAM. If unknown, default to max_bs.
    Tuned for 512x512 tiles, AMP on. Adjusts conservatively.
    """
    if free_bytes is None:
        return max_bs
    gb = free_bytes / (1024**3)
    if gb < 3.5:
        return min(8, max_bs)
    if gb < 7.0:
        return min(16, max_bs)
    if gb < 11.0:
        return min(32, max_bs)
    return max_bs

def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def _infer_batch_size(batch: Dict[str, Any]) -> int:
    for value in batch.values():
        if torch.is_tensor(value):
            return int(value.shape[0])
    raise ValueError("Could not infer batch size from batch; no tensor values found.")