from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class InferenceConfig:
    tile_size: int = 512
    stride: int = 256
    batch_size: int = 4
    seed: int = 1337
    save_logits: bool = True
    save_probabilities: bool = True


@dataclass(frozen=True)
class InferencePrediction:
    """
    Task postprocessor output for a batch of windows.

    probability:
      torch.Tensor [B,H,W] or [B,C,H,W], task-defined.

    mask:
      torch.Tensor [B,H,W] or [B,C,H,W], task-defined.

    extra:
      Optional task-specific metadata.
    """

    logits: torch.Tensor
    probability: torch.Tensor
    extra: dict[str, Any] | None = None


@dataclass(frozen=True)
class InferenceArtifacts:
    """
    Task postprocessor output for a batch of windows.

    probability:
      torch.Tensor [B,H,W] or [B,C,H,W], task-defined.

    mask:
      torch.Tensor [B,H,W] or [B,C,H,W], task-defined.

    extra:
      Optional task-specific metadata.
    """

    logits_path: Path
    probability_path: Path
