from __future__ import annotations

import json
import numpy as np
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Tuple
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
    extra: Optional[Dict[str, Any]] = None

  
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