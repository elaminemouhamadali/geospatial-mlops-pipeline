# geo_mlops/core/datasets/types.py

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class DatasetLayout:
    pan_dirname: str = "PAN"
    gt_dirname: str | None = "GT-Mask"
    context_dirname: str | None = "Context"
    preds_dirname: str | None = None


@dataclass(frozen=True)
class DiscoveredScene:
    region: str
    subregion: str
    stem: str
    scene_id: str

    pan_path: Path
    gt_path: Path | None = None
    context_path: Path | None = None
    pred_path: Path | None = None

    meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class SceneArrays:
    """
    Task-provided full-scene arrays.

    image:
      torch.Tensor [C,H,W], float32, model-ready.

    target:
      Optional task-specific target. For segmentation this is often [H,W].

    context:
      Optional context tensor [C,H,W] or lower-res [C,h,w].

    profile:
      Optional raster profile/metadata passed back to the task writer.
    """

    image: torch.Tensor
    target: Any | None = None
    context: torch.Tensor | None = None
    profile: dict[str, Any] | None = None
