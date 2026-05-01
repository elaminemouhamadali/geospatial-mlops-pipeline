from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch

from geo_mlops.core.config.loader import load_cfg, require_section
from geo_mlops.tasks.segmentation.building.evaluation.eval import (
    build_eval_engine_cfg,
    build_evaluation_cfg,
    build_eval_metric_accumulator,
    build_eval_postprocessor,
    iter_eval_scenes,
    load_checkpoint,
    load_eval_scene,
    save_eval_prediction,
)
from geo_mlops.tasks.segmentation.building.modeling.losses import build_loss
from geo_mlops.tasks.segmentation.building.modeling.metrics import build_metrics_fn
from geo_mlops.tasks.segmentation.building.modeling.factory import build_model
from geo_mlops.tasks.segmentation.building.modeling.forward import building_forward_fn
from geo_mlops.tasks.segmentation.building.tiling.factory import build_tiling_components
from geo_mlops.tasks.segmentation.building.data.train_data import (
    build_dataset,
    build_train_val_datasets,
)


@dataclass(frozen=True)
class BuildingSegmentationTask:
    name: str = "building_seg"

    # ------------------------------------------------------------------
    # Tiling
    # ------------------------------------------------------------------
    def build_tiling_components(self, task_cfg_path: str | Path):
        return build_tiling_components(task_cfg_path)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def build_training_cfg(self, task_cfg_path: str | Path) -> Dict[str, Any]:
        cfg = load_cfg(task_cfg_path)
        return require_section(cfg, "training")

    def build_model(self, train_cfg: Dict[str, Any]) -> torch.nn.Module:
        return build_model(train_cfg)

    def build_loss(self, train_cfg: Dict[str, Any]):
        return build_loss(train_cfg)

    def build_metrics_fn(self, train_cfg: Dict[str, Any]):
        return build_metrics_fn(train_cfg)

    def get_forward_fn(self):
       return building_forward_fn

    def build_dataset(self, **kwargs):
        return build_dataset(**kwargs)

    def build_train_val_datasets(self, **kwargs):
        return build_train_val_datasets(**kwargs)

    # ------------------------------------------------------------------
    # Full-scene golden evaluation
    # ------------------------------------------------------------------
    def build_evaluation_cfg(self, task_cfg_path: str | Path) -> Dict[str, Any]:
        return build_evaluation_cfg(task_cfg_path)

    def build_eval_engine_cfg(self, eval_cfg: Dict[str, Any]):
        return build_eval_engine_cfg(eval_cfg)

    def iter_eval_scenes(self, **kwargs):
        return iter_eval_scenes(**kwargs)

    def load_eval_scene(self, *args, **kwargs):
        return load_eval_scene(*args, **kwargs)

    def build_eval_postprocessor(self, eval_cfg: Dict[str, Any]):
        return build_eval_postprocessor(eval_cfg)

    def save_eval_prediction(self, *args, **kwargs):
        return save_eval_prediction(*args, **kwargs)

    def build_eval_metric_accumulator(self, eval_cfg: Dict[str, Any]):
        return build_eval_metric_accumulator(eval_cfg)

    def load_checkpoint(self, **kwargs):
        return load_checkpoint(**kwargs)