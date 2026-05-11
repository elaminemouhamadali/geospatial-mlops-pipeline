from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch


from geo_mlops.core.splitting.factory import (
    build_splitting_cfg,
    build_split_engine_cfg,
)
from geo_mlops.core.training.factory import (
    build_training_cfg,
    build_train_engine_cfg
)
from geo_mlops.core.inference.factory import (
    build_inference_cfg,
    build_inference_engine_cfg,
    resolve_inference_layout,
)
from geo_mlops.core.evaluation.factory import build_evaluation_cfg

from geo_mlops.tasks.segmentation.building.tiling.factory import build_tiling_components
from geo_mlops.tasks.segmentation.building.modeling.losses import build_loss
from geo_mlops.tasks.segmentation.building.modeling.metrics import build_metrics_fn
from geo_mlops.tasks.segmentation.building.modeling.factory import build_model
from geo_mlops.tasks.segmentation.building.modeling.forward import building_forward_fn
from geo_mlops.tasks.segmentation.building.data.factory import (
    build_dataset,
    build_train_val_datasets,
)
from geo_mlops.tasks.segmentation.building.inference.factory import (
    load_inference_scene,
    build_inference_postprocessor,
    save_inference_prediction,
    load_checkpoint,
)
from geo_mlops.tasks.segmentation.building.evaluation.factory import (
    build_prediction_evaluator,
    build_eval_metric_accumulator,
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
    # Splitting
    # ------------------------------------------------------------------
    def build_splitting_cfg(self, task_cfg_path: str | Path):
        return build_splitting_cfg(task_cfg_path)
    
    def build_split_engine_cfg(self, splitting_cfg):
        return build_split_engine_cfg(splitting_cfg)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def build_training_cfg(self, task_cfg_path: str | Path) -> Dict[str, Any]:
        return build_training_cfg(task_cfg_path)
    
    def build_train_engine_cfg(self, train_cfg: Dict[str, Any]):
        return build_train_engine_cfg(train_cfg)

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
    # Full-scene golden inference
    # ------------------------------------------------------------------
    def build_inference_cfg(self, task_cfg_path: str | Path) -> Dict[str, Any]:
        return build_inference_cfg(task_cfg_path)
    
    def build_inference_engine_cfg(self, inference_cfg: Dict[str, Any]):
        return build_inference_engine_cfg(inference_cfg)
    
    def resolve_inference_layout(self, inference_cfg: Dict[str, Any]):
        return resolve_inference_layout(inference_cfg)
    
    def build_inference_postprocessor(self, inference_cfg: Dict[str, Any]):
        return build_inference_postprocessor(inference_cfg)
    
    def load_inference_scene(self, *args, **kwargs):
        return load_inference_scene(*args, **kwargs)

    def save_inference_prediction(self, *args, **kwargs):
        return save_inference_prediction(*args, **kwargs)
    
    def load_checkpoint(self, **kwargs):
        return load_checkpoint(**kwargs)
    
    # ------------------------------------------------------------------
    # Full-scene golden evaluation
    # ------------------------------------------------------------------
    def build_evaluation_cfg(self, task_cfg_path: str | Path) -> Dict[str, Any]:
        return build_evaluation_cfg(task_cfg_path)

    def build_prediction_evaluator(self, *args, **kwargs):
        return build_prediction_evaluator(*args, **kwargs)

    def build_eval_metric_accumulator(self, eval_cfg: Dict[str, Any]):
        return build_eval_metric_accumulator(eval_cfg)

    