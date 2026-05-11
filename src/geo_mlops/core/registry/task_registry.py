from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Dict, Protocol, runtime_checkable



@runtime_checkable
class TaskPlugin(Protocol):
    # task name
    name: str

    # tiling
    def build_tiling_components(self): ...

    # splitting
    def build_splitting_cfg(self): ...
    def build_split_engine_cfg(self): ...

    # training
    def build_training_cfg(self): ...
    def build_train_engine_cfg(self): ...
    def build_model(self): ...
    def build_loss(self): ...
    def build_metrics_fn(self): ...
    def get_forward_fn(self): ...
    def build_dataset(self): ...
    def build_train_val_datasets(self): ...

    # inference
    def build_inference_cfg(self): ...
    def build_inference_engine_cfg(self): ...
    def resolve_inference_layout(self): ...
    def build_inference_postprocessor(self): ...
    def load_inference_scene(self): ...
    def save_inference_prediction(self): ...
    def load_checkpoint(self): ...

    # evaluation
    def build_evaluation_cfg(self): ...
    def build_prediction_evaluator(self): ...
    def build_eval_metric_accumulator(self): ...
    


@dataclass(frozen=True)
class TaskSpec:
    name: str
    plugin_path: str  # "pkg.module:object_or_factory"


_TASKS: Dict[str, TaskSpec] = {
    "building_seg": TaskSpec(
        name="building_seg",
        plugin_path="geo_mlops.tasks.segmentation.building.task:BuildingSegmentationTask",
    ),
    # Keep commented until implemented with the same plugin interface:
    # "noise_cls": TaskSpec(
    #     name="noise_cls",
    #     plugin_path="geo_mlops.tasks.classification.noise.task:NoiseClassificationTask",
    # ),
}


_PLUGIN_CACHE: Dict[str, TaskPlugin] = {}


def list_tasks() -> list[str]:
    return sorted(_TASKS.keys())


def get_task_spec(task_name: str) -> TaskSpec:
    if task_name not in _TASKS:
        raise KeyError(f"Unknown task {task_name!r}. Available: {list_tasks()}")
    return _TASKS[task_name]


def _load_symbol(path: str):
    mod_name, symbol_name = path.split(":", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, symbol_name)


def get_task(task_name: str) -> TaskPlugin:
    if task_name in _PLUGIN_CACHE:
        return _PLUGIN_CACHE[task_name]

    spec = get_task_spec(task_name)
    symbol = _load_symbol(spec.plugin_path)

    # Supports either:
    #   plugin_path="...:BuildingSegmentationTask"  # class
    # or:
    #   plugin_path="...:TASK"                      # instance
    if isinstance(symbol, type):
        plugin = symbol()
    else:
        plugin = symbol

    if not isinstance(plugin, TaskPlugin):
        raise TypeError(
            f"Task plugin for {task_name!r} does not implement TaskPlugin protocol. "
            f"Loaded object: {plugin!r}"
        )

    if plugin.name != task_name:
        raise ValueError(
            f"Task plugin name mismatch: registry key={task_name!r}, "
            f"plugin.name={plugin.name!r}"
        )

    _PLUGIN_CACHE[task_name] = plugin
    return plugin