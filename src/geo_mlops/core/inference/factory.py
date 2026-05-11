from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from geo_mlops.core.config.loader import load_cfg, require_section
from geo_mlops.core.data.types import DatasetLayout
from geo_mlops.core.inference.types import InferenceConfig


def build_inference_cfg(task_cfg_path: str | Path) -> dict[str, Any]:
    cfg = load_cfg(task_cfg_path)

    return {
        "inference": require_section(cfg, "inference"),
        "tiling": require_section(cfg, "tiling"),
        "training": require_section(cfg, "training"),
    }


def resolve_inference_layout(inference_cfg: dict[str, Any]) -> DatasetLayout:
    tiling_engine_cfg = inference_cfg.get("tiling").get("engine")

    return DatasetLayout(
        pan_dirname=str(tiling_engine_cfg.get("pan_dirname")),
        gt_dirname=tiling_engine_cfg.get("gt_dirname"),
        context_dirname=tiling_engine_cfg.get("context_dirname"),
        preds_dirname=tiling_engine_cfg.get("preds_dirname", None),
    )


def resolve_inference_data_cfg(inference_cfg: dict[str, Any]) -> dict[str, Any]:
    training_dataset_cfg = inference_cfg.get("training").get("dataset")
    training_loss_cfg = inference_cfg.get("training").get("loss")

    return {
        "reflectance_max": training_dataset_cfg.get("reflectance_max"),
        "use_context": training_dataset_cfg.get("use_context"),
        "tile_out_channels": training_dataset_cfg.get("tile_out_channels"),
        "context_out_channels": training_dataset_cfg.get("context_out_channels", 1),
        "foreground_label": training_loss_cfg.get("foreground_label"),
    }


def build_inference_engine_cfg(inference_cfg: dict[str, Any]) -> InferenceConfig:
    engine_cfg = inference_cfg["inference"].get("engine")

    return InferenceConfig(
        tile_size=int(engine_cfg.get("tile_size")),
        stride=int(engine_cfg.get("stride")),
        batch_size=int(engine_cfg.get("batch_size")),
        seed=int(engine_cfg.get("seed")),
        save_logits=bool(engine_cfg.get("save_logits")),
        save_probabilities=bool(engine_cfg.get("save_probabilities")),
    )


def load_checkpoint(
    *,
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    state = torch.load(Path(checkpoint_path), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
