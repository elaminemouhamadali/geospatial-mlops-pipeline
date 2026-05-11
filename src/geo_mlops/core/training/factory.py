from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from geo_mlops.core.training.engine import (
    TrainConfig
)

from geo_mlops.core.config.loader import load_cfg, require_section


   
def build_training_cfg(task_cfg_path: str | Path) -> Dict[str, Any]:
    cfg = load_cfg(task_cfg_path)
    return require_section(cfg, "training")


def build_train_engine_cfg(train_cfg: Dict[str, Any]) -> TrainConfig:
    engine_cfg = train_cfg.get("engine", {}) or {}

    return TrainConfig(
        batch_size=int(engine_cfg.get("batch_size", 8)),
        num_workers=int(engine_cfg.get("num_workers", 4)),
        epochs=int(engine_cfg.get("epochs", 5)),
        lr=float(engine_cfg.get("lr", 3e-4)),
        seed=int(engine_cfg.get("seed", 1337)),
        selection_metric=str(engine_cfg.get("selection_metric", "val/loss")),
        selection_mode=str(engine_cfg.get("selection_mode", "min")),
    )
    