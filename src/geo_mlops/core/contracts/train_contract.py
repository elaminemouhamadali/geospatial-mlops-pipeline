from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TrainContract:
    """
    Output of the training stage as consumed by downstream stages.

    This is intentionally task-agnostic. Tasks own model construction,
    dataset construction, loss functions, forward passes, and metric logic.
    Core only records the canonical artifacts and reproducibility metadata.
    """

    # -------------------------
    # Stage identity
    # -------------------------
    task: str
    train_dir_path: Path

    # -------------------------
    # Canonical training artifacts
    # -------------------------
    model_path: Path
    metrics_path: Path

    # -------------------------
    # Dataset sizes actually used
    # -------------------------
    num_train_tiles: int
    num_val_tiles: int

    # -------------------------
    # Task training config snapshot
    # -------------------------
    train_cfg: Dict[str, Any]

    # -------------------------
    # Model selection metadata
    # -------------------------
    best_metric_value: Optional[float]
    best_epoch: Optional[int]

    # -------------------------
    # Optional tracking/callback state
    # -------------------------
    tracking: Dict[str, Any] = field(default_factory=dict)