from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalContract:
    """
    Output of the evaluation stage as consumed by gating and downstream stages.

    Canonical artifacts:
      - metrics.json: flat gate-ready metrics
      - tables/per_scene_metrics.csv: per-scene diagnostics
      - predictions/: saved masks/probabilities
    """

    eval_dir_path: Path
    task: str

    metrics_path: Path

    eval_cfg: dict[str, Any]

    metrics: dict[str, float]
    artifacts: dict[str, Any]
    analytics: dict[str, Any]
