from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InferenceContract:
    """
    Output of the inference stage as consumed by eval and downstream stages.

    Canonical artifacts:
      - predictions/: saved logits/probabilities
    """

    inference_dir_path: Path
    task: str

    model_path: Path

    num_scenes: int

    inference_cfg: dict[str, Any]

    prediction_table_path: Path
