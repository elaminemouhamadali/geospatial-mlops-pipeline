from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class EvalConfig:
    threshold: float = 0.5
    foreground_label: int = 1
    eps: int = 0.0000001


@runtime_checkable
class EvalMetricAccumulator(Protocol):
    """
    Task-specific metric/analytics accumulator.

    File loading belongs outside this object.

    The task-specific prediction evaluator reads one prediction_inventory.csv
    record, loads the required artifacts/targets, prepares arrays, and calls
    update_from_arrays(...).

    Examples:
      - segmentation: target mask + probability map + binary mask
      - detection: GT boxes + predicted boxes/scores/classes
      - classification: label + predicted logits/probabilities
      - SSL/retrieval: labels/metadata + embeddings/rankings
    """

    def update_from_arrays(
        self,
        *,
        scene_id: str,
        roi: str,
        sub_roi: str,
        target: Any,
        prediction: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Update internal metric state from one already-loaded prediction item.

        For building segmentation:
          target     = GT boolean mask or label mask
          prediction = {
              "probability": np.ndarray,
              "mask": np.ndarray,
          }

        Returns one per-scene/per-item metrics row.
        """
        ...

    def finalize(self, *, out_dir: Path) -> dict[str, Any]:
        """
        Finalize metrics and optional analytics.

        Returns:
          {
            "metrics": {
              "micro/f1": ...,
              "scene_macro/f1": ...,
              "roi_macro/f1": ...
            },
            "artifacts": {
              "building_per_image_metrics_csv": "...",
              "building_per_roi_metrics_csv": "...",
              "building_pareto_images_csv": "..."
            },
            "analytics": {
              ...
            }
          }
        """
        ...
