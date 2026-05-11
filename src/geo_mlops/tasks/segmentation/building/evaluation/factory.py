from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from geo_mlops.tasks.segmentation.building.modeling.metrics import (
    BuildingSegmentationEvalAccumulator,
)


def build_eval_metric_accumulator(eval_cfg: dict[str, Any]):
    return BuildingSegmentationEvalAccumulator(eval_cfg)


def build_prediction_evaluator(
    eval_cfg: dict[str, Any],
    metric_accumulator: BuildingSegmentationEvalAccumulator,
):
    threshold = eval_cfg["threshold"]
    foreground_label = eval_cfg["foreground_label"]

    def evaluate_row(record: dict[str, Any]) -> dict[str, Any]:
        probability_path = Path(record["probability_path"])
        gt_path = Path(record["gt_path"])

        if not probability_path.exists():
            raise FileNotFoundError(f"Missing probability raster: {probability_path}")
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing GT raster: {gt_path}")

        with rasterio.open(probability_path) as src:
            prob = src.read(1).astype(np.float32)

        with rasterio.open(gt_path) as src:
            gt = src.read(1).astype(np.int64)

        target = gt == int(foreground_label)
        mask = prob >= threshold

        row_metrics = metric_accumulator.update_from_arrays(
            scene_id=record["scene_id"],
            roi=record["region"],
            sub_roi=record["subregion"],
            target=target,
            probability=prob,
            mask=mask,
            metadata=record,
        )

        return {
            "scene_id": record["scene_id"],
            "roi": record.get("region", ""),
            "sub_roi": record.get("subregion", ""),
            "gt_path": str(gt_path),
            "probability_path": str(probability_path),
            **row_metrics,
        }

    return evaluate_row
