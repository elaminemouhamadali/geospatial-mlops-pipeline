from __future__ import annotations

import json
from pathlib import Path

from geo_mlops.core.contracts.train_contract import (
    TrainContract,
)
from geo_mlops.core.utils.dataclasses import _as_plain_dict, _load_json, _to_jsonable

TRAIN_MANIFEST_NAME = "train_manifest.json"
METRICS_MANIFEST_NAME = "metrics.json"


def write_train_contract(contract: TrainContract, manifest_name: str = TRAIN_MANIFEST_NAME) -> Path:
    """
    Write train_manifest.json for a completed training stage.
    """
    contract.train_dir_path.mkdir(parents=True, exist_ok=True)
    manifest_path = contract.train_dir_path / manifest_name

    payload = _as_plain_dict(_to_jsonable(contract))

    manifest_path.write_text(json.dumps(payload, indent=2))
    return manifest_path


def load_train_contract(manifest_path: Path) -> TrainContract:
    """
    Load a TrainContract from either:
      - the training output directory, or
      - the train_manifest.json path directly.
    """

    data = _load_json(manifest_path)

    return TrainContract(
        train_dir_path=Path(data["train_dir_path"]),
        task=data["task"],
        # tiles_manifest_path=Path(data["tiles_manifest_path"]),
        # split_manifest_path=Path(data["split_manifest_path"]),
        # task_cfg_path=Path(data["task_cfg_path"]),
        model_path=Path(data["model_path"]),
        metrics_path=Path(data["metrics_path"]),
        num_train_tiles=int(data["num_train_tiles"]),
        num_val_tiles=int(data["num_val_tiles"]),
        train_cfg=dict(data["train_cfg"]),
        best_metric_value=data.get("best_metric_value"),
        best_epoch=data.get("best_epoch"),
        tracking=dict(data.get("tracking", {})),
    )
