from __future__ import annotations

import json
from pathlib import Path

from geo_mlops.core.contracts.inference_contract import (
    InferenceContract,
)
from geo_mlops.core.utils.dataclasses import _as_plain_dict, _load_json, _to_jsonable

INFERENCE_MANIFEST_NAME = "inference_manifest.json"


def write_inference_contract(
    contract: InferenceContract,
    manifest_name: str = INFERENCE_MANIFEST_NAME,
) -> Path:
    """
    Serialize InferenceContract to JSON in contract.inference_manifest.json.
    """
    contract.inference_dir_path.mkdir(parents=True, exist_ok=True)
    manifest_path = contract.inference_dir_path / manifest_name

    payload = _as_plain_dict(_to_jsonable(contract))

    manifest_path.write_text(json.dumps(payload, indent=2))
    return manifest_path


def load_inference_contract(manifest_path: Path) -> InferenceContract:
    """
    Load InferenceContract from inference_manifest.json.
    """

    data = _load_json(manifest_path)

    return InferenceContract(
        inference_dir_path=data["inference_dir_path"],
        task=str(data.get("task")),
        model_path=Path(data.get("model_path")),
        num_scenes=int(data.get("num_scenes")),
        inference_cfg=dict(data.get("inference_cfg")),
        prediction_table_path=Path(data.get("prediction_table_path")),
    )
