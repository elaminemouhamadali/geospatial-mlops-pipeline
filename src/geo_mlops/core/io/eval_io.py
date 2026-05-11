from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from geo_mlops.core.utils.dataclasses import _as_plain_dict, _load_json, _to_jsonable
from geo_mlops.core.contracts.eval_contract import (
    EvalContract,
)
EVAL_MANIFEST_NAME = "eval_manifest.json"


def write_eval_contract(
    contract: EvalContract,
    manifest_name: str = EVAL_MANIFEST_NAME,
) -> Path:
    """
    Serialize EvalContract to JSON in contract.eval_dir/eval_manifest.json.
    """
    contract.eval_dir_path.mkdir(parents=True, exist_ok=True)
    manifest_path = contract.eval_dir_path / manifest_name

    payload = _as_plain_dict(_to_jsonable(contract))

    manifest_path.write_text(json.dumps(payload, indent=2))
    return manifest_path


def load_eval_contract(
    manifest_path: Path
) -> EvalContract:
    """
    Load EvalContract from eval_dir/eval_manifest.json.
    """
    
    data = _load_json(manifest_path)

    return EvalContract(
        eval_dir_path=data["eval_dir_path"],
        task=str(data.get("task")),
        metrics_path=Path(data.get("metrics_path")),
        eval_cfg=dict(data.get("eval_cfg")),
        metrics=dict(data.get("metrics")),
        artifacts=dict(data.get("artifacts")),
    )