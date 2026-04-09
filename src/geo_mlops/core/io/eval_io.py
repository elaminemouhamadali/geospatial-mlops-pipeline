from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

from geo_mlops.core.contracts.eval_contract import EvalContract

EVAL_CONTRACT_FILENAME = "eval_manifest.json"


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]

    return obj


def _ensure_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _validate_payload(payload: Dict[str, Any]) -> None:
    required = {
        "eval_dir",
        "schema_version",
        "task",
        "split_name",
        "metrics_path",
        "model_path",
        "num_eval_tiles",
        "group_col",
        "selection_source",
        "metrics",
        "upstream",
        "meta",
    }
    missing = sorted(k for k in required if k not in payload)
    if missing:
        raise ValueError(f"Eval contract missing required field(s): {missing}")

    if not isinstance(payload["metrics"], dict):
        raise TypeError(f"'metrics' must be a dict, got {type(payload['metrics'])}")

    if not isinstance(payload["upstream"], dict):
        raise TypeError(f"'upstream' must be a dict, got {type(payload['upstream'])}")

    if not isinstance(payload["meta"], dict):
        raise TypeError(f"'meta' must be a dict, got {type(payload['meta'])}")

    if not isinstance(payload["num_eval_tiles"], int):
        raise TypeError(f"'num_eval_tiles' must be an int, got {type(payload['num_eval_tiles'])}")


def eval_contract_path(eval_dir: str | Path) -> Path:
    return _ensure_path(eval_dir) / EVAL_CONTRACT_FILENAME


def write_eval_contract(
    contract: EvalContract,
    *,
    out_path: str | Path | None = None,
    indent: int = 2,
) -> Path:
    if not isinstance(contract, EvalContract):
        raise TypeError(f"Expected EvalContract, got {type(contract)}")

    payload = _to_jsonable(contract)
    _validate_payload(payload)

    path = _ensure_path(out_path) if out_path is not None else eval_contract_path(contract.eval_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=indent, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_eval_contract(path_or_dir: str | Path) -> EvalContract:
    p = _ensure_path(path_or_dir)
    path = p if p.is_file() else eval_contract_path(p)

    if not path.exists():
        raise FileNotFoundError(f"Eval contract not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    _validate_payload(payload)

    payload["eval_dir"] = _ensure_path(payload["eval_dir"])
    payload["metrics_path"] = _ensure_path(payload["metrics_path"])
    payload["model_path"] = _ensure_path(payload["model_path"])
    payload["selection_source"] = (
        _ensure_path(payload["selection_source"])
        if payload["selection_source"] is not None
        else None
    )

    return EvalContract(**payload)


def summarize_eval_contract(contract: EvalContract) -> Dict[str, Any]:
    return {
        "task": contract.task,
        "split_name": contract.split_name,
        "num_eval_tiles": contract.num_eval_tiles,
        "model_path": str(contract.model_path),
        "metrics_path": str(contract.metrics_path),
        "eval_dir": str(contract.eval_dir),
        "metric_keys": sorted(contract.metrics.keys()),
    }