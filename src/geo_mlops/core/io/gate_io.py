from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from geo_mlops.core.contracts.gate_contract import GateContract

GATE_CONTRACT_FILENAME = "gate_decision.json"


def _to_jsonable(obj: Any) -> Any:
    """
    Recursively convert common Python objects into JSON-serializable values.

    Handles:
      - dataclasses
      - pathlib.Path
      - dict / list / tuple / set
    """
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
    """
    Lightweight validation for required top-level fields.

    Keep this intentionally minimal; schema evolution should remain easy.
    """
    required = {
        "gate_dir",
        "schema_version",
        "gate_name",
        "task",
        "decision",
        "passed",
        "checks",
        "summary",
        "upstream",
        "threshold_spec",
        "meta",
    }
    missing = sorted(k for k in required if k not in payload)
    if missing:
        raise ValueError(f"Gate contract missing required field(s): {missing}")

    if payload["decision"] not in {"pass", "fail"}:
        raise ValueError(f"Invalid gate decision '{payload['decision']}'. Expected 'pass' or 'fail'.")

    if not isinstance(payload["passed"], bool):
        raise TypeError(f"'passed' must be bool, got {type(payload['passed'])}.")

    if not isinstance(payload["checks"], list):
        raise TypeError(f"'checks' must be a list, got {type(payload['checks'])}.")

    for field in ("summary", "upstream", "threshold_spec", "meta"):
        if not isinstance(payload[field], dict):
            raise TypeError(f"'{field}' must be a dict, got {type(payload[field])}.")


def gate_contract_path(gate_dir: str | Path) -> Path:
    """
    Return the canonical contract path for a gating output directory.
    """
    return _ensure_path(gate_dir) / GATE_CONTRACT_FILENAME


def write_gate_contract(
    contract: GateContract,
    *,
    out_path: str | Path | None = None,
    indent: int = 2,
) -> Path:
    """
    Write the canonical gate contract JSON to disk.

    By default writes:
        <contract.gate_dir>/gate_decision.json
    """
    if not isinstance(contract, GateContract):
        raise TypeError(f"Expected GateContract, got {type(contract)}")

    payload = _to_jsonable(contract)
    _validate_payload(payload)

    path = _ensure_path(out_path) if out_path is not None else gate_contract_path(contract.gate_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=indent, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_gate_contract(path_or_dir: str | Path) -> GateContract:
    """
    Load a gate contract from either:
      - the JSON file itself, or
      - a gate output directory containing gate_decision.json
    """
    p = _ensure_path(path_or_dir)
    path = p if p.is_file() else gate_contract_path(p)

    if not path.exists():
        raise FileNotFoundError(f"Gate contract not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    _validate_payload(payload)

    payload["gate_dir"] = _ensure_path(payload["gate_dir"])
    return GateContract(**payload)


def gate_passed(path_or_dir: str | Path) -> bool:
    """
    Convenience helper for downstream stages.
    """
    return load_gate_contract(path_or_dir).passed


def summarize_gate_contract(contract: GateContract) -> Dict[str, Any]:
    """
    Small convenience summary for logging / CLI output.
    """
    total_checks = len(contract.checks)
    passed_checks = sum(1 for c in contract.checks if bool(c.get("passed", False)))
    failed_checks = total_checks - passed_checks

    return {
        "gate_name": contract.gate_name,
        "task": contract.task,
        "decision": contract.decision,
        "passed": contract.passed,
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
        "gate_dir": str(contract.gate_dir),
    }