from __future__ import annotations

import json
from pathlib import Path

from geo_mlops.core.contracts.gate_contract import GateContract
from geo_mlops.core.utils.dataclasses import _as_plain_dict, _load_json, _to_jsonable

GATE_MANIFEST_NAME = "gate_decision.json"


def write_gate_contract(
    contract: GateContract,
    manifest_name: str = GATE_MANIFEST_NAME,
) -> Path:
    """
    Serialize GateContract to JSON in contract.gate_dir/gate_decision.json.
    """
    contract.gate_dir_path.mkdir(parents=True, exist_ok=True)
    manifest_path = contract.gate_dir_path / manifest_name

    payload = _as_plain_dict(_to_jsonable(contract))

    manifest_path.write_text(json.dumps(payload, indent=2))
    return manifest_path


def load_gate_contract(manifest_path) -> GateContract:
    """
    Load GateContract from gate_dir/gate_decision.json.
    """
    data = _load_json(manifest_path)

    return GateContract(
        gate_dir_path=data["gate_dir_path"],
        gate_name=str(data.get("gate_name")),
        task=str(data.get("task")),
        decision=str(data.get("decision")),
        passed=bool(data.get("passed")),
        checks=list(data.get("checks", [])),
        summary=dict(data.get("summary", {})),
        threshold_spec=dict(data.get("threshold_spec", {})),
    )


def gate_passed(manifest_path: Path) -> bool:
    """
    Convenience helper for downstream stages.
    """
    return load_gate_contract(manifest_path).passed
