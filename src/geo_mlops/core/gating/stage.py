from collections.abc import Mapping
from pathlib import Path

from geo_mlops.core.config.loader import load_cfg
from geo_mlops.core.contracts.gate_contract import GateContract
from geo_mlops.core.gating.engine import run_gate


def run_gate_stage(
    *,
    task: str,
    gate_name: str,
    task_cfg_path: str | Path,
    metrics_file_path: str | Path,
    gate_dir_path: str | Path,
) -> GateContract:
    """
    Stage wrapper for gate evaluation.

    Loads:
      - task config from task_cfg_path
      - metrics payload from metrics_file

    Extracts:
      - task_cfg["gating"][gate_name]

    Then:
      - runs gate evaluation
      - writes gate_decision.json
      - returns GateContract
    """

    task_cfg = load_cfg(task_cfg_path)

    gating = task_cfg.get("gating")
    if not isinstance(gating, Mapping):
        raise ValueError("Task config must include a 'gating' mapping.")

    threshold_spec = gating.get(gate_name)
    if not isinstance(threshold_spec, Mapping):
        raise ValueError(f"Task config missing gating.{gate_name!r}. Available gates: {sorted(gating.keys())}")

    metrics_payload = load_cfg(metrics_file_path)

    metrics = metrics_payload.get("best_metrics", metrics_payload)

    manifest_path, contract = run_gate(
        gate_dir_path=Path(gate_dir_path),
        gate_name=gate_name,
        task=task,
        metrics=metrics,
        threshold_spec=dict(threshold_spec),
    )

    return manifest_path, contract
