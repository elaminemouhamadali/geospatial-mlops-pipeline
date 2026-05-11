from __future__ import annotations

import operator
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from geo_mlops.core.contracts.gate_contract import GateContract
from geo_mlops.core.io.gate_io import write_gate_contract

_COMPARATORS = {
    ">=": operator.ge,
    ">": operator.gt,
    "<=": operator.le,
    "<": operator.lt,
    "==": operator.eq,
    "!=": operator.ne,
}


def _metrics_for_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """
    Return the flat metric dict that gate checks should evaluate.

    Supports:
      1. Flat metrics directly:
         {"val/micro_f1": 0.44, "val/iou": 0.30}

      2. Training metrics payload:
         {
           "best_epoch": 4,
           "best_metric_value": 0.44,
           "selection_metric": "val/micro_f1",
           "history": {
             "epoch_4": {
               "val/micro_f1": 0.44,
               "val/iou": 0.30
             }
           }
         }
    """
    history = metrics.get("history")
    best_epoch = metrics.get("best_epoch")

    if isinstance(history, Mapping) and best_epoch is not None:
        epoch_key = f"epoch_{int(best_epoch)}"
        epoch_metrics = history.get(epoch_key)

        if not isinstance(epoch_metrics, Mapping):
            raise ValueError(f"metrics['history'] missing best epoch key {epoch_key!r}")

        out = dict(epoch_metrics)

        if "best_metric_value" in metrics:
            out["best_metric_value"] = metrics["best_metric_value"]

        selection_metric = metrics.get("selection_metric")
        if isinstance(selection_metric, str) and "best_metric_value" in metrics:
            out.setdefault(selection_metric, metrics["best_metric_value"])

        return out

    return dict(metrics)


def _metric_key(check: Mapping[str, Any]) -> str:
    metric = str(check["metric"])
    scope = check.get("scope")

    if scope:
        return f"{scope}/{metric}"

    return metric


def _get_metric(metrics: Mapping[str, Any], key: str) -> float:
    if key not in metrics:
        raise KeyError(f"Metric not found: {key!r}. Available metrics: {sorted(metrics.keys())}")

    value = metrics[key]

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Metric {key!r} must be numeric, got {type(value).__name__}")

    return float(value)


def _evaluate_check(
    *,
    metrics: Mapping[str, Any],
    check: Mapping[str, Any],
) -> dict[str, Any]:
    if "metric" not in check:
        raise ValueError(f"Gate check missing required field 'metric': {check}")

    if "threshold" not in check:
        raise ValueError(f"Gate check missing required field 'threshold': {check}")

    comparator = str(check.get("comparator", ">="))

    if comparator not in _COMPARATORS:
        raise ValueError(f"Unsupported comparator {comparator!r}. Allowed: {sorted(_COMPARATORS.keys())}")

    key = _metric_key(check)
    actual = _get_metric(metrics, key)
    threshold = float(check["threshold"])
    passed = bool(_COMPARATORS[comparator](actual, threshold))

    return {
        "metric": str(check["metric"]),
        "scope": str(check.get("scope", "")),
        "comparator": comparator,
        "threshold": threshold,
        "actual": actual,
        "passed": passed,
    }


def _summarize_checks(checks: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(checks)
    passed = sum(1 for c in checks if c["passed"])
    failed = total - passed

    return {
        "total_checks": total,
        "passed_checks": passed,
        "failed_checks": failed,
        "failed_metrics": [
            {
                "metric": c["metric"],
                "scope": c["scope"],
                "actual": c["actual"],
                "threshold": c["threshold"],
                "comparator": c["comparator"],
            }
            for c in checks
            if not c["passed"]
        ],
    }


def run_gate(
    *,
    gate_dir_path: str | Path,
    gate_name: str,
    task: str,
    metrics: Mapping[str, Any],
    threshold_spec: Mapping[str, Any],
) -> GateContract:
    checks_spec = threshold_spec.get("checks")

    if not isinstance(checks_spec, list) or not checks_spec:
        raise ValueError("threshold_spec['checks'] must be a non-empty list.")

    gate_metrics = _metrics_for_gate(metrics)

    checks = [_evaluate_check(metrics=gate_metrics, check=check) for check in checks_spec]

    summary = _summarize_checks(checks)

    passed = all(check["passed"] for check in checks)
    decision = "pass" if passed else "fail"

    contract = GateContract(
        gate_dir_path=Path(gate_dir_path),
        gate_name=str(gate_name),
        task=str(task),
        decision=decision,
        passed=passed,
        checks=checks,
        summary=summary,
        threshold_spec=dict(threshold_spec),
    )

    manifest_path = write_gate_contract(contract)

    return manifest_path, contract
