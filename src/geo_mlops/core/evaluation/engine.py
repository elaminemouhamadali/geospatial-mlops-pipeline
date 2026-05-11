from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from geo_mlops.core.contracts.eval_contract import EvalContract
from geo_mlops.core.evaluation.types import EvalConfig
from geo_mlops.core.io.eval_io import write_eval_contract
from geo_mlops.core.utils.dataclasses import _as_plain_dict


def run_prediction_evaluation(
    *,
    task: str,
    prediction_table_path: Path,
    out_dir: Path,
    evaluate_prediction_row_fn,
    metric_accumulator,
    eval_cfg_raw: EvalConfig | None = None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(prediction_table_path)
    if df.empty:
        raise ValueError(f"Prediction table is empty: {prediction_table_path}")

    per_scene_rows = []

    for record in df.to_dict(orient="records"):
        row = evaluate_prediction_row_fn(record)
        per_scene_rows.append(row)

    # Generic core table.
    per_scene_table_path = tables_dir / "per_scene_metrics.csv"
    pd.DataFrame(per_scene_rows).to_csv(per_scene_table_path, index=False)

    # Task-specific accumulator outputs.
    task_summary = metric_accumulator.finalize(out_dir=out_dir)

    metrics = _as_plain_dict(task_summary.get("metrics", {}))
    artifacts = _as_plain_dict(task_summary.get("artifacts", {}))
    analytics = _as_plain_dict(task_summary.get("analytics", {}))

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    contract = EvalContract(
        eval_dir_path=out_dir,
        task=task,
        metrics_path=metrics_path,
        eval_cfg=_as_plain_dict(eval_cfg_raw),
        metrics=metrics,
        artifacts={
            "prediction_table_csv": str(prediction_table_path),
            "per_scene_metrics_csv": str(per_scene_table_path),
            **artifacts,
        },
        analytics=analytics,
    )

    manifest_path = write_eval_contract(contract)

    return manifest_path, contract
