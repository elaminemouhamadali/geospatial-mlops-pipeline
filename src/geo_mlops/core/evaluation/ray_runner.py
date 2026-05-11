# src/geo_mlops/core/evaluation/ray_runner.py

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from geo_mlops.core.contracts.eval_contract import EvalContract
from geo_mlops.core.evaluation.types import EvalConfig
from geo_mlops.core.execution.ray_backend import require_ray_initialized
from geo_mlops.core.execution.sharding import shard_sequence
from geo_mlops.core.io.eval_io import write_eval_contract
from geo_mlops.core.utils.dataclasses import _as_plain_dict


def run_prediction_evaluation_ray(
    *,
    task: str,
    prediction_table_path: Path,
    out_dir: Path,
    eval_cfg: EvalConfig | dict[str, Any],
    build_metric_accumulator_fn: Callable[[dict[str, Any]], Any],
    build_prediction_evaluator_fn: Callable[[dict[str, Any], Any], Callable[[dict[str, Any]], dict[str, Any]]],
    num_workers: int | None = None,
    items_per_shard: int | None = None,
    num_cpus_per_worker: int = 4,
) -> tuple[Path, EvalContract]:
    """
    Distributed prediction scoring.

    Core responsibilities:
      - read merged inference prediction inventory
      - shard prediction records
      - run task-specific row evaluation inside Ray workers
      - write one shard per-scene metrics CSV per worker
      - merge shard per-scene metrics
      - rebuild/finalize the global metric accumulator once on the driver
      - write final metrics.json and EvalContract

    This runner intentionally does not know about task plugins.
    The CLI/task layer supplies metric accumulator and row evaluator factories.
    """
    import ray

    require_ray_initialized()

    prediction_table_path = Path(prediction_table_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    shard_dir = out_dir / "_ray_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(prediction_table_path)
    if df.empty:
        raise ValueError(f"Prediction table is empty: {prediction_table_path}")

    records = df.to_dict(orient="records")

    record_shards = shard_sequence(
        records,
        num_shards=num_workers,
        items_per_shard=items_per_shard,
        drop_empty=True,
    )

    if not record_shards:
        raise ValueError("Prediction record sharding produced no shards.")

    eval_cfg_dict = _as_plain_dict(eval_cfg)

    remote_worker = ray.remote(
        num_cpus=num_cpus_per_worker,
        num_gpus=0,
    )(_run_evaluation_shard)

    refs = []
    for shard_idx, shard_records in enumerate(record_shards):
        refs.append(
            remote_worker.remote(
                shard_idx=shard_idx,
                records=shard_records,
                eval_cfg=eval_cfg_dict,
                build_metric_accumulator_fn=build_metric_accumulator_fn,
                build_prediction_evaluator_fn=build_prediction_evaluator_fn,
                shard_dir=shard_dir,
            )
        )

    shard_table_paths = ray.get(refs)

    shard_dfs = []
    for shard_table_path in shard_table_paths:
        shard_table_path = Path(shard_table_path)
        if shard_table_path.exists():
            shard_df = pd.read_csv(shard_table_path)
            if not shard_df.empty:
                shard_dfs.append(shard_df)

    if not shard_dfs:
        raise ValueError("Distributed evaluation produced no per-scene metric rows.")

    merged_df = pd.concat(shard_dfs, ignore_index=True)

    per_scene_table_path = tables_dir / "per_scene_metrics.csv"
    merged_df.to_csv(per_scene_table_path, index=False)

    per_scene_rows = merged_df.to_dict(orient="records")

    metric_accumulator = build_metric_accumulator_fn(eval_cfg_dict)

    if not hasattr(metric_accumulator, "ingest_rows"):
        raise AttributeError("Distributed evaluation requires the metric accumulator to implement ingest_rows(rows: list[dict[str, Any]]).")

    metric_accumulator.ingest_rows(per_scene_rows)

    task_summary = metric_accumulator.finalize(out_dir=out_dir)

    metrics = _as_plain_dict(task_summary.get("metrics", {}))
    artifacts = _as_plain_dict(task_summary.get("artifacts", {}))
    analytics = _as_plain_dict(task_summary.get("analytics", {}))

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    artifacts_path = out_dir / "artifacts.json"
    artifacts_path.write_text(json.dumps(artifacts, indent=2))

    analytics_path = out_dir / "analytics.json"
    analytics_path.write_text(json.dumps(analytics, indent=2))

    contract = EvalContract(
        eval_dir_path=out_dir,
        task=task,
        metrics_path=metrics_path,
        eval_cfg=eval_cfg_dict,
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


def _run_evaluation_shard(
    *,
    shard_idx: int,
    records: list[dict[str, Any]],
    eval_cfg: dict[str, Any],
    build_metric_accumulator_fn: Callable[[dict[str, Any]], Any],
    build_prediction_evaluator_fn: Callable[[dict[str, Any], Any], Callable[[dict[str, Any]], dict[str, Any]]],
    shard_dir: Path,
) -> str:
    """
    Ray worker.

    Writes only this shard's per-scene metric table.
    The driver writes the final merged table, metrics.json, analytics, artifacts,
    and EvalContract.
    """
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    metric_accumulator = build_metric_accumulator_fn(eval_cfg)
    evaluate_row_fn = build_prediction_evaluator_fn(eval_cfg, metric_accumulator)

    per_scene_rows: list[dict[str, Any]] = []

    for record in records:
        per_scene_rows.append(evaluate_row_fn(record))

    shard_table_path = shard_dir / f"shard_{shard_idx:04d}_per_scene_metrics.csv"
    pd.DataFrame(per_scene_rows).to_csv(shard_table_path, index=False)

    return str(shard_table_path)
