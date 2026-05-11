from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence
from datetime import datetime

from geo_mlops.core.utils.cuda import _resolve_device
from geo_mlops.core.io.split_io import load_split_contract
from geo_mlops.core.io.tile_io import load_tiles_contract
from geo_mlops.core.registry.task_registry import get_task
from geo_mlops.core.training.engine import train_one_run
from geo_mlops.core.training.mlflow_callbacks import MLflowTrainingCallback


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Train a task model from tiling + split contracts."
    )

    ap.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task key registered in task_registry, e.g. building_seg.",
    )
    ap.add_argument(
        "--task-cfg-path",
        "--task_cfg_path",
        dest="task_cfg_path",
        type=Path,
        required=True,
        help="Unified task config YAML/JSON containing a `training:` section.",
    )
    ap.add_argument(
        "--tiles-manifest-path",
        "--tiles_manifest_path",
        dest="tiles_manifest_path",
        type=Path,
        required=True,
        help="Tiling output manifest tiles_manifest.json.",
    )
    ap.add_argument(
        "--split-manifest-path",
        "--split_manifest_path",
        dest="split_manifest_path",
        type=Path,
        required=True,
        help="Split manifest split.json.",
    )
    ap.add_argument(
        "--train-dir-path",
        "--train_dir_path",
        dest="train_dir_path",
        type=Path,
        required=True,
        help="Output directory for training artifacts.",
    )

    # Optional runtime overrides. If omitted, values come from training.engine.
    ap.add_argument("--device", type=str, default="cuda")
    
    ap.add_argument("--mlflow", action="store_true", help="Enable MLflow logging.")
    ap.add_argument("--mlflow-tracking-uri", type=str, default=None)
    ap.add_argument("--mlflow-experiment", type=str, default=None)

    return ap


def default_run_name(
    task: str,
    stage: str = "train",
    cfg_name: str | None = None,
) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [task, stage]
    if cfg_name:
        parts.append(cfg_name)
    parts.append(ts)
    return "/".join(parts)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    args.train_dir_path.mkdir(parents=True, exist_ok=True)

    task_plugin = get_task(args.task)

    # -------------------------------------------------------------------------
    # Load stage contracts
    # -------------------------------------------------------------------------
    tiles_contract = load_tiles_contract(args.tiles_manifest_path)
    split_contract = load_split_contract(args.split_manifest_path)

    if tiles_contract.task != args.task:
        raise ValueError(
            f"Task mismatch: --task={args.task!r}, tiles.task={tiles_contract.task!r}"
        )

    # -------------------------------------------------------------------------
    # Load task training config
    # -------------------------------------------------------------------------
    train_cfg = task_plugin.build_training_cfg(args.task_cfg_path)
    train_engine_cfg = task_plugin.build_train_engine_cfg(train_cfg)

    # -------------------------------------------------------------------------
    # Build task-specific components through plugin
    # -------------------------------------------------------------------------
    train_ds, val_ds = task_plugin.build_train_val_datasets(
        tiles=tiles_contract,
        split=split_contract,
        train_cfg=train_cfg,
    )

    model = task_plugin.build_model(train_cfg)
    loss_fn = task_plugin.build_loss(train_cfg)
    metrics_fn = task_plugin.build_metrics_fn(train_cfg)
    forward_fn = task_plugin.get_forward_fn()

    device = _resolve_device(args.device)

    callbacks = []

    if args.mlflow:
        run_name = default_run_name(
            task=args.task,
            stage="train",
            cfg_name=args.task_cfg_path.stem,
        )
        callbacks.append(
            MLflowTrainingCallback(
                tracking_uri=args.mlflow_tracking_uri,
                experiment_name=args.mlflow_experiment or args.task,
                run_name=run_name,
                tags={
                    "task": args.task,
                    "stage": "train",
                    "task_cfg": str(args.task_cfg_path),
                    "tiles_manifest_path": str(args.tiles_manifest_path),
                    "split_manifest_path": str(args.split_manifest_path),
                }
            )
        )
    # -------------------------------------------------------------------------
    # Run generic core trainer
    # -------------------------------------------------------------------------
    manifest_path, contract = train_one_run(
        model=model,
        loss_fn=loss_fn,
        train_ds=train_ds,
        val_ds=val_ds,
        train_dir_path=args.train_dir_path,
        device=device,
        engine_cfg=train_engine_cfg,
        forward_fn=forward_fn,
        metrics_fn=metrics_fn,
        callbacks=callbacks,
        task=args.task,
        train_cfg=train_cfg,
    )

    print(f"[train] done")
    print(f"[train] model={contract.model_path}")
    print(f"[train] metrics={contract.metrics_path}")
    print(f"[train] manifest={manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
