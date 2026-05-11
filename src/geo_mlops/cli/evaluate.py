from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from geo_mlops.core.evaluation.engine import run_prediction_evaluation
from geo_mlops.core.io.inference_io import load_inference_contract
from geo_mlops.core.registry.task_registry import get_task


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=("Run full-scene sliding-window evaluation on a golden dataset using a trained task model."))
    p.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task key registered in task_registry, e.g. building_seg.",
    )
    p.add_argument(
        "--task-cfg-path",
        "--task_cfg_path",
        dest="task_cfg_path",
        type=Path,
        required=True,
        help="Unified task config YAML/JSON containing training/evaluation sections.",
    )
    p.add_argument(
        "--inference-manifest-path",
        "--inference_manifest_path",
        dest="inference_manifest_path",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--eval-dir-path",
        "--eval_dir_path",
        dest="eval_dir_path",
        type=Path,
        required=True,
        help="Output directory for eval_summary.json, eval_manifest.json",
    )

    p.add_argument("--execution-backend", choices=["local", "ray"], default="local")

    p.add_argument("--ray-address", type=str, default=None)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--items-per-shard", "--items_per_shard", dest="items_per_shard", type=int, default=None)
    p.add_argument("--num-gpus-per-worker", type=float, default=1.0)
    p.add_argument("--num-cpus-per-worker", type=int, default=4)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    args.eval_dir_path.mkdir(parents=True, exist_ok=True)

    task_plugin = get_task(args.task)

    inference_contract = load_inference_contract(args.inference_manifest_path)

    eval_cfg = task_plugin.build_evaluation_cfg(args.task_cfg_path)
    metric_accumulator = task_plugin.build_eval_metric_accumulator(eval_cfg)

    evaluate_row_fn = task_plugin.build_prediction_evaluator(
        eval_cfg=eval_cfg,
        metric_accumulator=metric_accumulator,
    )

    if args.execution_backend == "ray":
        from geo_mlops.core.evaluation.ray_runner import run_prediction_evaluation_ray
        from geo_mlops.core.execution.ray_backend import (
            RayBackendConfig,
            init_ray_backend,
            shutdown_ray_backend,
        )

        ray_cfg = RayBackendConfig(
            address=args.ray_address,
            namespace="geo-mlops",
        )

        init_ray_backend(ray_cfg)

        try:
            manifest_path, contract = run_prediction_evaluation_ray(
                task=args.task,
                prediction_table_path=inference_contract.prediction_table_path,
                out_dir=args.eval_dir_path,
                eval_cfg=eval_cfg,
                build_metric_accumulator_fn=task_plugin.build_eval_metric_accumulator,
                build_prediction_evaluator_fn=task_plugin.build_prediction_evaluator,
                num_workers=args.num_workers,
                items_per_shard=args.items_per_shard,
                num_cpus_per_worker=args.num_cpus_per_worker,
            )
        finally:
            shutdown_ray_backend(ray_cfg)

    else:
        manifest_path, contract = run_prediction_evaluation(
            task=args.task,
            prediction_table_path=inference_contract.prediction_table_path,
            out_dir=args.eval_dir_path,
            evaluate_prediction_row_fn=evaluate_row_fn,
            metric_accumulator=metric_accumulator,
            eval_cfg_raw=eval_cfg,
        )

    print("[evaluate] done")
    print(f"[evaluate] manifest={manifest_path}")
    print(f"[evaluate] metrics={contract.metrics}")
    print(f"[evaluate] artifacts={contract.artifacts}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
