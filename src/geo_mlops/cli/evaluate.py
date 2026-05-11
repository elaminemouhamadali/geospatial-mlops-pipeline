from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from geo_mlops.core.io.inference_io import load_inference_contract
from geo_mlops.core.evaluation.engine import run_prediction_evaluation
from geo_mlops.core.registry.task_registry import get_task


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run full-scene sliding-window evaluation on a golden dataset using "
            "a trained task model."
        )
    )
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
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
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