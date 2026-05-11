from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from geo_mlops.core.utils.dataclasses import _load_json
from geo_mlops.cli import evaluate, gate, register, split, tile, train, inference
from geo_mlops.core.io.tile_io import TILES_MANIFEST_NAME
from geo_mlops.core.io.split_io import SPLIT_MANIFEST_NAME
from geo_mlops.core.io.train_io import TRAIN_MANIFEST_NAME, METRICS_MANIFEST_NAME
from geo_mlops.core.io.gate_io import GATE_MANIFEST_NAME
from geo_mlops.core.io.inference_io import INFERENCE_MANIFEST_NAME
from geo_mlops.core.io.eval_io import EVAL_MANIFEST_NAME
from geo_mlops.core.registry.model_registry import REGISTRY_MANIFEST_NAME



def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run the full geospatial MLOps pipeline: "
            "tiling -> split -> train -> gate A -> register candidate -> "
            "golden evaluation -> gate B -> promote production."
        )
    )
    p.add_argument("--task", type=str, required=True, help="Task key, e.g. building_seg.")
    p.add_argument(
        "--task-cfg-path",
        "--task_cfg_path",
        dest="task_cfg_path",
        type=Path,
        required=True,
        help="Unified task config YAML/JSON.",
    )
    p.add_argument(
        "--dataset-root-path",
        "--dataset_root_path",
        dest="dataset_root_path",
        type=Path,
        required=True,
        help="Training/validation dataset root used for tiling.",
    )
    p.add_argument(
        "--golden-root-path",
        "--golden_root_path",
        dest="golden_root_path",
        type=Path,
        required=True,
        help="Golden full-scene evaluation dataset root.",
    )
    p.add_argument(
        "--run-dir-path",
        "--run_dir_path",
        dest="run_dir_path",
        type=Path,
        required=True,
        help="Root output directory for all pipeline artifacts.",
    )
    p.add_argument("--device", type=str, default="cuda")

    # MLflow
    p.add_argument("--mlflow", action="store_true", help="Enable MLflow during training.")
    p.add_argument("--mlflow-tracking-uri", type=str, default=None)
    p.add_argument("--mlflow-experiment", type=str, default=None)

    # Stage control
    p.add_argument("--force-tiling", action="store_true")
    p.add_argument("--verbose-tiling", action="store_true")

    p.add_argument("--skip-tiling", action="store_true")
    p.add_argument("--skip-splitting", action="store_true")
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--skip-gate-a", action="store_true")
    p.add_argument("--skip-register-candidate", action="store_true")
    p.add_argument("--skip-inference", action="store_true")
    p.add_argument("--skip-evaluation", action="store_true")
    p.add_argument("--skip-gate-b", action="store_true")
    p.add_argument("--skip-promote-production", action="store_true")

    p.add_argument(
        "--stop-after-gate-a",
        action="store_true",
        help="Stop after Gate A and candidate registration.",
    )

    return p


def _append_optional(argv: list[str], flag: str, value) -> None:
    if value is not None:
        argv.extend([flag, str(value)])


def _run_stage(name: str, fn, argv: list[str]) -> None:
    print("\n" + "=" * 88)
    print(f"[run_pipeline] START {name}")
    print("[run_pipeline] argv:", " ".join(argv))
    print("=" * 88)

    rc = fn(argv)

    if rc not in (None, 0):
        raise RuntimeError(f"Stage {name!r} failed with return code {rc}")

    print(f"[run_pipeline] DONE {name}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    run_dir_path = args.run_dir_path
    run_dir_path.mkdir(parents=True, exist_ok=True)

    tiles_dir_path = run_dir_path / "tiles"
    split_dir_path = run_dir_path / "split"
    train_dir_path = run_dir_path/ "train"
    gate_a_dir_path = run_dir_path/ "gate_a"
    registry_candidate_dir_path = run_dir_path/ "registry_candidate"
    golden_inference_dir_path = run_dir_path/ "golden_inference"
    golden_eval_dir_path = run_dir_path/ "golden_eval"
    gate_b_dir_path = run_dir_path/ "gate_b"
    registry_production_dir_path = run_dir_path/ "registry_production"

    tiles_manifest_path = tiles_dir_path / TILES_MANIFEST_NAME
    split_manifest_path = split_dir_path / SPLIT_MANIFEST_NAME
    train_manifest_path = train_dir_path / TRAIN_MANIFEST_NAME
    train_metrics_path = train_dir_path / METRICS_MANIFEST_NAME
    gate_a_manifest_path = gate_a_dir_path / GATE_MANIFEST_NAME
    registry_candidate_manifest_path = registry_candidate_dir_path / REGISTRY_MANIFEST_NAME
    inference_manifest_path = golden_inference_dir_path / INFERENCE_MANIFEST_NAME
    eval_manifest_path = golden_eval_dir_path / EVAL_MANIFEST_NAME
    eval_metrics_path = golden_eval_dir_path / METRICS_MANIFEST_NAME
    gate_b_manifest_path = gate_b_dir_path / GATE_MANIFEST_NAME
    registry_production_manifest_path = registry_production_dir_path / REGISTRY_MANIFEST_NAME

    # -------------------------------------------------------------------------
    # 1. Tiling
    # -------------------------------------------------------------------------
    if not args.skip_tiling:
        tile_argv = [
            "--task", args.task,
            "--task-cfg-path", str(args.task_cfg_path),
            "--dataset-root-path", str(args.dataset_root_path),
            "--tiles-dir-path", str(tiles_dir_path),
        ]

        if args.force_tiling:
            tile_argv.append("--force")
        if args.verbose_tiling:
            tile_argv.append("--verbose")

        _run_stage("tile", tile.main, tile_argv)
    else:
        print("[run_pipeline] SKIP tiling")

    # -------------------------------------------------------------------------
    # 2. Split
    # -------------------------------------------------------------------------
    if not args.skip_splitting:
        split_argv = [
            "--task", args.task,
            "--task-cfg-path", str(args.task_cfg_path),
            "--tiles-manifest-path", str(tiles_manifest_path),
            "--split-dir-path", str(split_dir_path),
        ]

        _run_stage("split", split.main, split_argv)
    else:
        print("[run_pipeline] SKIP splitting")

    # -------------------------------------------------------------------------
    # 3. Train
    # -------------------------------------------------------------------------
    if not args.skip_training:
        train_argv = [
            "--task", args.task,
            "--task-cfg-path", str(args.task_cfg_path),
            "--tiles-manifest-path", str(tiles_manifest_path),
            "--split-manifest-path", str(split_manifest_path),
            "--train-dir-path", str(train_dir_path),
            "--device", args.device,
        ]

        if args.mlflow:
            train_argv.append("--mlflow")
        _append_optional(train_argv, "--mlflow-tracking-uri", args.mlflow_tracking_uri)
        _append_optional(train_argv, "--mlflow-experiment", args.mlflow_experiment)

        _run_stage("train", train.main, train_argv)
    else:
        print("[run_pipeline] SKIP training")

    # -------------------------------------------------------------------------
    # 4. Gate A
    # -------------------------------------------------------------------------
    if not args.skip_gate_a:
        gate_a_argv = [
            "--task", args.task,
            "--task-cfg-path", str(args.task_cfg_path),
            "--gate-name", "gate_a",
            "--metrics-file-path", str(train_metrics_path),
            "--gate-dir-path", str(gate_a_dir_path),
        ]

        _run_stage("gate_a", gate.main, gate_a_argv)
    else:
        print("[run_pipeline] SKIP gate_a")

    # -------------------------------------------------------------------------
    # 5. Register candidate
    # -------------------------------------------------------------------------
    if not args.skip_register_candidate:
        register_candidate_argv = [
            "--task", args.task,
            "--task-cfg-path", str(args.task_cfg_path),
            "--action", "register-candidate",
            "--gate-manifest-path", str(gate_a_manifest_path),
            "--register-dir-path", str(registry_candidate_dir_path),
            "--train-manifest-path", str(train_manifest_path),
        ]

        _run_stage("register_candidate", register.main, register_candidate_argv)
    else:
        print("[run_pipeline] SKIP register_candidate")

    if args.stop_after_gate_a:
        print("[run_pipeline] stop requested after Gate A / candidate registration.")
        return 0


    # -------------------------------------------------------------------------
    # 6. Golden full-scene inference
    # -------------------------------------------------------------------------
    if not args.skip_inference:
        inference_argv = [
            "--task", args.task,
            "--task-cfg-path", str(args.task_cfg_path),
            "--dataset-root-path", str(args.golden_root_path),
            "--train-manifest-path", str(train_manifest_path),
            "--inference-dir-path", str(golden_inference_dir_path),
            "--device", args.device,
        ]

        _run_stage("evaluate_golden", inference.main, eval_argv)
    else:
        print("[run_pipeline] SKIP evaluation")

    # -------------------------------------------------------------------------
    # 7. Golden full-scene evaluation (scoring)
    # -------------------------------------------------------------------------
    if not args.skip_evaluation:
        eval_argv = [
            "--task", args.task,
            "--task-cfg-path", str(args.task_cfg_path),
            "--inference-manifest-path", str(inference_manifest_path),
            "--eval-dir-path", str(golden_eval_dir_path),
        ]

        _run_stage("evaluate_golden", evaluate.main, eval_argv)
    else:
        print("[run_pipeline] SKIP evaluation")

    # -------------------------------------------------------------------------
    # 8. Gate B
    # -------------------------------------------------------------------------
    if not args.skip_gate_b:
        gate_b_argv = [
            "--task", args.task,
            "--task-cfg-path", str(args.task_cfg_path),
            "--gate-name", "gate_b",
            "--metrics-file-path", str(eval_metrics_path),
            "--gate-dir-path", str(gate_b_dir_path) ,
        ]

        _run_stage("gate_b", gate.main, gate_b_argv)
    else:
        print("[run_pipeline] SKIP gate_b")

    # -------------------------------------------------------------------------
    # 8. Promote production
    # -------------------------------------------------------------------------
    if not args.skip_promote_production:
        candidate = _load_json(registry_candidate_manifest_path)
        model_version = str(candidate.get("model_version"))

        promote_argv = [
            "--task", args.task,
            "--task-cfg-path", str(args.task_cfg_path),
            "--action", "promote-production",
            "--gate-manifest-path", str(gate_b_manifest_path),
            "--register-dir-path", str(registry_production_dir_path),
            "--train-manifest-path", str(train_manifest_path),
            "--model-version", model_version,
        ]

        _run_stage("promote_production", register.main, promote_argv)
    else:
        print("[run_pipeline] SKIP promote_production")

    print("\n" + "=" * 88)
    print("[run_pipeline] PIPELINE COMPLETE")
    print(f"[run_pipeline] run_dir_path={run_dir_path}")
    print(f"[run_pipeline] train_manifest={train_manifest_path}")
    print(f"[run_pipeline] gate_a={gate_a_manifest_path}")
    print(f"[run_pipeline] registry_candidate={registry_candidate_manifest_path}")
    print(f"[run_pipeline] inference_manifest={inference_manifest_path}")
    print(f"[run_pipeline] eval_manifest={eval_manifest_path}")
    print(f"[run_pipeline] gate_b={gate_b_manifest_path}")
    print(f"[run_pipeline] registry_production={registry_production_manifest_path}")
    print("=" * 88)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())