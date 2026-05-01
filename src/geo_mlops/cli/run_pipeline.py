from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from geo_mlops.cli import evaluate, gate, register, split, tile, train


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
        "--task-cfg",
        "--task_cfg",
        dest="task_cfg",
        type=Path,
        required=True,
        help="Unified task config YAML/JSON.",
    )

    p.add_argument(
        "--dataset-root",
        "--dataset_root",
        dest="dataset_root",
        type=Path,
        required=True,
        help="Training/validation dataset root used for tiling.",
    )
    p.add_argument(
        "--golden-root",
        "--golden_root",
        dest="golden_root",
        type=Path,
        required=True,
        help="Golden full-scene evaluation dataset root.",
    )
    p.add_argument(
        "--run-dir",
        "--run_dir",
        dest="run_dir",
        type=Path,
        required=True,
        help="Root output directory for all pipeline artifacts.",
    )

    p.add_argument(
        "--csv-name",
        "--csv_name",
        dest="csv_name",
        type=str,
        default="tiles.csv",
        help="Per-ROI tile CSV cache filename.",
    )

    p.add_argument("--device", type=str, default="cuda")

    # MLflow
    p.add_argument("--mlflow", action="store_true", help="Enable MLflow during training.")
    p.add_argument("--mlflow-tracking-uri", type=str, default=None)
    p.add_argument("--mlflow-experiment", type=str, default=None)
    p.add_argument("--mlflow-run-name", type=str, default=None)

    # Runtime overrides for training
    p.add_argument("--train-batch-size", type=int, default=None)
    p.add_argument("--train-num-workers", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--selection-metric", type=str, default=None)
    p.add_argument("--selection-mode", choices=("min", "max"), default=None)

    # Runtime overrides for evaluation
    p.add_argument("--eval-tile-size", type=int, default=None)
    p.add_argument("--eval-stride", type=int, default=None)
    p.add_argument("--eval-batch-size", type=int, default=None)
    p.add_argument("--eval-threshold", type=float, default=None)

    # Stage control
    p.add_argument("--force-tiling", action="store_true")
    p.add_argument("--verbose-tiling", action="store_true")

    p.add_argument("--skip-tiling", action="store_true")
    p.add_argument("--skip-splitting", action="store_true")
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--skip-gate-a", action="store_true")
    p.add_argument("--skip-register-candidate", action="store_true")
    p.add_argument("--skip-evaluation", action="store_true")
    p.add_argument("--skip-gate-b", action="store_true")
    p.add_argument("--skip-promote-production", action="store_true")

    p.add_argument(
        "--stop-after-gate-a",
        action="store_true",
        help="Stop after Gate A and candidate registration.",
    )

    return p


def _path(run_dir: Path, name: str) -> Path:
    return run_dir / name


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


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Expected JSON file does not exist: {path}")

    obj = json.loads(path.read_text())

    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object at root of: {path}")

    return obj


def _assert_gate_passed(path: Path, gate_name: str) -> None:
    gate_contract = _load_json(path)

    if not bool(gate_contract.get("passed", False)):
        decision = gate_contract.get("decision", "<unknown>")
        raise RuntimeError(
            f"{gate_name} did not pass. decision={decision!r}. "
            f"See gate contract: {path}"
        )


def _resolve_candidate_model_version(registry_result_path: Path) -> str:
    result = _load_json(registry_result_path)

    version = result.get("model_version")
    if version is None:
        raise ValueError(
            f"registry_result.json does not contain model_version: {registry_result_path}"
        )

    return str(version)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    tiles_dir = _path(run_dir, "tiles")
    split_dir = _path(run_dir, "split")
    train_dir = _path(run_dir, "train")
    gate_a_dir = _path(run_dir, "gate_a")
    registry_candidate_dir = _path(run_dir, "registry_candidate")
    golden_eval_dir = _path(run_dir, "golden_eval")
    gate_b_dir = _path(run_dir, "gate_b")
    registry_production_dir = _path(run_dir, "registry_production")

    tiles_manifest = tiles_dir / "tiles_manifest.json"
    split_json = split_dir / "split.json"
    train_manifest = train_dir / "train_manifest.json"
    gate_a_contract = gate_a_dir / "gate_decision.json"
    registry_candidate_result = registry_candidate_dir / "registry_result.json"
    eval_summary = golden_eval_dir / "eval_summary.json"
    eval_manifest = golden_eval_dir / "eval_manifest.json"
    gate_b_contract = gate_b_dir / "gate_decision.json"

    # -------------------------------------------------------------------------
    # 1. Tiling
    # -------------------------------------------------------------------------
    if not args.skip_tiling:
        tile_argv = [
            "--task", args.task,
            "--task-cfg", str(args.task_cfg),
            "--dataset-root", str(args.dataset_root),
            "--csv-name", args.csv_name,
            "--out-dir", str(tiles_dir),
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
            "--task-cfg", str(args.task_cfg),
            "--tiles-dir", str(tiles_dir),
            "--out-dir", str(split_dir),
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
            "--task-cfg", str(args.task_cfg),
            "--tiles-dir", str(tiles_dir),
            "--split-dir", str(split_dir),
            "--out-dir", str(train_dir),
            "--device", args.device,
        ]

        _append_optional(train_argv, "--batch-size", args.train_batch_size)
        _append_optional(train_argv, "--num-workers", args.train_num_workers)
        _append_optional(train_argv, "--epochs", args.epochs)
        _append_optional(train_argv, "--lr", args.lr)
        _append_optional(train_argv, "--seed", args.seed)
        _append_optional(train_argv, "--selection-metric", args.selection_metric)
        _append_optional(train_argv, "--selection-mode", args.selection_mode)

        if args.mlflow:
            train_argv.append("--mlflow")
        _append_optional(train_argv, "--mlflow-tracking-uri", args.mlflow_tracking_uri)
        _append_optional(train_argv, "--mlflow-experiment", args.mlflow_experiment)
        _append_optional(train_argv, "--mlflow-run-name", args.mlflow_run_name)

        _run_stage("train", train.main, train_argv)
    else:
        print("[run_pipeline] SKIP training")

    # -------------------------------------------------------------------------
    # 4. Gate A
    # -------------------------------------------------------------------------
    if not args.skip_gate_a:
        gate_a_argv = [
            "--task", args.task,
            "--task-cfg", str(args.task_cfg),
            "--gate-name", "gate_a",
            "--metrics-file", str(train_dir / "metrics.json"),
            "--out-dir", str(gate_a_dir),
            "--train-manifest", str(train_manifest),
            "--split-json", str(split_json),
            "--tiles-manifest", str(tiles_manifest),
        ]

        _run_stage("gate_a", gate.main, gate_a_argv)
        _assert_gate_passed(gate_a_contract, "gate_a")
    else:
        print("[run_pipeline] SKIP gate_a")

    # -------------------------------------------------------------------------
    # 5. Register candidate
    # -------------------------------------------------------------------------
    if not args.skip_register_candidate:
        register_candidate_argv = [
            "--task", args.task,
            "--task-cfg", str(args.task_cfg),
            "--action", "register-candidate",
            "--gate-contract", str(gate_a_contract),
            "--out-dir", str(registry_candidate_dir),
        ]

        _append_optional(
            register_candidate_argv,
            "--mlflow-tracking-uri",
            args.mlflow_tracking_uri,
        )

        _run_stage("register_candidate", register.main, register_candidate_argv)
    else:
        print("[run_pipeline] SKIP register_candidate")

    if args.stop_after_gate_a:
        print("[run_pipeline] stop requested after Gate A / candidate registration.")
        return 0

    # -------------------------------------------------------------------------
    # 6. Golden full-scene evaluation
    # -------------------------------------------------------------------------
    if not args.skip_evaluation:
        eval_argv = [
            "--task", args.task,
            "--task-cfg", str(args.task_cfg),
            "--dataset-root", str(args.golden_root),
            "--train-manifest", str(train_manifest),
            "--out-dir", str(golden_eval_dir),
            "--device", args.device,
        ]

        _append_optional(eval_argv, "--tile-size", args.eval_tile_size)
        _append_optional(eval_argv, "--stride", args.eval_stride)
        _append_optional(eval_argv, "--batch-size", args.eval_batch_size)
        _append_optional(eval_argv, "--threshold", args.eval_threshold)
        _append_optional(eval_argv, "--seed", args.seed)

        _run_stage("evaluate_golden", evaluate.main, eval_argv)
    else:
        print("[run_pipeline] SKIP evaluation")

    # -------------------------------------------------------------------------
    # 7. Gate B
    # -------------------------------------------------------------------------
    if not args.skip_gate_b:
        gate_b_argv = [
            "--task", args.task,
            "--task-cfg", str(args.task_cfg),
            "--gate-name", "gate_b",
            "--metrics-file", str(eval_summary),
            "--out-dir", str(gate_b_dir),
            "--eval-manifest", str(eval_manifest),
            "--train-manifest", str(train_manifest),
        ]

        _run_stage("gate_b", gate.main, gate_b_argv)
        _assert_gate_passed(gate_b_contract, "gate_b")
    else:
        print("[run_pipeline] SKIP gate_b")

    # -------------------------------------------------------------------------
    # 8. Promote production
    # -------------------------------------------------------------------------
    if not args.skip_promote_production:
        model_version = _resolve_candidate_model_version(registry_candidate_result)

        promote_argv = [
            "--task", args.task,
            "--task-cfg", str(args.task_cfg),
            "--action", "promote-production",
            "--gate-contract", str(gate_b_contract),
            "--model-version", model_version,
            "--out-dir", str(registry_production_dir),
        ]

        _append_optional(promote_argv, "--mlflow-tracking-uri", args.mlflow_tracking_uri)

        _run_stage("promote_production", register.main, promote_argv)
    else:
        print("[run_pipeline] SKIP promote_production")

    print("\n" + "=" * 88)
    print("[run_pipeline] PIPELINE COMPLETE")
    print(f"[run_pipeline] run_dir={run_dir}")
    print(f"[run_pipeline] train_manifest={train_manifest}")
    print(f"[run_pipeline] gate_a={gate_a_contract}")
    print(f"[run_pipeline] eval_summary={eval_summary}")
    print(f"[run_pipeline] gate_b={gate_b_contract}")
    print(f"[run_pipeline] registry_candidate={registry_candidate_result}")
    print(f"[run_pipeline] registry_production={registry_production_dir / 'registry_result.json'}")
    print("=" * 88)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())