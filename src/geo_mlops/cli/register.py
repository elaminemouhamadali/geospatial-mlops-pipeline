from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from geo_mlops.core.config.loader import load_cfg
from geo_mlops.core.io.train_io import load_train_contract
from geo_mlops.core.registry.model_registry import (
    promote_model_to_production,
    register_candidate_model,
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Register or promote MLflow model versions based on gate contracts.")

    p.add_argument("--task", type=str, required=True, help="Task name, e.g. building_seg.")
    p.add_argument(
        "--task-cfg-path",
        "--task_cfg_path",
        dest="task_cfg_path",
        type=Path,
        required=True,
        help="Unified task config containing registry settings.",
    )
    p.add_argument(
        "--action",
        required=True,
        choices=("register-candidate", "promote-production"),
        help="Registry transition to perform.",
    )
    p.add_argument(
        "--gate-manifest-path",
        type=Path,
        required=True,
        help="Path to gate contract JSON. Must have passed=true.",
    )
    p.add_argument(
        "--register-dir-path",
        type=Path,
        default=None,
        help="Optional output directory for registry_result.json.",
    )

    # Optional for register-candidate; inferred from train manifest when omitted.
    p.add_argument(
        "--train-manifest-path",
        type=Path,
        default=None,
        help=("Optional train_manifest.json. If omitted, register.py attempts to read gate_contract.upstream.train_manifest."),
    )
    # Required for promote-production.
    p.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="Registered model version to promote.",
    )

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    task_cfg = load_cfg(args.task_cfg_path)
    reg_cfg = task_cfg.get("registry")

    model_name = reg_cfg.get("model_name")

    training_contract = load_train_contract(args.train_manifest_path)
    tracking = training_contract.tracking
    model_artifact_path = reg_cfg.get("model_artifact_path")
    model_artifact_path = str(model_artifact_path)

    tracking_uri = reg_cfg.get("tracking_uri")

    if args.action == "register-candidate":
        mlflow_run_id = tracking.get("mlflow_run_id")

        result = register_candidate_model(
            model_name=model_name,
            mlflow_run_id=mlflow_run_id,
            gate_contract_path=args.gate_manifest_path,
            model_artifact_path=model_artifact_path,
            tracking_uri=tracking_uri,
            candidate_alias=str(reg_cfg.get("candidate_alias", "candidate")),
            out_dir=args.register_dir_path,
            extra_tags={
                "task": args.task,
                "task_cfg": str(args.task_cfg_path),
                "train_manifest": str(args.train_manifest_path),
            },
        )

        print("[registry] registered candidate")
        print(f"[registry] model={result.model_name}")
        print(f"[registry] version={result.model_version}")
        print(f"[registry] uri={result.model_uri}")
        return 0

    if args.action == "promote-production":
        if not args.model_version:
            raise ValueError("--model-version is required for --action promote-production")

        result = promote_model_to_production(
            model_name=model_name,
            model_version=args.model_version,
            gate_contract_path=args.gate_manifest_path,
            tracking_uri=tracking_uri,
            production_alias=str(reg_cfg.get("production_alias", "production")),
            archive_candidate_alias=bool(reg_cfg.get("archive_candidate_alias", False)),
            out_dir=args.register_dir_path,
            extra_tags={
                "task": args.task,
                "task_cfg": str(args.task_cfg_path),
            },
        )

        print("[registry] promoted model to production")
        print(f"[registry] model={result.model_name}")
        print(f"[registry] version={result.model_version}")
        return 0

    raise ValueError(f"Unhandled action: {args.action}")


if __name__ == "__main__":
    raise SystemExit(main())
