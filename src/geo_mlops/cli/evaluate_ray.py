from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from geo_mlops.core.evaluation.ray_engine import run_ray_full_scene_evaluation
from geo_mlops.core.execution.ray_backend import RayBackendConfig


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run Ray-distributed full-scene sliding-window evaluation on a golden "
            "dataset using a trained task model."
        )
    )

    p.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task key registered in task_registry, e.g. building_seg.",
    )
    p.add_argument(
        "--task-cfg",
        "--task_cfg",
        dest="task_cfg",
        type=Path,
        required=True,
        help="Unified task config YAML/JSON containing training/evaluation sections.",
    )
    p.add_argument(
        "--dataset-root",
        "--dataset_root",
        dest="dataset_root",
        type=Path,
        required=True,
        help="Golden evaluation dataset root. This is full-scene based, not tile/split based.",
    )
    p.add_argument(
        "--train-manifest",
        "--train_manifest",
        dest="train_manifest",
        type=Path,
        default=None,
        help=(
            "Optional train_manifest.json. Used to infer checkpoint path when "
            "--checkpoint is not provided."
        ),
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to local model checkpoint/state_dict. Overrides train_manifest model_path.",
    )
    p.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        type=Path,
        required=True,
        help="Output directory for Ray eval summary, manifest, shard outputs, masks, probabilities, and tables.",
    )

    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=(
            "Device used inside Ray workers. For geospatial-mlops:ray-cpu, use cpu. "
            "For future GPU workers, use cuda."
        ),
    )

    # Ray connection/config.
    p.add_argument(
        "--ray-address",
        "--ray_address",
        dest="ray_address",
        type=str,
        default="auto",
        help=(
            "Ray address. Use 'auto' when running inside the Ray head container or Ray driver pod. "
            "Use empty string/none only for local in-process Ray."
        ),
    )
    p.add_argument(
        "--ray-namespace",
        "--ray_namespace",
        dest="ray_namespace",
        type=str,
        default="geo-mlops",
        help="Ray namespace.",
    )
    p.add_argument(
        "--ray-shutdown-on-exit",
        "--ray_shutdown_on_exit",
        dest="ray_shutdown_on_exit",
        action="store_true",
        help=(
            "Call ray.shutdown() when done. Usually false when connecting to an existing cluster."
        ),
    )

    # Sharding controls.
    p.add_argument(
        "--num-shards",
        "--num_shards",
        dest="num_shards",
        type=int,
        default=None,
        help=(
            "Number of scene shards. Default: one shard per available Ray CPU, capped by number of scenes."
        ),
    )
    p.add_argument(
        "--scenes-per-shard",
        "--scenes_per_shard",
        dest="scenes_per_shard",
        type=int,
        default=None,
        help=(
            "Fixed number of scenes per shard. Overrides --num-shards when provided."
        ),
    )

    # Optional eval engine overrides.
    p.add_argument("--tile-size", "--tile_size", dest="tile_size", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)

    p.add_argument(
        "--model-uri",
        "--model_uri",
        dest="model_uri",
        type=str,
        default=None,
        help="Optional MLflow/model registry URI to record in eval manifest.",
    )

    return p


def _load_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")

    obj = json.loads(p.read_text())

    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object at root of {p}")

    return obj


def _resolve_checkpoint(
    *,
    checkpoint: Optional[Path],
    train_manifest: Optional[Path],
) -> Path:
    if checkpoint is not None:
        return checkpoint

    if train_manifest is None:
        raise ValueError("Provide --checkpoint or --train-manifest.")

    manifest = _load_json(train_manifest)

    model_path = manifest.get("model_path")
    if not model_path:
        raise ValueError(
            f"train_manifest does not contain model_path: {train_manifest}"
        )

    return Path(str(model_path))


def _none_if_none_string(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    v = value.strip()

    if v.lower() in {"", "none", "null"}:
        return None

    return v


def _build_ray_cfg(args: argparse.Namespace) -> RayBackendConfig:
    address = _none_if_none_string(args.ray_address)

    # This assumes your RayBackendConfig has these fields from the earlier version.
    # If your local dataclass has fewer fields, this filters unsupported kwargs.
    candidate_kwargs = {
        "address": address,
        "namespace": args.ray_namespace,
        "ignore_reinit_error": True,
        "log_cluster_resources": True,
        "runtime_env": None,
        "shutdown_on_exit": bool(args.ray_shutdown_on_exit),
    }

    valid_fields = {f.name for f in fields(RayBackendConfig)}
    kwargs = {k: v for k, v in candidate_kwargs.items() if k in valid_fields}

    return RayBackendConfig(**kwargs)


def _eval_overrides_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    overrides = {
        "tile_size": args.tile_size,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "threshold": args.threshold,
        "seed": args.seed,
    }

    return {k: v for k, v in overrides.items() if v is not None}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = _resolve_checkpoint(
        checkpoint=args.checkpoint,
        train_manifest=args.train_manifest,
    )

    ray_cfg = _build_ray_cfg(args)
    eval_overrides = _eval_overrides_from_args(args)

    outputs = run_ray_full_scene_evaluation(
        task=args.task,
        task_cfg_path=args.task_cfg,
        dataset_root=args.dataset_root,
        checkpoint_path=checkpoint_path,
        out_dir=args.out_dir,
        device=args.device,
        ray_cfg=ray_cfg,
        num_shards=args.num_shards,
        scenes_per_shard=args.scenes_per_shard,
        eval_overrides=eval_overrides,
        model_uri=args.model_uri,
    )

    print("[evaluate-ray] done")
    print(f"[evaluate-ray] scenes={outputs.summary.get('num_scenes')}")
    print(f"[evaluate-ray] summary={outputs.summary_path}")
    print(f"[evaluate-ray] manifest={outputs.manifest_path}")
    print(f"[evaluate-ray] per_scene_table={outputs.per_scene_table_path}")
    print(f"[evaluate-ray] probabilities={outputs.probability_dir}")
    print(f"[evaluate-ray] masks={outputs.mask_dir}")

    metrics = outputs.summary.get("metrics", {})
    if metrics:
        print(f"[evaluate-ray] metrics={json.dumps(metrics, indent=2)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())