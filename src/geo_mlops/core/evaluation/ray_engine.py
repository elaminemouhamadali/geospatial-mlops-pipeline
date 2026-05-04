from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd
import ray
import torch

from geo_mlops.core.config.loader import load_cfg
from geo_mlops.core.evaluation.engine import EvalOutputs, EvalScene, run_full_scene_evaluation
from geo_mlops.core.execution.ray_backend import RayBackendConfig, init_ray_backend
from geo_mlops.core.registry.task_registry import get_task


def run_ray_full_scene_evaluation(
    *,
    task: str,
    task_cfg_path: str | Path,
    dataset_root: str | Path,
    checkpoint_path: str | Path,
    out_dir: str | Path,
    device: str = "cpu",
    ray_cfg: Optional[RayBackendConfig] = None,
    num_shards: Optional[int] = None,
    scenes_per_shard: Optional[int] = None,
    eval_overrides: Optional[Mapping[str, Any]] = None,
    model_uri: Optional[str] = None,
) -> EvalOutputs:
    """
    Ray-distributed full-scene evaluation.

    This engine distributes evaluation at the scene/shard level. Each Ray worker
    reconstructs the task plugin, model, checkpoint, eval config, and task hooks,
    then calls the existing local run_full_scene_evaluation(...) for its shard.

    This intentionally does not distribute individual sliding-window tiles yet.
    The local eval engine remains responsible for full-scene sliding-window
    inference and stitching.

    Args:
        task:
            Registered task key, e.g. "building_seg".
        task_cfg_path:
            Path to unified task config.
        dataset_root:
            Golden/full-scene evaluation dataset root.
        checkpoint_path:
            Path to model checkpoint.
        out_dir:
            Final Ray evaluation output directory.
        device:
            Device string used inside Ray workers. For ray-cpu image, use "cpu".
        ray_cfg:
            Ray connection config. Defaults to address="auto".
        num_shards:
            Number of scene shards. If None, use one shard per available CPU,
            capped by number of scenes.
        scenes_per_shard:
            Optional fixed number of scenes per shard. Overrides num_shards.
        eval_overrides:
            Optional overrides for EvalConfig fields such as tile_size, stride,
            batch_size, threshold, seed.
        model_uri:
            Optional MLflow/model registry URI to carry into summaries.

    Returns:
        EvalOutputs pointing to the final merged eval outputs.
    """

    ray_cfg = ray_cfg or RayBackendConfig(address="auto")
    init_ray_backend(ray_cfg)

    task_cfg_path = Path(task_cfg_path)
    dataset_root = Path(dataset_root)
    checkpoint_path = Path(checkpoint_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_plugin = get_task(task)

    eval_cfg = task_plugin.build_evaluation_cfg(task_cfg_path)
    eval_engine_cfg = task_plugin.build_eval_engine_cfg(eval_cfg)
    eval_engine_cfg = _apply_eval_overrides(eval_engine_cfg, eval_overrides)

    scenes = list(
        task_plugin.iter_eval_scenes(
            dataset_root=dataset_root,
            eval_cfg=eval_cfg,
        )
    )

    if not scenes:
        raise ValueError("No evaluation scenes were discovered.")

    shards = _build_scene_shards(
        scenes=scenes,
        num_shards=num_shards,
        scenes_per_shard=scenes_per_shard,
    )

    shard_root = out_dir / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)

    scene_payload_shards = [
        [_scene_to_payload(scene) for scene in shard]
        for shard in shards
    ]

    refs = []

    for shard_idx, scene_payloads in enumerate(scene_payload_shards):
        shard_out_dir = shard_root / f"shard_{shard_idx:04d}"

        refs.append(
            _run_eval_shard.remote(
                task=task,
                task_cfg_path=str(task_cfg_path),
                checkpoint_path=str(checkpoint_path),
                scene_payloads=scene_payloads,
                out_dir=str(shard_out_dir),
                device=device,
                eval_overrides=dict(eval_overrides or {}),
                model_uri=model_uri,
            )
        )

    shard_results = ray.get(refs)

    return _merge_shard_eval_outputs(
        task=task,
        out_dir=out_dir,
        cfg=eval_engine_cfg,
        eval_cfg_raw=eval_cfg,
        checkpoint_path=checkpoint_path,
        model_uri=model_uri,
        shard_results=shard_results,
    )


@ray.remote(num_cpus=1)
def _run_eval_shard(
    *,
    task: str,
    task_cfg_path: str,
    checkpoint_path: str,
    scene_payloads: List[Dict[str, Any]],
    out_dir: str,
    device: str,
    eval_overrides: Dict[str, Any],
    model_uri: Optional[str],
) -> Dict[str, Any]:
    """
    Ray worker entrypoint for one scene shard.

    Important: this function reconstructs everything inside the Ray worker
    instead of receiving a live model/plugin/metric accumulator from the driver.
    """

    task_cfg_path_p = Path(task_cfg_path)
    checkpoint_path_p = Path(checkpoint_path)
    out_dir_p = Path(out_dir)

    torch_device = _resolve_device(device)

    task_plugin = get_task(task)

    eval_cfg = task_plugin.build_evaluation_cfg(task_cfg_path_p)
    eval_engine_cfg = task_plugin.build_eval_engine_cfg(eval_cfg)
    eval_engine_cfg = _apply_eval_overrides(eval_engine_cfg, eval_overrides)

    train_cfg = _load_training_cfg_from_task_cfg(task_cfg_path_p)

    model = task_plugin.build_model(train_cfg)
    model = task_plugin.load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path_p,
        device=torch_device,
    )

    scenes = [_scene_from_payload(payload) for payload in scene_payloads]

    outputs = run_full_scene_evaluation(
        task=task,
        model=model,
        scenes=scenes,
        out_dir=out_dir_p,
        device=torch_device,
        cfg=eval_engine_cfg,
        load_scene_fn=lambda scene: task_plugin.load_eval_scene(scene, eval_cfg),
        forward_fn=task_plugin.get_forward_fn(),
        postprocess_fn=task_plugin.build_eval_postprocessor(eval_cfg),
        save_prediction_fn=task_plugin.save_eval_prediction,
        metric_accumulator=task_plugin.build_eval_metric_accumulator(eval_cfg),
        eval_cfg_raw=eval_cfg,
        checkpoint_path=checkpoint_path_p,
        model_uri=model_uri,
    )

    return {
        "eval_dir": str(outputs.eval_dir),
        "summary_path": str(outputs.summary_path),
        "manifest_path": str(outputs.manifest_path),
        "per_scene_table_path": str(outputs.per_scene_table_path),
        "probability_dir": str(outputs.probability_dir),
        "mask_dir": str(outputs.mask_dir),
        "num_scenes": int(outputs.summary.get("num_scenes", len(scenes))),
        "scene_ids": list(outputs.summary.get("scene_ids", [s.scene_id for s in scenes])),
        "summary": outputs.summary,
    }


def _merge_shard_eval_outputs(
    *,
    task: str,
    out_dir: Path,
    cfg: Any,
    eval_cfg_raw: Mapping[str, Any],
    checkpoint_path: Path,
    model_uri: Optional[str],
    shard_results: Sequence[Mapping[str, Any]],
) -> EvalOutputs:
    """
    Merge per-shard eval outputs into a final eval_summary/eval_manifest.

    This implementation supports generic per-scene CSV merging and computes
    common segmentation-style aggregate metrics when tp/fp/fn/tn columns exist.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    probability_dir = out_dir / "predictions" / "probabilities"
    mask_dir = out_dir / "predictions" / "masks"
    probability_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    shard_manifest_paths = [str(r["manifest_path"]) for r in shard_results]
    shard_summary_paths = [str(r["summary_path"]) for r in shard_results]

    per_scene_frames: List[pd.DataFrame] = []

    for result in shard_results:
        table_path = Path(str(result["per_scene_table_path"]))
        if table_path.exists():
            per_scene_frames.append(pd.read_csv(table_path))

    if per_scene_frames:
        per_scene_df = pd.concat(per_scene_frames, ignore_index=True)
    else:
        per_scene_df = pd.DataFrame()

    per_scene_table_path = tables_dir / "per_scene_metrics.csv"
    per_scene_df.to_csv(per_scene_table_path, index=False)

    scene_ids: List[str] = []
    for result in shard_results:
        scene_ids.extend(str(s) for s in result.get("scene_ids", []))

    metrics = _aggregate_metrics_from_per_scene_table(per_scene_df)

    summary = {
        "schema_version": "eval.v1",
        "task": task,
        "eval_type": "ray_full_scene_sliding_window",
        "num_scenes": len(scene_ids),
        "scene_ids": scene_ids,
        "tile_size": int(getattr(cfg, "tile_size")),
        "stride": int(getattr(cfg, "stride")),
        "batch_size": int(getattr(cfg, "batch_size")),
        "threshold": float(getattr(cfg, "threshold")),
        "metrics": metrics,
        **{k: v for k, v in metrics.items() if isinstance(v, Mapping)},
        "artifacts": {
            "probability_dir": str(probability_dir),
            "mask_dir": str(mask_dir),
            "per_scene_metrics_csv": str(per_scene_table_path),
            "shard_manifest_paths": shard_manifest_paths,
            "shard_summary_paths": shard_summary_paths,
            "shards_dir": str(out_dir / "shards"),
        },
        "analytics": {
            "num_shards": len(shard_results),
            "shard_dirs": [str(r["eval_dir"]) for r in shard_results],
        },
    }

    summary_path = out_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest = {
        "schema_version": "eval_manifest.v1",
        "task": task,
        "eval_dir": str(out_dir),
        "summary_path": str(summary_path),
        "checkpoint_path": str(checkpoint_path),
        "model_uri": model_uri,
        "config": _as_plain_dict(cfg),
        "eval_cfg": dict(eval_cfg_raw or {}),
        "num_scenes": len(scene_ids),
        "scene_ids": scene_ids,
        "artifacts": summary["artifacts"],
        "execution": {
            "backend": "ray",
            "num_shards": len(shard_results),
        },
    }

    manifest_path = out_dir / "eval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return EvalOutputs(
        eval_dir=out_dir,
        summary_path=summary_path,
        manifest_path=manifest_path,
        per_scene_table_path=per_scene_table_path,
        probability_dir=probability_dir,
        mask_dir=mask_dir,
        summary=summary,
    )


def _aggregate_metrics_from_per_scene_table(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Best-effort generic metric aggregation.

    For building segmentation, the per-scene table includes tp/fp/fn/tn and
    scene-level metrics. We compute:
      - micro metrics from global summed counts
      - macro metrics from per-scene metric means

    Other tasks can later provide a task-specific distributed merge hook.
    """

    if df.empty:
        return {}

    metrics: Dict[str, Any] = {}

    count_cols = {"tp", "fp", "fn", "tn"}

    if count_cols.issubset(set(df.columns)):
        counts = {
            key: int(pd.to_numeric(df[key], errors="coerce").fillna(0).sum())
            for key in sorted(count_cols)
        }
        metrics["micro"] = _metrics_from_counts(counts)
        metrics["micro"]["counts"] = counts

    macro_cols = [
        "precision",
        "recall",
        "f1",
        "iou",
        "pixel_accuracy",
    ]

    macro: Dict[str, float] = {}

    metric_df = df.copy()
    if "has_target" in metric_df.columns:
        metric_df = metric_df[metric_df["has_target"] == True].copy()  # noqa: E712

    for col in macro_cols:
        if col in metric_df.columns:
            values = pd.to_numeric(metric_df[col], errors="coerce")
            macro[col] = float(values.mean())

    if macro:
        metrics["macro"] = macro

    return metrics


def _metrics_from_counts(
    counts: Mapping[str, int],
    *,
    eps: float = 1e-7,
) -> Dict[str, float]:
    tp = float(counts.get("tp", 0))
    fp = float(counts.get("fp", 0))
    fn = float(counts.get("fn", 0))
    tn = float(counts.get("tn", 0))

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    pixel_accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "pixel_accuracy": float(pixel_accuracy),
    }


def _build_scene_shards(
    *,
    scenes: Sequence[EvalScene],
    num_shards: Optional[int],
    scenes_per_shard: Optional[int],
) -> List[List[EvalScene]]:
    if scenes_per_shard is not None:
        if scenes_per_shard <= 0:
            raise ValueError("scenes_per_shard must be > 0.")

        return [
            list(scenes[i : i + scenes_per_shard])
            for i in range(0, len(scenes), scenes_per_shard)
        ]

    if num_shards is None:
        resources = ray.cluster_resources()
        available_cpus = int(resources.get("CPU", 1))
        num_shards = max(1, min(len(scenes), available_cpus))

    if num_shards <= 0:
        raise ValueError("num_shards must be > 0.")

    num_shards = min(num_shards, len(scenes))
    shard_size = int(math.ceil(len(scenes) / num_shards))

    return [
        list(scenes[i : i + shard_size])
        for i in range(0, len(scenes), shard_size)
    ]


def _scene_to_payload(scene: EvalScene) -> Dict[str, Any]:
    return {
        "scene_id": scene.scene_id,
        "image_path": str(scene.image_path),
        "gt_path": str(scene.gt_path) if scene.gt_path else None,
        "context_path": str(scene.context_path) if scene.context_path else None,
        "region": scene.region,
        "subregion": scene.subregion,
        "meta": dict(scene.meta or {}),
    }


def _scene_from_payload(payload: Mapping[str, Any]) -> EvalScene:
    return EvalScene(
        scene_id=str(payload["scene_id"]),
        image_path=Path(str(payload["image_path"])),
        gt_path=Path(str(payload["gt_path"])) if payload.get("gt_path") else None,
        context_path=Path(str(payload["context_path"])) if payload.get("context_path") else None,
        region=str(payload["region"]) if payload.get("region") is not None else None,
        subregion=str(payload["subregion"]) if payload.get("subregion") is not None else None,
        meta=dict(payload.get("meta") or {}),
    )


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[ray-evaluate] CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")

    return torch.device(device_arg)


def _load_training_cfg_from_task_cfg(task_cfg_path: Path) -> Dict[str, Any]:
    cfg = load_cfg(task_cfg_path)

    if not isinstance(cfg, dict):
        raise ValueError(f"Task config root must be a mapping: {task_cfg_path}")

    training = cfg.get("training")
    if not isinstance(training, dict):
        raise ValueError("Task config must include a 'training' mapping.")

    return training


def _apply_eval_overrides(eval_engine_cfg: Any, overrides: Optional[Mapping[str, Any]]) -> Any:
    overrides = dict(overrides or {})
    updates = {
        key: value
        for key, value in overrides.items()
        if value is not None
    }

    if not updates:
        return eval_engine_cfg

    from dataclasses import replace

    return replace(eval_engine_cfg, **updates)


def _as_plain_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}

    if is_dataclass(obj):
        return asdict(obj)

    if isinstance(obj, dict):
        return dict(obj)

    return {"value": str(obj)}