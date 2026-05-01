from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from geo_mlops.core.utils.random import _seed_everything
from geo_mlops.core.utils.windows import build_grid


@dataclass(frozen=True)
class EvalConfig:
    tile_size: int = 512
    stride: int = 256
    batch_size: int = 4
    seed: int = 1337
    threshold: float = 0.5

    save_probabilities: bool = True
    save_masks: bool = True


@dataclass(frozen=True)
class EvalScene:
    scene_id: str
    image_path: Path
    gt_path: Optional[Path] = None
    context_path: Optional[Path] = None
    region: Optional[str] = None
    subregion: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class SceneArrays:
    """
    Task-provided full-scene arrays.

    image:
      torch.Tensor [C,H,W], float32, model-ready.

    target:
      Optional task-specific target. For segmentation this is often [H,W].

    context:
      Optional context tensor [C,H,W] or lower-res [C,h,w].

    profile:
      Optional raster profile/metadata passed back to the task writer.
    """

    image: torch.Tensor
    target: Optional[Any] = None
    context: Optional[torch.Tensor] = None
    profile: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class EvalPrediction:
    """
    Task postprocessor output for a batch of windows.

    probability:
      torch.Tensor [B,H,W] or [B,C,H,W], task-defined.

    mask:
      torch.Tensor [B,H,W] or [B,C,H,W], task-defined.

    extra:
      Optional task-specific metadata.
    """

    probability: torch.Tensor
    mask: torch.Tensor
    extra: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ScenePrediction:
    """
    Full-scene stitched prediction.
    """

    probability: np.ndarray
    mask: np.ndarray
    extra: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class PredictionArtifacts:
    probability_path: Optional[Path]
    mask_path: Optional[Path]
    extra: Optional[Dict[str, Any]] = None


class EvalMetricAccumulator(Protocol):
    """
    Task-specific metric/analytics accumulator.

    Building segmentation can implement:
      - global pixel counts for micro metrics
      - per-image metrics for macro metrics
      - Pareto/hardest-image tables

    Object detection, classification, water segmentation, etc. can implement
    completely different logic without changing core.
    """

    def update(
        self,
        *,
        scene: EvalScene,
        arrays: SceneArrays,
        prediction: ScenePrediction,
        artifacts: PredictionArtifacts,
    ) -> Dict[str, Any]:
        """
        Update internal state from one full-scene prediction.

        Returns a per-scene row to be written to tables/per_scene_metrics.csv.
        """
        ...

    def finalize(self, *, out_dir: Path) -> Dict[str, Any]:
        """
        Finalize metrics and optional analytics.

        Returns eval_summary-compatible payload, for example:
          {
            "metrics": {
              "micro": {...},
              "macro": {...}
            },
            "artifacts": {
              "pareto_images_csv": "..."
            }
          }
        """
        ...


@dataclass(frozen=True)
class EvalOutputs:
    eval_dir: Path
    summary_path: Path
    manifest_path: Path
    per_scene_table_path: Path
    probability_dir: Path
    mask_dir: Path
    summary: Dict[str, Any]


def run_full_scene_evaluation(
    *,
    task: str,
    model: torch.nn.Module,
    scenes: Iterable[EvalScene],
    out_dir: Path,
    device: torch.device,
    cfg: EvalConfig,
    load_scene_fn: Callable[[EvalScene], SceneArrays],
    forward_fn: Callable[[torch.nn.Module, Dict[str, Any], torch.device], Any],
    postprocess_fn: Callable[[Any, Dict[str, Any]], EvalPrediction],
    save_prediction_fn: Callable[
        [EvalScene, SceneArrays, ScenePrediction, Path, Path, EvalConfig],
        PredictionArtifacts,
    ],
    metric_accumulator: EvalMetricAccumulator,
    eval_cfg_raw: Optional[Mapping[str, Any]] = None,
    checkpoint_path: Optional[Path] = None,
    model_uri: Optional[str] = None,
) -> EvalOutputs:
    """
    Task-agnostic full-scene evaluation engine.

    Core responsibilities:
      - load full scenes via task hook
      - run sliding-window inference
      - stitch predictions
      - save outputs via task hook
      - update metrics via task accumulator
      - write eval summary/manifest

    Task responsibilities:
      - scene discovery
      - scene loading / normalization
      - output postprocessing
      - raster/prediction saving
      - metric and analytics definitions
    """

    _seed_everything(cfg.seed)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    probability_dir = out_dir / "predictions" / "probabilities"
    mask_dir = out_dir / "predictions" / "masks"
    tables_dir = out_dir / "tables"

    probability_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    scene_list = list(scenes)
    if not scene_list:
        raise ValueError("No evaluation scenes were provided.")

    model.to(device)
    model.eval()

    per_scene_rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for scene in tqdm(scene_list, desc="Eval scenes"):
            arrays = load_scene_fn(scene)

            prediction = _predict_scene(
                model=model,
                arrays=arrays,
                device=device,
                cfg=cfg,
                forward_fn=forward_fn,
                postprocess_fn=postprocess_fn,
            )

            artifacts = save_prediction_fn(
                scene,
                arrays,
                prediction,
                probability_dir,
                mask_dir,
                cfg,
            )

            row = metric_accumulator.update(
                scene=scene,
                arrays=arrays,
                prediction=prediction,
                artifacts=artifacts,
            )

            base_row = {
                "scene_id": scene.scene_id,
                "region": scene.region or "",
                "subregion": scene.subregion or "",
                "image_path": str(scene.image_path),
                "gt_path": str(scene.gt_path) if scene.gt_path else "",
                "context_path": str(scene.context_path) if scene.context_path else "",
                "probability_path": str(artifacts.probability_path) if artifacts.probability_path else "",
                "mask_path": str(artifacts.mask_path) if artifacts.mask_path else "",
            }

            per_scene_rows.append({**base_row, **row})

    per_scene_table_path = tables_dir / "per_scene_metrics.csv"
    pd.DataFrame(per_scene_rows).to_csv(per_scene_table_path, index=False)

    task_summary = metric_accumulator.finalize(out_dir=out_dir)

    summary = {
        "schema_version": "eval.v1",
        "task": task,
        "eval_type": "full_scene_sliding_window",
        "num_scenes": len(scene_list),
        "scene_ids": [s.scene_id for s in scene_list],
        "tile_size": int(cfg.tile_size),
        "stride": int(cfg.stride),
        "batch_size": int(cfg.batch_size),
        "threshold": float(cfg.threshold),
        "metrics": task_summary.get("metrics", {}),
        # Convenience top-level scopes for gate engine:
        **{
            k: v
            for k, v in task_summary.get("metrics", {}).items()
            if isinstance(v, Mapping)
        },
        "artifacts": {
            "probability_dir": str(probability_dir),
            "mask_dir": str(mask_dir),
            "per_scene_metrics_csv": str(per_scene_table_path),
            **task_summary.get("artifacts", {}),
        },
        "analytics": task_summary.get("analytics", {}),
    }

    summary_path = out_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    manifest = {
        "schema_version": "eval_manifest.v1",
        "task": task,
        "eval_dir": str(out_dir),
        "summary_path": str(summary_path),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "model_uri": model_uri,
        "config": _as_plain_dict(cfg),
        "eval_cfg": dict(eval_cfg_raw or {}),
        "num_scenes": len(scene_list),
        "scene_ids": [s.scene_id for s in scene_list],
        "artifacts": summary["artifacts"],
    }

    manifest_path = out_dir / "eval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return EvalOutputs(
        eval_dir=out_dir,
        summary_path=summary_path,
        manifest_path=manifest_path,
        per_scene_table_path=per_scene_table_path,
        probability_dir=probability_dir,
        mask_dir=mask_dir,
        summary=summary,
    )


def _predict_scene(
    *,
    model: torch.nn.Module,
    arrays: SceneArrays,
    device: torch.device,
    cfg: EvalConfig,
    forward_fn: Callable[[torch.nn.Module, Dict[str, Any], torch.device], Any],
    postprocess_fn: Callable[[Any, Dict[str, Any]], EvalPrediction],
) -> ScenePrediction:
    image = arrays.image

    if not torch.is_tensor(image):
        raise TypeError(f"SceneArrays.image must be torch.Tensor, got {type(image).__name__}")

    if image.ndim != 3:
        raise ValueError(f"SceneArrays.image must be [C,H,W], got {tuple(image.shape)}")

    channels, height, width = image.shape
    _ = channels

    ys, xs = build_grid(height, width, cfg.tile_size, cfg.stride)

    image = image.to(device)

    context = arrays.context
    if context is not None:
        if not torch.is_tensor(context):
            raise TypeError(
                f"SceneArrays.context must be torch.Tensor if provided, got {type(context).__name__}"
            )
        context = context.to(device)

    prob_sum: Optional[torch.Tensor] = None
    mask_sum: Optional[torch.Tensor] = None
    weight_sum: Optional[torch.Tensor] = None

    batch_tiles: List[torch.Tensor] = []
    batch_contexts: List[torch.Tensor] = []
    batch_windows: List[Tuple[int, int, int, int]] = []

    def flush_batch() -> None:
        nonlocal batch_tiles
        nonlocal batch_contexts
        nonlocal batch_windows
        nonlocal prob_sum
        nonlocal mask_sum
        nonlocal weight_sum

        if not batch_tiles:
            return

        tile_tensor = torch.stack(batch_tiles, dim=0)

        batch: Dict[str, Any] = {
            "tile_tensor": tile_tensor,
            "windows": batch_windows,
        }

        if context is not None:
            batch["context_tensor"] = torch.stack(batch_contexts, dim=0)

        outputs = forward_fn(model, batch, device)
        pred = postprocess_fn(outputs, batch)

        prob = _ensure_bchw(pred.probability).to(device=device, dtype=torch.float32)
        mask = _ensure_bchw(pred.mask).to(device=device, dtype=torch.float32)

        if prob.shape[0] != len(batch_windows):
            raise ValueError(
                f"postprocess_fn returned probability batch size {prob.shape[0]}, "
                f"expected {len(batch_windows)}"
            )

        if mask.shape[0] != len(batch_windows):
            raise ValueError(
                f"postprocess_fn returned mask batch size {mask.shape[0]}, "
                f"expected {len(batch_windows)}"
            )

        if prob_sum is None:
            out_channels = int(prob.shape[1])
            prob_sum = torch.zeros(
                (out_channels, height, width),
                dtype=torch.float32,
                device=device,
            )
            mask_sum = torch.zeros(
                (out_channels, height, width),
                dtype=torch.float32,
                device=device,
            )
            weight_sum = torch.zeros(
                (1, height, width),
                dtype=torch.float32,
                device=device,
            )

        assert prob_sum is not None
        assert mask_sum is not None
        assert weight_sum is not None

        for i, (y0, x0, y1, x1) in enumerate(batch_windows):
            h = y1 - y0
            w = x1 - x0

            prob_crop = prob[i, :, :h, :w]
            mask_crop = mask[i, :, :h, :w]

            prob_sum[:, y0:y1, x0:x1] += prob_crop
            mask_sum[:, y0:y1, x0:x1] += mask_crop
            weight_sum[:, y0:y1, x0:x1] += 1.0

        batch_tiles = []
        batch_contexts = []
        batch_windows = []

    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + cfg.tile_size, height)
            x1 = min(x0 + cfg.tile_size, width)

            tile = image[:, y0:y1, x0:x1]
            tile = _pad_to_size(tile, cfg.tile_size, cfg.tile_size)

            batch_tiles.append(tile)
            batch_windows.append((y0, x0, y1, x1))

            if context is not None:
                ctx_tile = _extract_context_window(
                    context=context,
                    y0=y0,
                    x0=x0,
                    y1=y1,
                    x1=x1,
                    image_height=height,
                    image_width=width,
                    tile_size=cfg.tile_size,
                )
                batch_contexts.append(ctx_tile)

            if len(batch_tiles) >= cfg.batch_size:
                flush_batch()

    flush_batch()

    if prob_sum is None or mask_sum is None or weight_sum is None:
        raise RuntimeError("No windows were processed during scene prediction.")

    weight_sum = torch.clamp(weight_sum, min=1.0)
    prob_full = prob_sum / weight_sum
    mask_score_full = mask_sum / weight_sum

    # Task postprocessor already thresholded window masks.
    # When overlaps exist, threshold averaged binary votes at 0.5.
    mask_full = mask_score_full >= 0.5

    prob_np = _squeeze_single_channel(prob_full.detach().cpu().numpy())
    mask_np = _squeeze_single_channel(mask_full.detach().cpu().numpy())

    return ScenePrediction(
        probability=prob_np,
        mask=mask_np,
        extra=None,
    )


def _ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize postprocessor output to [B,C,H,W].

    Accepts:
      [B,H,W]   -> [B,1,H,W]
      [B,1,H,W] -> unchanged
      [B,C,H,W] -> unchanged
    """
    if not torch.is_tensor(x):
        raise TypeError(f"Expected torch.Tensor, got {type(x).__name__}")

    if x.ndim == 3:
        return x.unsqueeze(1)

    if x.ndim == 4:
        return x

    raise ValueError(f"Expected [B,H,W] or [B,C,H,W], got {tuple(x.shape)}")


def _squeeze_single_channel(x: np.ndarray) -> np.ndarray:
    """
    Convert [1,H,W] -> [H,W], leave [C,H,W] unchanged.
    """
    if x.ndim == 3 and x.shape[0] == 1:
        return x[0]
    return x


def _extract_context_window(
    *,
    context: torch.Tensor,
    y0: int,
    x0: int,
    y1: int,
    x1: int,
    image_height: int,
    image_width: int,
    tile_size: int,
) -> torch.Tensor:
    if context.ndim != 3:
        raise ValueError(f"context must be [C,H,W], got {tuple(context.shape)}")

    _, context_height, context_width = context.shape

    if context_height == image_height and context_width == image_width:
        ctx = context[:, y0:y1, x0:x1]
        return _pad_to_size(ctx, tile_size, tile_size)

    cy0 = int(round(y0 * context_height / image_height))
    cy1 = int(round(y1 * context_height / image_height))
    cx0 = int(round(x0 * context_width / image_width))
    cx1 = int(round(x1 * context_width / image_width))

    cy0 = max(0, min(cy0, context_height - 1))
    cy1 = max(cy0 + 1, min(cy1, context_height))
    cx0 = max(0, min(cx0, context_width - 1))
    cx1 = max(cx0 + 1, min(cx1, context_width))

    ctx = context[:, cy0:cy1, cx0:cx1].unsqueeze(0)
    ctx = F.interpolate(
        ctx,
        size=(tile_size, tile_size),
        mode="bilinear",
        align_corners=False,
    )
    return ctx.squeeze(0)


def _pad_to_size(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(x.shape)}")

    _, h, w = x.shape

    pad_h = max(0, height - h)
    pad_w = max(0, width - w)

    if pad_h == 0 and pad_w == 0:
        return x

    return F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)


def _as_plain_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}

    if is_dataclass(obj):
        return asdict(obj)

    if isinstance(obj, dict):
        return dict(obj)

    return {"value": str(obj)}