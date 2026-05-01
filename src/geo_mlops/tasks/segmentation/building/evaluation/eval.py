from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import torch
import numpy as np
import rasterio
from geo_mlops.core.utils.windows import _to_channels

from geo_mlops.core.evaluation.engine import (
    EvalConfig,
    EvalPrediction,
    EvalScene,
    PredictionArtifacts,
    SceneArrays,
    ScenePrediction,
)
from geo_mlops.tasks.segmentation.building.modeling.metrics import (
    BuildingSegmentationEvalAccumulator,
)

from geo_mlops.core.config.loader import load_cfg, require_section


   
def build_evaluation_cfg(task_cfg_path: str | Path) -> Dict[str, Any]:
    cfg = load_cfg(task_cfg_path)
    return require_section(cfg, "evaluation")


def build_eval_engine_cfg(eval_cfg: Dict[str, Any]) -> EvalConfig:
    engine_cfg = eval_cfg.get("engine", {}) or {}

    return EvalConfig(
        tile_size=int(engine_cfg.get("tile_size", 512)),
        stride=int(engine_cfg.get("stride", 256)),
        batch_size=int(engine_cfg.get("batch_size", 4)),
        seed=int(engine_cfg.get("seed", 1337)),
        threshold=float(engine_cfg.get("threshold", 0.5)),
        save_probabilities=bool(engine_cfg.get("save_probabilities", True)),
        save_masks=bool(engine_cfg.get("save_masks", True)),
    )
    
def iter_eval_scenes(
    *,
    dataset_root: str | Path,
    eval_cfg: Dict[str, Any],
) -> list[EvalScene]:
    dataset_root = Path(dataset_root)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Evaluation dataset_root does not exist: {dataset_root}")

    data_cfg = eval_cfg.get("data", {}) or {}

    pan_dirname = str(data_cfg.get("pan_dirname", "PAN"))
    gt_dirname = data_cfg.get("gt_dirname", "GT-Mask")
    context_dirname = data_cfg.get("context_dirname", "Context")

    dataset_buckets = data_cfg.get("dataset_buckets", None)
    regions = data_cfg.get("regions", None)

    if dataset_buckets:
        bucket_dirs = [dataset_root / str(b) for b in dataset_buckets]
    else:
        bucket_dirs = [p for p in sorted(dataset_root.iterdir()) if p.is_dir()]

    scenes: list[EvalScene] = []

    for bucket_dir in bucket_dirs:
        if not bucket_dir.exists():
            raise FileNotFoundError(f"Missing eval bucket directory: {bucket_dir}")

        if regions:
            region_dirs = [bucket_dir / str(r) for r in regions]
        else:
            region_dirs = [p for p in sorted(bucket_dir.iterdir()) if p.is_dir()]

        for region_dir in region_dirs:
            if not region_dir.exists():
                continue

            roi_dirs = [p for p in sorted(region_dir.iterdir()) if p.is_dir()]

            for roi_dir in roi_dirs:
                pan_dir = roi_dir / pan_dirname
                if not pan_dir.exists():
                    continue

                pan_paths = sorted(
                    list(pan_dir.glob("*.tif")) + list(pan_dir.glob("*.tiff"))
                )

                for pan_path in pan_paths:
                    gt_path = None
                    ctx_path = None

                    if gt_dirname:
                        candidate = roi_dir / str(gt_dirname) / pan_path.name
                        if candidate.exists():
                            gt_path = candidate

                    if context_dirname:
                        candidate = roi_dir / str(context_dirname) / pan_path.name
                        if candidate.exists():
                            ctx_path = candidate

                    scene_id = f"{bucket_dir.name}__{region_dir.name}__{roi_dir.name}__{pan_path.stem}"

                    scenes.append(
                        EvalScene(
                            scene_id=scene_id,
                            image_path=pan_path,
                            gt_path=gt_path,
                            context_path=ctx_path,
                            region=region_dir.name,
                            subregion=roi_dir.name,
                            meta={
                                "dataset_bucket": bucket_dir.name,
                                "stem": pan_path.stem,
                            },
                        )
                    )

    if not scenes:
        raise ValueError(f"No evaluation scenes found under: {dataset_root}")

    return scenes
    
def load_eval_scene(
    scene: EvalScene,
    eval_cfg: Dict[str, Any],
) -> SceneArrays:
    data_cfg = eval_cfg.get("data", {}) or {}

    reflectance_max = float(data_cfg.get("reflectance_max", 10_000.0))
    use_context = bool(data_cfg.get("use_context", True))
    tile_out_channels = int(data_cfg.get("tile_out_channels", 1))
    context_out_channels = int(data_cfg.get("context_out_channels", 1))
    foreground_label = int(data_cfg.get("foreground_label", 1))

    with rasterio.open(scene.image_path) as src:
        image_np = src.read(1).astype(np.float32)
        profile = dict(src.profile)

    image_np = np.clip(image_np, 0.0, reflectance_max) / max(1e-6, reflectance_max)
    image_t = torch.from_numpy(image_np).unsqueeze(0)
    image_t = _to_channels(image_t, tile_out_channels)

    target_t = None
    if scene.gt_path is not None:
        with rasterio.open(scene.gt_path) as src:
            target_np = src.read(1).astype(np.int64)
        target_t = torch.from_numpy((target_np == foreground_label).astype(np.uint8))

    context_t = None
    if use_context and scene.context_path is not None:
        with rasterio.open(scene.context_path) as src:
            ctx = src.read().astype(np.float32)

        # Context stage outputs are assumed uint8 [0,255].
        ctx = ctx / 255.0
        context_t = torch.from_numpy(ctx)
        context_t = _to_channels(context_t, context_out_channels)

    return SceneArrays(
        image=image_t.float(),
        target=target_t,
        context=context_t.float() if context_t is not None else None,
        profile=profile,
    )
    
def build_eval_postprocessor(eval_cfg: Dict[str, Any]):
    engine_cfg = eval_cfg.get("engine", {}) or {}
    data_cfg = eval_cfg.get("data", {}) or {}

    threshold = float(engine_cfg.get("threshold", 0.5))
    output_channel = int(data_cfg.get("output_channel", 0))

    def postprocess_fn(outputs: torch.Tensor, batch: Dict[str, Any]) -> EvalPrediction:
        if not torch.is_tensor(outputs):
            raise TypeError(f"Building eval expected tensor outputs, got {type(outputs).__name__}")

        if outputs.ndim != 4:
            raise ValueError(f"Expected outputs [B,C,H,W], got {tuple(outputs.shape)}")

        logits = outputs[:, output_channel, :, :]
        probs = torch.sigmoid(logits)
        masks = probs >= threshold

        return EvalPrediction(
            probability=probs,
            mask=masks,
            extra={"threshold": threshold},
        )

    return postprocess_fn
    

def save_eval_prediction(
    scene: EvalScene,
    arrays: SceneArrays,
    prediction: ScenePrediction,
    probability_dir: Path,
    mask_dir: Path,
    eval_cfg: EvalConfig,
) -> PredictionArtifacts:
    profile = dict(arrays.profile or {})
    profile.update(
        count=1,
        compress="deflate",
    )

    probability_path = None
    mask_path = None

    if eval_cfg.save_probabilities:
        probability_path = probability_dir / f"{scene.scene_id}_prob.tif"
        prob_profile = dict(profile)
        prob_profile.update(dtype="float32", nodata=None)

        with rasterio.open(probability_path, "w", **prob_profile) as dst:
            dst.write(prediction.probability.astype(np.float32), 1)

    if eval_cfg.save_masks:
        mask_path = mask_dir / f"{scene.scene_id}_mask.tif"
        mask_profile = dict(profile)
        mask_profile.update(dtype="uint8", nodata=0)

        with rasterio.open(mask_path, "w", **mask_profile) as dst:
            dst.write(prediction.mask.astype(np.uint8), 1)

    return PredictionArtifacts(
        probability_path=probability_path,
        mask_path=mask_path,
        extra={},
    )

    
def build_eval_metric_accumulator(eval_cfg: Dict[str, Any]):
    metrics_cfg = eval_cfg.get("metrics", {}) or {}
    return BuildingSegmentationEvalAccumulator(metrics_cfg)
    
    
def load_checkpoint(
    *,
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    state = torch.load(Path(checkpoint_path), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model