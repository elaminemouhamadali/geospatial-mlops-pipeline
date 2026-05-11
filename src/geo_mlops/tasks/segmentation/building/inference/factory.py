from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import torch
import numpy as np
import rasterio
from geo_mlops.core.utils.windows import _to_channels
from geo_mlops.core.data.types import DiscoveredScene, SceneArrays
from geo_mlops.core.inference.types import (
    InferenceConfig,
    InferencePrediction,
    InferenceArtifacts,
)

from geo_mlops.core.inference.factory import resolve_inference_data_cfg
    
    
def load_inference_scene(
    scene: DiscoveredScene,
    inference_cfg: Dict[str, Any],
) -> SceneArrays:
    data_cfg = resolve_inference_data_cfg(inference_cfg)

    reflectance_max = float(data_cfg["reflectance_max"])
    use_context = bool(data_cfg["use_context"])
    tile_out_channels = int(data_cfg["tile_out_channels"])
    context_out_channels = int(data_cfg["context_out_channels"])

    with rasterio.open(scene.pan_path) as src:
        image_np = src.read(1).astype(np.float32)
        profile = dict(src.profile)

    image_np = np.clip(image_np, 0.0, reflectance_max) / max(1e-6, reflectance_max)
    image_t = torch.from_numpy(image_np).unsqueeze(0)
    image_t = _to_channels(image_t, tile_out_channels)


    context_t = None
    if use_context and scene.context_path is not None:
        with rasterio.open(scene.context_path) as src:
            ctx = src.read().astype(np.float32)

        # Context stage outputs are assumed uint8 [0, 255].
        ctx = ctx / 255.0
        context_t = torch.from_numpy(ctx)
        context_t = _to_channels(context_t, context_out_channels)

    return SceneArrays(
        image=image_t.float(),
        context=context_t.float() if context_t is not None else None,
        profile=profile,
    )
    
def build_inference_postprocessor(
    inference_cfg: Dict[str, Any],
):
    data_cfg = resolve_inference_data_cfg(inference_cfg)

    output_channel = int(data_cfg.get("output_channel", 0))

    def postprocess_fn(outputs: torch.Tensor, batch: Dict[str, Any]) -> InferencePrediction:
        if not torch.is_tensor(outputs):
            raise TypeError(
                f"Building eval expected tensor outputs, got {type(outputs).__name__}"
            )

        if outputs.ndim != 4:
            raise ValueError(f"Expected outputs [B,C,H,W], got {tuple(outputs.shape)}")

        logits = outputs[:, output_channel, :, :]
        probs = torch.sigmoid(logits)

        return InferencePrediction(
            logits=logits,
            probability=probs,
        )

    return postprocess_fn
    

def save_inference_prediction(
    scene: DiscoveredScene,
    arrays: SceneArrays,
    prediction: InferencePrediction,
    inference_dir: Path,
    inference_cfg: InferenceConfig,
):
    scene_out_dir = Path(inference_dir) / scene.region / scene.subregion
    scene_out_dir.mkdir(exist_ok=True, parents=True)

    probability_dir = scene_out_dir / "probabilities"
    logits_dir = scene_out_dir / "logits"

    probability_dir.mkdir(exist_ok=True, parents=True)
    logits_dir.mkdir(exist_ok=True, parents=True)

    profile = dict(arrays.profile or {})
    profile.update(
        count=1,
        compress="deflate",
    )

    probability_path = None
    logits_path = None

    if inference_cfg.save_probabilities:
        probability_path = probability_dir / f"{scene.scene_id}.tif"
        prob_profile = dict(profile)
        prob_profile.update(dtype="float32", nodata=None)

        with rasterio.open(probability_path, "w", **prob_profile) as dst:
            dst.write(prediction.probability.astype(np.float32), 1)

    if inference_cfg.save_logits:
        logits_path = logits_dir / f"{scene.scene_id}.tif"
        logits_profile = dict(profile)
        logits_profile.update(dtype="float32", nodata=None)

        with rasterio.open(logits_path, "w", **logits_profile) as dst:
            dst.write(prediction.logits.astype(np.uint8), 1)

    return InferenceArtifacts(
        probability_path=probability_path,
        logits_path=logits_path,
    )




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