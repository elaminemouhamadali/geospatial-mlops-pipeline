from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
from tqdm import tqdm

from geo_mlops.core.contracts.inference_contract import InferenceContract
from geo_mlops.core.inference.types import (
    InferenceConfig,
    InferencePrediction,
)
from geo_mlops.core.data.types import DiscoveredScene, SceneArrays
from geo_mlops.core.io.inference_io import write_inference_contract
from geo_mlops.core.utils.random import _seed_everything
from geo_mlops.core.utils.windows import(
    _ensure_bchw,
    build_grid,
    _pad_to_size,
    _extract_context_window,
    _squeeze_single_channel,
)
from geo_mlops.core.utils.dataclasses import _as_plain_dict


def run_full_scene_inference(
    *,
    task: str,
    model: torch.nn.Module,
    device: torch.device,
    scenes: Iterable[DiscoveredScene],
    inference_out_dir: Path,
    inference_cfg: InferenceConfig,
    load_scene_fn: Callable[[DiscoveredScene], SceneArrays],
    forward_fn: Callable[[torch.nn.Module, Dict[str, Any], torch.device], Any],
    postprocess_fn: Callable[[Any, Dict[str, Any]], InferencePrediction],
    save_prediction_fn: Callable[
        [DiscoveredScene, SceneArrays, InferencePrediction, Path, InferenceConfig],
        Any,
    ],
    checkpoint_path: Optional[Path] = None,
) -> Tuple[Path, InferenceContract]:
    _seed_everything(inference_cfg.seed)

    inference_out_dir = Path(inference_out_dir)
    inference_out_dir.mkdir(parents=True, exist_ok=True)

    tables_dir = inference_out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    scene_list = list(scenes)
    if not scene_list:
        raise ValueError("No inference scenes were provided.")

    model.to(device)
    model.eval()

    prediction_rows: list[Dict[str, Any]] = []

    with torch.no_grad():
        for scene in tqdm(scene_list, desc="Inference scenes"):
            arrays = load_scene_fn(scene)

            prediction = predict_scene_sliding_window(
                model=model,
                arrays=arrays,
                device=device,
                cfg=inference_cfg,
                forward_fn=forward_fn,
                postprocess_fn=postprocess_fn,
            )

            artifacts = save_prediction_fn(
                scene,
                arrays,
                prediction,
                inference_out_dir,
                inference_cfg,
            )

            prediction_rows.append(
                {
                    "scene_id": scene.scene_id,
                    "region": scene.region,
                    "subregion": scene.subregion,
                    "stem": scene.stem,
                    "pan_path": str(scene.pan_path),
                    "gt_path": str(scene.gt_path) if scene.gt_path else "",
                    "context_path": str(scene.context_path) if scene.context_path else "",
                    "probability_path": (str(artifacts.probability_path)),
                    "logits_path": (str(artifacts.logits_path)),
                }
            )

    prediction_table_path = tables_dir / "prediction_inventory.csv"
    pd.DataFrame(prediction_rows).to_csv(prediction_table_path, index=False)

    contract = InferenceContract(
        inference_dir_path=inference_out_dir,
        task=task,
        model_path=Path(checkpoint_path) if checkpoint_path else Path(""),
        num_scenes=len(scene_list),
        inference_cfg=_as_plain_dict(inference_cfg),
        prediction_table_path=prediction_table_path,
    )

    manifest_path = write_inference_contract(contract)

    return manifest_path, contract


def predict_scene_sliding_window(
    *,
    model: torch.nn.Module,
    arrays: SceneArrays,
    device: torch.device,
    cfg: InferenceConfig,
    forward_fn: Callable[[torch.nn.Module, Dict[str, Any], torch.device], Any],
    postprocess_fn: Callable[[Any, Dict[str, Any]], InferencePrediction],
) -> InferencePrediction:
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
    logits_sum: Optional[torch.Tensor] = None
    weight_sum: Optional[torch.Tensor] = None

    batch_tiles: List[torch.Tensor] = []
    batch_contexts: List[torch.Tensor] = []
    batch_windows: List[Tuple[int, int, int, int]] = []

    def flush_batch() -> None:
        nonlocal batch_tiles
        nonlocal batch_contexts
        nonlocal batch_windows
        nonlocal prob_sum
        nonlocal logits_sum
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
        logits = _ensure_bchw(pred.logits).to(device=device, dtype=torch.float32)

        if prob.shape[0] != len(batch_windows):
            raise ValueError(
                f"postprocess_fn returned probability batch size {prob.shape[0]}, "
                f"expected {len(batch_windows)}"
            )

        if logits.shape[0] != len(batch_windows):
            raise ValueError(
                f"postprocess_fn returned logits batch size {logits.shape[0]}, "
                f"expected {len(batch_windows)}"
            )

        if prob_sum is None:
            out_channels = int(prob.shape[1])
            prob_sum = torch.zeros(
                (out_channels, height, width),
                dtype=torch.float32,
                device=device,
            )
            logits_sum = torch.zeros(
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
        assert logits_sum is not None
        assert weight_sum is not None

        for i, (y0, x0, y1, x1) in enumerate(batch_windows):
            h = y1 - y0
            w = x1 - x0

            prob_crop = prob[i, :, :h, :w]
            logits_crop = logits[i, :, :h, :w]

            prob_sum[:, y0:y1, x0:x1] += prob_crop
            logits_sum[:, y0:y1, x0:x1] += logits_crop
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

    if prob_sum is None or logits_sum is None or weight_sum is None:
        raise RuntimeError("No windows were processed during scene prediction.")

    weight_sum = torch.clamp(weight_sum, min=1.0)
    prob_full = prob_sum / weight_sum
    logits_score_full = logits_sum / weight_sum

    prob_np = _squeeze_single_channel(prob_full.detach().cpu().numpy())
    logits_np = _squeeze_single_channel(logits_score_full.detach().cpu().numpy())

    return InferencePrediction(
        probability=prob_np,
        logits=logits_np,
        extra=None,
    )

