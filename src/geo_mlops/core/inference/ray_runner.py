# src/geo_mlops/core/inference/ray_runner.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import pandas as pd
import torch

from geo_mlops.core.contracts.inference_contract import InferenceContract
from geo_mlops.core.data.types import DiscoveredScene, SceneArrays
from geo_mlops.core.execution.ray_backend import require_ray_initialized
from geo_mlops.core.execution.sharding import shard_sequence
from geo_mlops.core.inference.engine import process_inference_scenes
from geo_mlops.core.inference.types import InferenceConfig, InferencePrediction
from geo_mlops.core.io.inference_io import write_inference_contract
from geo_mlops.core.utils.dataclasses import _as_plain_dict


def run_full_scene_inference_ray(
    *,
    task: str,
    scenes: Iterable[DiscoveredScene],
    inference_out_dir: Path,
    inference_cfg: InferenceConfig,
    model_factory_fn: Callable[[], torch.nn.Module],
    load_checkpoint_fn: Callable[..., torch.nn.Module],
    checkpoint_path: Optional[Path],
    device: str,
    load_scene_fn: Callable[[DiscoveredScene], SceneArrays],
    forward_fn: Callable[[torch.nn.Module, Dict[str, Any], torch.device], Any],
    postprocess_fn: Callable[[Any, Dict[str, Any]], InferencePrediction],
    save_prediction_fn: Callable[
        [DiscoveredScene, SceneArrays, InferencePrediction, Path, InferenceConfig],
        Any,
    ],
    num_workers: Optional[int] = None,
    items_per_shard: Optional[int] = None,
    num_cpus_per_worker: int = 4,
    num_gpus_per_worker: float = 1.0,
) -> Tuple[Path, InferenceContract]:
    """
    Distributed full-scene inference.

    Core responsibilities:
      - shard discovered scenes
      - run process_inference_scenes(...) inside Ray workers
      - write one shard inventory CSV per worker
      - merge shard inventories
      - write one final inference contract

    This runner intentionally does not know about task plugins.
    The CLI/task layer supplies model/loading/postprocess/save callbacks.
    """
    import ray

    require_ray_initialized()

    inference_out_dir = Path(inference_out_dir)
    inference_out_dir.mkdir(parents=True, exist_ok=True)

    tables_dir = inference_out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    shard_dir = inference_out_dir / "_ray_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    scene_list = list(scenes)
    if not scene_list:
        raise ValueError("No inference scenes were provided.")

    scene_shards = shard_sequence(
        scene_list,
        num_shards=num_workers,
        items_per_shard=items_per_shard,
        drop_empty=True,
    )

    if not scene_shards:
        raise ValueError("Scene sharding produced no shards.")

    remote_worker = ray.remote(
        num_cpus=num_cpus_per_worker,
        num_gpus=num_gpus_per_worker,
    )(_run_inference_shard)

    refs = []
    for shard_idx, shard_scenes in enumerate(scene_shards):
        refs.append(
            remote_worker.remote(
                shard_idx=shard_idx,
                scenes=shard_scenes,
                inference_out_dir=inference_out_dir,
                inference_cfg=inference_cfg,
                model_factory_fn=model_factory_fn,
                load_checkpoint_fn=load_checkpoint_fn,
                checkpoint_path=checkpoint_path,
                device=device,
                load_scene_fn=load_scene_fn,
                forward_fn=forward_fn,
                postprocess_fn=postprocess_fn,
                save_prediction_fn=save_prediction_fn,
                shard_dir=shard_dir,
            )
        )

    shard_table_paths = ray.get(refs)

    dfs = []
    for shard_table_path in shard_table_paths:
        shard_table_path = Path(shard_table_path)
        if shard_table_path.exists():
            df = pd.read_csv(shard_table_path)
            if not df.empty:
                dfs.append(df)

    prediction_df = (
        pd.concat(dfs, ignore_index=True)
        if dfs
        else pd.DataFrame()
    )

    prediction_table_path = tables_dir / "prediction_inventory.csv"
    prediction_df.to_csv(prediction_table_path, index=False)

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


def _run_inference_shard(
    *,
    shard_idx: int,
    scenes: list[DiscoveredScene],
    inference_out_dir: Path,
    inference_cfg: InferenceConfig,
    model_factory_fn: Callable[[], torch.nn.Module],
    load_checkpoint_fn: Callable[..., torch.nn.Module],
    checkpoint_path: Optional[Path],
    device: str,
    load_scene_fn: Callable[[DiscoveredScene], SceneArrays],
    forward_fn: Callable[[torch.nn.Module, Dict[str, Any], torch.device], Any],
    postprocess_fn: Callable[[Any, Dict[str, Any]], InferencePrediction],
    save_prediction_fn: Callable[
        [DiscoveredScene, SceneArrays, InferencePrediction, Path, InferenceConfig],
        Any,
    ],
    shard_dir: Path,
) -> str:
    """
    Ray worker.

    Writes only this shard's prediction inventory.
    The driver writes the final merged inventory + contract.
    """
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    device_obj = torch.device(
        "cuda" if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    )

    model = model_factory_fn()

    if checkpoint_path is not None:
        model = load_checkpoint_fn(
            model=model,
            checkpoint_path=checkpoint_path,
            device=device_obj,
        )
    else:
        model.to(device_obj)
        model.eval()

    prediction_rows, _ = process_inference_scenes(
        model=model,
        device=device_obj,
        scenes=scenes,
        inference_out_dir=inference_out_dir,
        inference_cfg=inference_cfg,
        load_scene_fn=load_scene_fn,
        forward_fn=forward_fn,
        postprocess_fn=postprocess_fn,
        save_prediction_fn=save_prediction_fn,
    )

    shard_table_path = shard_dir / f"shard_{shard_idx:04d}_prediction_inventory.csv"
    pd.DataFrame(prediction_rows).to_csv(shard_table_path, index=False)

    return str(shard_table_path)