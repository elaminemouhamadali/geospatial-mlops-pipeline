from __future__ import annotations

from typing import Any, Dict, Optional, Sequence
import pandas as pd

from geo_mlops.core.training.sampling import _apply_training_sampler
from geo_mlops.core.contracts.split_contract import SplitContract
from geo_mlops.core.contracts.tile_contract import TilesContract
from geo_mlops.tasks.segmentation.building.data.dataset import BuildingDataset, BuildingDatasetConfig

def build_dataset(
    *,
    tiles_df: pd.DataFrame,
    indices: Optional[Sequence[int]],
    cfg: Dict[str, Any],
    split_name: str,
) -> BuildingDataset:
    dataset_cfg = dict(cfg.get("dataset", {}) or {})

    # Never augment validation/evaluation data.
    if split_name != "train":
        dataset_cfg["do_aug"] = False

    return BuildingDataset(
        tiles_df=tiles_df,
        indices=indices,
        cfg=BuildingDatasetConfig.from_dict(dataset_cfg),
        cache_context=bool(dataset_cfg.get("cache_context", True)),
        context_cache_max_items=int(dataset_cfg.get("context_cache_max_items", 256)),
    )

def build_train_val_datasets(
    *,
    tiles: TilesContract,
    split: SplitContract,
    train_cfg: Dict[str, Any],
) -> tuple[BuildingDataset, BuildingDataset]:
    df = pd.read_csv(tiles.master_csv_path)

    split_cfg = split.split_cfg
    group_col = str(split_cfg.get("group_col", "scene_id"))

    if group_col not in df.columns:
        raise ValueError(
            f"Split group_col={group_col!r} not found in tiles CSV. "
            f"Available columns: {list(df.columns)}"
        )

    train_groups = set(map(str, split.train_regions))
    val_groups = set(map(str, split.val_regions))

    group_values = df[group_col].astype(str)

    train_indices = df.index[group_values.isin(train_groups)].tolist()
    val_indices = df.index[group_values.isin(val_groups)].tolist()

    if not train_indices:
        raise ValueError("No training rows matched split.train_regions.")
    if not val_indices:
        raise ValueError("No validation rows matched split.val_regions.")
    
    train_df = df.loc[train_indices].copy()
    val_df = df.loc[val_indices].copy()

    train_df = _apply_training_sampler(train_df, train_cfg)

    train_ds = build_dataset(
        tiles_df=train_df.reset_index(drop=True),
        indices=None,
        cfg=train_cfg,
        split_name="train",
    )

    val_ds = build_dataset(
        tiles_df=val_df.reset_index(drop=True),
        indices=None,
        cfg=train_cfg,
        split_name="val",
    )

    return train_ds, val_ds