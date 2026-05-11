from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple
from geo_mlops.core.contracts.tile_contract import (
    TilesContract,
)
from geo_mlops.core.io.tile_io import write_tiles_contract, write_tiles_master_csv
from geo_mlops.core.tiling.adapters.base import TaskAdapter, TilingPolicy
from geo_mlops.core.tiling.engine import TilingEngineConfig, RoiTilingEngine
from geo_mlops.core.utils.dataclasses import _as_plain_dict
from geo_mlops.core.tiling.dataset_scanner import scan_full_dataset



def run_tiling_stage(
    *,
    task: str,
    dataset_root_path: Path,
    tiles_dir_path: Path,
    tile_engine_cfg: TilingEngineConfig,
    adapter: TaskAdapter,
    policy: TilingPolicy,
    force: bool = False,
    verbose: bool = False,
) -> Tuple[Path, TilesContract]:

    tiles_dir_path = Path(tiles_dir_path)
    tiles_dir_path.mkdir(parents=True, exist_ok=True)

    csv_name = f"{task}.csv"

    engine = RoiTilingEngine(
        cfg=tile_engine_cfg,
        adapter=adapter,
        policy=policy,
    )

    df, dataset_rois = scan_full_dataset(
        dataset_root_path=Path(dataset_root_path),
        engine=engine,
        csv_name=csv_name,
        force=force,
        verbose=verbose,
    )

    master_csv_path = tiles_dir_path / f"{task}_master.csv"
    write_tiles_master_csv(df, master_csv_path)

    contract = TilesContract(
        task=task,
        tiles_dir_path=tiles_dir_path,
        master_csv_path=master_csv_path,
        datasets_root_path=Path(dataset_root_path),
        dataset_rois=dataset_rois,
        tile_engine_cfg=_as_plain_dict(tile_engine_cfg),
        adapter={
            "module": type(adapter).__module__,
            "name": type(adapter).__name__,
        },
        policy={
            "module": type(policy).__module__,
            "name": type(policy).__name__,
        },
        row_count=int(len(df)),
    )

    manifest_path = write_tiles_contract(contract)

    return manifest_path, contract