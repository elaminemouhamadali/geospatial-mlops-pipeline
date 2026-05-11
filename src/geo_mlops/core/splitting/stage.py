from __future__ import annotations

from pathlib import Path

from geo_mlops.core.contracts.split_contract import SplitContract
from geo_mlops.core.io.split_io import (
    write_split_contract,
    write_split_artifacts,
    build_split_contract,
)
from geo_mlops.core.io.tile_io import load_tiles_contract
from geo_mlops.core.splitting.split import (
    make_splits_from_csvs,
    SplitConfig,
)


def run_split_stage(
    *,
    task: str,
    split_engine_cfg: SplitConfig,
    group_list_prefix: str,
    tiles_manifest_path: Path,
    split_dir_path: Path,
) -> tuple[Path, SplitContract]:
    
    split_dir_path = Path(split_dir_path)
    split_dir_path.mkdir(parents=True, exist_ok=True)

    tiles_contract = load_tiles_contract(tiles_manifest_path)

    result = make_splits_from_csvs(
        [tiles_contract.master_csv_path],
        config=split_engine_cfg,
    )

    artifacts = write_split_artifacts(
        result=result,
        split_dir_path=split_dir_path,
        group_col=split_engine_cfg.group_col,
        group_list_prefix=group_list_prefix,
    )

    contract = build_split_contract(
        task=task,
        split_dir_path=split_dir_path,
        cfg=split_engine_cfg,
        result=result,
        artifacts=artifacts,
    )

    manifest_path = write_split_contract(contract)

    return manifest_path, contract