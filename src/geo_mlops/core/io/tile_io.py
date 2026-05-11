from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
from geo_mlops.core.utils.dataclasses import _as_plain_dict, _load_json, _to_jsonable

from geo_mlops.core.contracts.tile_contract import (
    TILES_MANIFEST_NAME,
    TilesContract,
)


def write_tiles_contract(contract: TilesContract, *, manifest_name: str = TILES_MANIFEST_NAME) -> Path:
    """
    Serialize TilesContract to JSON in contract.manifest_dir_path/tiles_manifest.json (by default).
    """
    contract.tiles_dir_path.mkdir(parents=True, exist_ok=True)
    manifest_path = contract.tiles_dir_path / manifest_name

    payload = _as_plain_dict(_to_jsonable(contract))

    manifest_path.write_text(json.dumps(payload, indent=2))
    return manifest_path


def load_tiles_contract(manifest_path: Path) -> TilesContract:
    """
    Load TilesContract from tiles_dir/tiles_manifest.json.
    Performs light validation and resolves paths relative to tiles_dir if needed.
    """
    data = _load_json(manifest_path)

    return TilesContract(
        task=data["task"],
        tiles_dir_path=Path(data["tiles_dir_path"]),
        master_csv_path=Path(data["master_csv_path"]),
        datasets_root_path=Path(data["datasets_root_path"]),
        dataset_buckets=list(data["dataset_buckets"]),
        tile_engine_cfg=dict(data["tile_engine_cfg"]),
        adapter=dict(data["adapter"]),
        policy=dict(data["policy"]),
        row_count=int(data["row_count"]),
    )


def load_tiles_manifest_and_table(manifest_path: Path) -> tuple[TilesContract, pd.DataFrame]:
    contract = load_tiles_contract(manifest_path)
    df = pd.read_csv(contract.master_csv_path)
    return contract, df

def write_tiles_master_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path

