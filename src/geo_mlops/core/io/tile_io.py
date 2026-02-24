from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from geo_mlops.core.contracts.tile_contract import (
    TILES_MANIFEST_NAME,
    TILES_SCHEMA_VERSION_V1,
    TilesContract,
)


def write_tiles_contract(contract: TilesContract, *, manifest_name: str = TILES_MANIFEST_NAME) -> Path:
    """
    Serialize TilesContract to JSON in contract.tiles_dir/tiles_manifest.json (by default).
    """
    contract.tiles_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = contract.tiles_dir / manifest_name

    payload = asdict(contract)

    # stringify Paths for JSON
    for k, v in list(payload.items()):
        if isinstance(v, Path):
            payload[k] = str(v)
        elif isinstance(v, list):
            payload[k] = [str(x) if isinstance(x, Path) else x for x in v]

    manifest_path.write_text(json.dumps(payload, indent=2))
    return manifest_path


def load_tiles_contract(tiles_dir: Path, *, manifest_name: str = TILES_MANIFEST_NAME) -> TilesContract:
    """
    Load TilesContract from tiles_dir/tiles_manifest.json.
    Performs light validation and resolves paths relative to tiles_dir if needed.
    """
    tiles_dir = Path(tiles_dir)
    manifest_path = tiles_dir / manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing tiling manifest: {manifest_path}")

    data: Dict[str, Any] = json.loads(manifest_path.read_text())

    schema_version = data.get("schema_version", "")
    if schema_version not in (TILES_SCHEMA_VERSION_V1,):
        raise ValueError(f"Unsupported tiles schema_version={schema_version!r} in {manifest_path}")

    master_csv_raw = data.get("master_csv")
    if not master_csv_raw:
        raise ValueError(f"tiles manifest missing 'master_csv': {manifest_path}")
    master_csv = Path(master_csv_raw)

    return TilesContract(
        tiles_dir=tiles_dir.resolve(),
        master_csv=master_csv,
        schema_version=schema_version,
        task=str(data.get("task", "")),
        datasets_root=Path(data.get("datasets_root", "")),
        dataset_buckets=list(data.get("dataset_buckets", [])),
        regions=data.get("regions", None),
        engine_cfg=dict(data.get("engine_cfg", {})),
        adapter=dict(data.get("adapter", {})),
        policy=dict(data.get("policy", {})),
        csv_name=str(data.get("csv_name", "")),
        row_count=int(data.get("row_count", 0)),
        meta=dict(data.get("meta", {})),
    )
