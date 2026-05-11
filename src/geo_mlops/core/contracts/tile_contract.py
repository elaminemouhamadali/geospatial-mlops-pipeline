from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

TILES_MANIFEST_NAME = "tiles_manifest.json"


@dataclass(frozen=True)
class TilesContract:
    """
    Output of the tiling stage (generate_tiles_csv) as consumed by downstream stages.
    Canonical artifact is the master tile CSV.
    Per-subdir CSVs are cache artifacts and not required for the contract.
    """

    task: str
    tiles_dir_path: Path  # directory that contains master CSV + manifest
    master_csv_path: Path | str  # canonical CSV path
    datasets_root_path: Path
    dataset_rois: list[str]
    tile_engine_cfg: dict[str, Any]  # TilingEngineConfig serialized to dict
    adapter: dict[str, str]  # {"module": "...", "name": "..."}
    policy: dict[str, str]  # {"module": "...", "name": "..."}
    row_count: int
