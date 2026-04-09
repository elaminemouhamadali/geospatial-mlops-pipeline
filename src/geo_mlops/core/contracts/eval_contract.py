from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

EVAL_SCHEMA_VERSION_V1 = "eval.v1"


@dataclass(frozen=True)
class EvalContract:
    eval_dir: Path
    schema_version: str

    task: str
    split_name: str                 # e.g. "golden_test", "val", "test"

    metrics_path: Path
    model_path: Path

    num_eval_tiles: int
    group_col: str                  # e.g. "region", "scene_id"
    selection_source: Optional[Path]  # txt/json file used to define evaluation subset

    metrics: Dict[str, Any]         # canonical summary metrics for quick access
    upstream: Dict[str, Any]        # train_manifest, split_json, tiles_manifest, train_cfg, etc.
    meta: Dict[str, Any]