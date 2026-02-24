from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from geo_mlops.core.contracts.split_contract import SplitContract

SPLIT_JSON_NAME = "split.json"


def load_split_contract(split_dir: Path) -> SplitContract:
    """
    Load a SplitContract from a make_splits output directory.
    Canonical source is `split.json`
    """
    split_dir = Path(split_dir)
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    split_json_path = split_dir / SPLIT_JSON_NAME
    if not split_json_path.exists():
        raise FileNotFoundError(f"Missing {SPLIT_JSON_NAME} in: {split_dir}")

    data: Dict[str, Any] = json.loads(split_json_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{split_json_path} root must be a JSON object/mapping.")

    # ---- regions ----
    train_regions = data.get("train_regions")
    val_regions = data.get("val_regions")

    if not isinstance(train_regions, list) or not all(isinstance(x, str) for x in train_regions):
        raise ValueError(f"Invalid train_regions in {split_json_path}: expected list[str].")
    if not isinstance(val_regions, list) or not all(isinstance(x, str) for x in val_regions):
        raise ValueError(f"Invalid val_regions in {split_json_path}: expected list[str].")

    # ---- extra partitions ----
    extra_partitions = data.get("extra_partitions")
    if extra_partitions is None:
        extra_partitions = {}
    if not isinstance(extra_partitions, dict):
        raise ValueError(f"Invalid extra_partitions in {split_json_path}: expected dict[str, list[str]].")

    cleaned_partitions: Dict[str, List[str]] = {}
    for k, v in extra_partitions.items():
        if not isinstance(k, str):
            continue
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            cleaned_partitions[k] = v
        else:
            # tolerate empty/invalid partitions but keep schema consistent
            cleaned_partitions[k] = []

    # ---- meta ----
    # Prefer explicit meta if present
    meta = data.get("meta", None)
    if meta is None:
        meta = {
            k: v
            for k, v in data.items()
            if k not in ("train_regions", "val_regions", "train", "val", "extra_partitions", "partitions")
        }
    if not isinstance(meta, dict):
        meta = {"meta_raw": meta}

    return SplitContract(
        split_dir=split_dir.resolve(),
        train_regions=list(train_regions),
        val_regions=list(val_regions),
        extra_partitions=cleaned_partitions,
        meta=dict(meta),
    )


def write_split_contract(contract: SplitContract, *, split_json_name: str = SPLIT_JSON_NAME) -> Path:
    """
    Write `split.json` for a SplitContract.
    a canonical writer shared across scripts.
    """
    split_dir = Path(contract.split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    out_path = split_dir / split_json_name

    payload: Dict[str, Any] = {
        "train_regions": list(contract.train_regions),
        "val_regions": list(contract.val_regions),
        "extra_partitions": dict(contract.extra_partitions),
        "meta": dict(contract.meta),
    }

    out_path.write_text(json.dumps(payload, indent=2))
    return out_path
