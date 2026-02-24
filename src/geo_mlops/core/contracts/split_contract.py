from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


# -------------------------
# Split stage contract
# -------------------------
@dataclass(frozen=True)
class SplitContract:
    """
    Output of make_splits stage as consumed by training.

    This is intentionally task-agnostic. Tasks may interpret the fields
    (e.g., water uses region lists to filter tiles), but core doesn't.
    """

    split_dir: Path

    train_regions: List[str]
    val_regions: List[str]

    # optional typed partitions (e.g., water uses val_with_water/val_no_water)
    extra_partitions: Dict[str, List[str]]

    # optional split metadata from split.json
    meta: Dict[str, Any]
