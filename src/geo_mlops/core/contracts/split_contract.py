from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


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

    task: str
    split_dir_path: Path

    train_regions: list[str]
    val_regions: list[str]

    split_cfg: dict[str, list[Any]]

    # optional typed partitions (e.g., water uses val_with_water/val_no_water)
    extra_partitions: dict[str, list[str]]

    artifacts: dict[str, str]
