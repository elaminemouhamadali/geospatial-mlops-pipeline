from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd


def write_subdir_csv(
    subdir: Path,
    rows: list[dict[str, Any]],
    csv_name: str,
) -> Path | None:
    if not rows:
        return None

    df = pd.DataFrame(rows)

    sort_keys = [k for k in ("image_src", "y0", "x0") if k in df.columns]
    if sort_keys:
        df.sort_values(sort_keys, inplace=True, ignore_index=True)

    out_path = subdir / csv_name
    df.to_csv(out_path, index=False)
    return out_path


def _ensure_bucket_column(df: pd.DataFrame, bucket_name: str) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    if "dataset_bucket" not in df.columns:
        df["dataset_bucket"] = bucket_name
    else:
        df["dataset_bucket"] = df["dataset_bucket"].fillna(bucket_name)

    return df


def _sort_master(
    df: pd.DataFrame,
    sort_keys: Sequence[str] = (
        "region",
        "subregion",
        "image_src",
        "y0",
        "x0",
    ),
) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    keys = [k for k in sort_keys if k in df.columns]

    if keys:
        df.sort_values(keys, inplace=True, ignore_index=True)

    return df


def _merge_stats(dst: dict[str, Any], src: dict[str, Any]) -> None:
    """
    Merge integer/bool stats without clobbering identity fields.
    """
    identity_keys = {
        "roi",
        "sub_roi",
    }

    for k, v in src.items():
        if k in identity_keys:
            dst.setdefault(k, v)
        elif isinstance(v, bool):
            dst[k] = bool(dst.get(k, False) or v)
        elif isinstance(v, int):
            dst[k] = int(dst.get(k, 0)) + int(v)
        elif k not in dst:
            dst[k] = v
