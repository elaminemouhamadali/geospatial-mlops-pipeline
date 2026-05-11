from __future__ import annotations

from typing import Any, Dict
import pandas as pd


def _apply_training_sampler(
    df: pd.DataFrame,
    train_cfg: Dict[str, Any],
) -> pd.DataFrame:
    sampler_cfg = train_cfg.get("sampler", {}) or {}
    kind = str(sampler_cfg.get("kind", "all")).lower().strip()

    if kind == "all":
        return df

    if kind in ("regular", "hard_mining", "policy"):
        col = str(sampler_cfg.get("include_col", "sample__include"))
        if col not in df.columns:
            raise ValueError(
                f"training.sampler requested {kind!r}, but column {col!r} "
                f"is missing from the tile CSV."
            )

        keep = df[col].astype(str).str.lower().isin(("true", "1", "yes"))
        out = df.loc[keep].copy()

        if out.empty:
            raise ValueError(
                f"Training sampler kept 0 rows using column {col!r}. "
                "Check tiling policy thresholds or use training.sampler.kind=all."
            )

        return out

    raise ValueError(
        f"Unknown training.sampler.kind={kind!r}. "
        "Expected one of: all, regular, hard_mining, policy."
    )