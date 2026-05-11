from __future__ import annotations

from pathlib import Path
from typing import Any

from geo_mlops.core.config.loader import load_cfg, require_section
from geo_mlops.core.splitting.split import SplitConfig, parse_ratios

_VALID_SPLIT_POLICIES = {"grouped", "stratified", "predefined"}


def build_splitting_cfg(task_cfg_path: str | Path) -> dict[str, Any]:
    cfg = load_cfg(task_cfg_path)
    return require_section(cfg, "splitting")


def build_split_engine_cfg(splitting_cfg: dict[str, Any]) -> tuple[SplitConfig, str]:
    engine_cfg = splitting_cfg.get("engine", {}) or {}
    outputs_cfg = splitting_cfg.get("outputs", {}) or {}

    splits = engine_cfg.get("splits", {}) or {}

    train = float(splits.get("train", 0.8))
    val = float(splits.get("val", 0.2))
    test = float(splits.get("test", 0.0))

    ratios = parse_ratios([train, val, test] if test > 0 else [train, val])

    policy = str(engine_cfg.get("policy", "grouped")).strip().lower()
    if policy not in _VALID_SPLIT_POLICIES:
        raise ValueError(f"Invalid splitting.engine.policy={policy!r}. Expected one of: {sorted(_VALID_SPLIT_POLICIES)}")

    bins_raw = engine_cfg.get("bins")
    bins = [float(x) for x in bins_raw] if bins_raw is not None else None

    split_engine_cfg = SplitConfig(
        policy=policy,
        seed=int(engine_cfg.get("seed", 42)),
        ratios=ratios,
        group_col=str(engine_cfg.get("group_col", "scene_id")),
        predefined_col=str(engine_cfg.get("predefined_col", "split")),
        min_any_ratio=engine_cfg.get("min_any_ratio"),
        ratio_cols=engine_cfg.get("ratio_cols"),
        dedupe_key=engine_cfg.get("dedupe_key"),
        group_metric_mode=engine_cfg.get("group_metric_mode"),
        group_metric_col=engine_cfg.get("group_metric_col"),
        presence_eps=float(engine_cfg.get("presence_eps", 0.001)),
        bins=bins,
        prefix=str(outputs_cfg.get("prefix", "tiles")),
    )

    group_list_prefix = str(
        outputs_cfg.get(
            "group_list_prefix",
            outputs_cfg.get("prefix", "tiles"),
        )
    )

    return split_engine_cfg, group_list_prefix
