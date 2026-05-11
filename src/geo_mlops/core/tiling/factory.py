from __future__ import annotations

from typing import Any, Dict

from geo_mlops.core.tiling.engine import TilingEngineConfig
from geo_mlops.core.tiling.policies import AllPolicy, HardMiningPolicy, RegularPolicy

TilingPolicyT = AllPolicy | RegularPolicy | HardMiningPolicy


def build_tiling_engine_cfg(engine_cfg: Dict[str, Any]) -> TilingEngineConfig:
    return TilingEngineConfig(
        pan_dirname=str(engine_cfg.get("pan_dirname", "PAN")),
        gt_dirname=str(engine_cfg.get("gt_dirname", "GT")),
        preds_dirname=engine_cfg.get("preds_dirname"),
        context_dirname=engine_cfg.get("context_dirname"),
        target_size_m=float(engine_cfg.get("target_size_m", 250.0)),
        overlap=float(engine_cfg.get("overlap", 0.5)),
        reflectance_max=int(engine_cfg.get("reflectance_max", 10_000)),
        context_max_side_cap=engine_cfg.get("context_max_side_cap"),
        verbose=bool(engine_cfg.get("verbose", False)),
        skip_tiles_with_nodata=bool(engine_cfg.get("skip_tiles_with_nodata", True)),
    )


def build_tiling_policy(policy_cfg: Dict[str, Any]) -> TilingPolicyT:
    kind = str(policy_cfg.get("kind", "all")).lower().strip()

    if kind == "all":
        return AllPolicy()

    if kind == "regular":
        return RegularPolicy(
            gt_presence_threshold=float(policy_cfg.get("gt_presence_threshold", 0.0)),
            require_presence=bool(policy_cfg.get("require_presence", False)),
            details_prefix=str(policy_cfg.get("details_prefix", "presence__")),
        )

    if kind == "hard_mining":
        if "min_difficulty" not in policy_cfg:
            raise ValueError("tiling.policy.min_difficulty is required for hard_mining policy.")

        return HardMiningPolicy(
            min_difficulty=float(policy_cfg["min_difficulty"]),
            include_if_gt_present=bool(policy_cfg.get("include_if_gt_present", True)),
            gt_presence_threshold=float(policy_cfg.get("gt_presence_threshold", 0.0)),
            presence_prefix=str(policy_cfg.get("presence_prefix", "presence__")),
            difficulty_prefix=str(policy_cfg.get("difficulty_prefix", "difficulty__")),
        )

    raise ValueError(
        f"Unknown tiling policy kind={kind!r}. "
        "Expected one of: all, regular, hard_mining."
    )