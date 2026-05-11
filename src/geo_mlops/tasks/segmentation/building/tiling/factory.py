from __future__ import annotations

from pathlib import Path
from typing import Any

from geo_mlops.core.config.loader import load_cfg, require_section
from geo_mlops.core.tiling.engine import TilingEngineConfig
from geo_mlops.core.tiling.factory import build_tiling_engine_cfg, build_tiling_policy
from geo_mlops.core.tiling.policies import AllPolicy, HardMiningPolicy, RegularPolicy
from geo_mlops.tasks.segmentation.building.tiling.adapter import BuildingSegmentationAdapter

TilingPolicyT = AllPolicy | RegularPolicy | HardMiningPolicy


def build_building_adapter(adapter_cfg: dict[str, Any]) -> BuildingSegmentationAdapter:
    return BuildingSegmentationAdapter(
        class_of_interest_id=int(adapter_cfg.get("class_of_interest_id")),
        shadow_id=adapter_cfg.get("shadow_id"),
        emit_shadow=bool(adapter_cfg.get("emit_shadow")),
        min_change_pixels=int(adapter_cfg.get("min_change_pixels")),
    )


def build_tiling_components(
    task_cfg_path: str | Path,
) -> tuple[TilingEngineConfig, BuildingSegmentationAdapter, TilingPolicyT, dict[str, Any]]:
    cfg = load_cfg(task_cfg_path)
    tiling_cfg = require_section(cfg, "tiling")

    engine_cfg = require_section(tiling_cfg, "engine")
    adapter_cfg = require_section(tiling_cfg, "adapter")
    policy_cfg = require_section(tiling_cfg, "policy")
    meta = tiling_cfg.get("meta", {}) or {}

    engine = build_tiling_engine_cfg(engine_cfg)
    adapter = build_building_adapter(adapter_cfg)
    policy = build_tiling_policy(policy_cfg)

    return engine, adapter, policy, meta
