from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from geo_mlops.core.utils.dataclasses import (
    _as_plain_dict, 
    _load_json,
    _to_jsonable,
    _unique_group_values
)
from geo_mlops.core.contracts.split_contract import SplitContract
from geo_mlops.core.splitting.split import SplitConfig, SplitResult

SPLIT_MANIFEST_NAME = "split.json"


def load_split_contract(manifest_path: Path) -> SplitContract:
    """
    Load a SplitContract from a make_splits output directory.
    Canonical source is `split.json`
    """

    data: Dict[str, Any] = _load_json(manifest_path)

    return SplitContract(
        task=data["task"],
        split_dir_path=Path(data["split_dir_path"]),
        train_regions=list(data.get("train_regions", [])),
        val_regions=list(data.get("val_regions", [])),
        split_cfg=dict(data.get("split_cfg", {})),
        extra_partitions={
            str(k): list(map(str, v))
            for k, v in dict(data.get("extra_partitions", {})).items()
        },
        artifacts=dict(data.get("artifacts", {})),
    )


def write_split_contract(contract: SplitContract, *, manifest_name: str = SPLIT_MANIFEST_NAME) -> Path:
    """
    Write `split.json` for a SplitContract.
    a canonical writer shared across scripts.
    """
    contract.split_dir_path.mkdir(parents=True, exist_ok=True)
    manifest_path = contract.split_dir_path / manifest_name

    payload = _as_plain_dict(_to_jsonable(contract))

    manifest_path.write_text(json.dumps(payload, indent=2))
    return manifest_path


def write_split_artifacts(
    *,
    result: SplitResult,
    split_dir_path: Path,
    group_col: str,
    group_list_prefix: str,
) -> dict[str, str]:
    split_dir_path = Path(split_dir_path)
    split_dir_path.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, str] = {}

    train_groups = _unique_group_values(result.train, group_col)
    val_groups = _unique_group_values(result.val, group_col)

    train_path = split_dir_path / f"train_{group_list_prefix}.txt"
    val_path = split_dir_path / f"val_{group_list_prefix}.txt"

    train_path.write_text("\n".join(train_groups) + "\n")
    val_path.write_text("\n".join(val_groups) + "\n")

    artifacts["train_groups_path"] = str(train_path)
    artifacts["val_groups_path"] = str(val_path)

    group_stats_path = split_dir_path / "group_stats.csv"
    result.group_stats.to_csv(group_stats_path, index=False)
    artifacts["group_stats_path"] = str(group_stats_path)

    return artifacts

def build_split_contract(
    *,
    task: str,
    split_dir_path: Path,
    cfg: SplitConfig,
    result: SplitResult,
    artifacts: dict[str, str],
) -> SplitContract:
    group_col = result.resolved_group_col or cfg.group_col

    train_groups = _unique_group_values(result.train, group_col)
    val_groups = _unique_group_values(result.val, group_col)

    checks = [
        {
            "name": check.name,
            "ok": bool(check.ok),
            "details": dict(check.details),
        }
        for check in result.checks
    ]

    split_cfg = {
        "policy": cfg.policy,
        "group_col": cfg.group_col,
        "resolved_group_col": group_col,
        "seed": cfg.seed,
        "ratios": cfg.ratios.as_dict(),
        "predefined_col": cfg.predefined_col,
        "min_any_ratio": cfg.min_any_ratio,
        "ratio_cols": cfg.ratio_cols,
        "dedupe_key": cfg.dedupe_key,
        "group_metric_mode": cfg.group_metric_mode,
        "group_metric_col": cfg.group_metric_col,
        "presence_eps": cfg.presence_eps,
        "bins": cfg.bins,
        "counts": {
            "train_groups": len(train_groups),
            "val_groups": len(val_groups),
        },
        "warnings": list(result.warnings),
        "checks": checks,
    }

    return SplitContract(
        task=task,
        split_dir_path=Path(split_dir_path),
        train_regions=train_groups,
        val_regions=val_groups,
        split_cfg=split_cfg,
        extra_partitions={},
        artifacts=dict(artifacts),
    )
