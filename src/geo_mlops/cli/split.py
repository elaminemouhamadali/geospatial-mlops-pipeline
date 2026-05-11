from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from geo_mlops.core.registry.task_registry import get_task
from geo_mlops.core.splitting.stage import run_split_stage


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create deterministic group-aware train/val(/test) splits from a tiling output directory.")
    p.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task name, e.g. building_seg.",
    )
    p.add_argument(
        "--task-cfg-path",
        "--task_cfg_path",
        dest="task_cfg_path",
        type=Path,
        required=True,
        help="Unified task config YAML/JSON containing a `splitting:` section.",
    )
    p.add_argument(
        "--tiles-manifest-path",
        "--tiles_manifest_path",
        dest="tiles_manifest_path",
        type=Path,
        required=True,
        help="Tiling output directory containing tiles_manifest.json.",
    )
    p.add_argument(
        "--split-dir-path",
        "--split_dir_path",
        dest="split_dir_path",
        type=Path,
        required=True,
        help="Output directory for split artifacts.",
    )

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    task_plugin = get_task(args.task)

    splitting_cfg = task_plugin.build_splitting_cfg(task_cfg_path=args.task_cfg_path)

    split_engine_cfg, group_list_prefix = task_plugin.build_split_engine_cfg(splitting_cfg)

    manifest_path, contract = run_split_stage(
        task=args.task,
        split_engine_cfg=split_engine_cfg,
        group_list_prefix=group_list_prefix,
        tiles_manifest_path=args.tiles_manifest_path,
        split_dir_path=args.split_dir_path,
    )

    print(f"[split] wrote split contract -> {manifest_path}")
    print(f"[split] train groups: {len(contract.train_regions)}")
    print(f"[split] val groups: {len(contract.val_regions)}")

    if contract.extra_partitions:
        for name, groups in contract.extra_partitions.items():
            print(f"[split] {name} groups: {len(groups)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
