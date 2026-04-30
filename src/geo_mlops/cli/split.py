"""
Task-agnostic MLOps CLI: create train/val(/test) splits from a tiling stage output directory.

Input:
- A unified task config containing a `splitting:` section
- A tiling output directory containing:
    - tiles_manifest.json
    - master tiles CSV referenced by the manifest

Outputs:
- split.json       canonical SplitContract
- group_stats.csv  optional debugging artifact
- train_<prefix>.txt / val_<prefix>.txt optional convenience artifacts
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from geo_mlops.core.splitting.stage import run_split_stage


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create deterministic group-aware train/val(/test) splits from a tiling output directory."
    )

    p.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task name, e.g. building_seg.",
    )
    p.add_argument(
        "--task-cfg",
        "--task_cfg",
        dest="task_cfg",
        type=Path,
        required=True,
        help="Unified task config YAML/JSON containing a `splitting:` section.",
    )
    p.add_argument(
        "--tiles-dir",
        "--tiles_dir",
        dest="tiles_dir",
        type=Path,
        required=True,
        help="Tiling output directory containing tiles_manifest.json.",
    )
    p.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        type=Path,
        required=True,
        help="Output directory for split artifacts.",
    )
    p.add_argument(
        "--no-group-lists",
        action="store_true",
        help="Do not write train_*.txt / val_*.txt.",
    )
    p.add_argument(
        "--no-group-stats",
        action="store_true",
        help="Do not write group_stats.csv.",
    )

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    contract = run_split_stage(
        task=args.task,
        task_cfg_path=args.task_cfg,
        tiles_dir=args.tiles_dir,
        out_dir=args.out_dir,
        write_group_lists=not args.no_group_lists,
        write_group_stats=not args.no_group_stats,
    )

    print(f"[split] wrote split contract -> {contract.split_dir / 'split.json'}")
    print(f"[split] train groups: {len(contract.train_regions)}")
    print(f"[split] val groups: {len(contract.val_regions)}")

    if contract.extra_partitions:
        for name, groups in contract.extra_partitions.items():
            print(f"[split] {name} groups: {len(groups)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())