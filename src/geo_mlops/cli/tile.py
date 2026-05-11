from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from geo_mlops.core.registry.task_registry import get_task
from geo_mlops.core.tiling.stage import run_tiling_stage


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=("Generate tile CSVs using a task-agnostic tiling engine and task-specific adapter/policy components."))
    ap.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task key registered in task_registry, e.g. building_seg.",
    )
    ap.add_argument(
        "--task-cfg-path",
        "--task_cfg_path",
        dest="task_cfg_path",
        type=Path,
        required=True,
        help="Unified task config YAML/JSON.",
    )
    ap.add_argument(
        "--dataset-root-path",
        "--dataset_root_path",
        dest="dataset_root_path",
        type=Path,
        required=True,
        help="Directory containing dataset bucket subdirectories.",
    )
    ap.add_argument(
        "--tiles-dir-path",
        "--tiles_dir_path",
        type=Path,
        default=None,
        help="Output directory for master CSV and tiles manifest.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Recompute per-subdir CSVs even if they already exist.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress logs.",
    )

    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    task_plugin = get_task(args.task)

    tile_engine_cfg, adapter, policy, _ = task_plugin.build_tiling_components(
        task_cfg_path=args.task_cfg_path,
    )

    manifest_path, contract = run_tiling_stage(
        task=args.task,
        dataset_root_path=args.dataset_root_path,
        tiles_dir_path=args.tiles_dir_path,
        tile_engine_cfg=tile_engine_cfg,
        adapter=adapter,
        policy=policy,
        force=args.force,
        verbose=args.verbose,
    )

    print(f"[tiling] wrote {contract.row_count} rows")
    print(f"[master] {contract.master_csv_path}")
    print(f"[manifest] {manifest_path}")
    print("Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
