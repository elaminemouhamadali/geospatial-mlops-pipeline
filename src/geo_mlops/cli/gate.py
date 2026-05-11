from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from geo_mlops.core.gating.stage import run_gate_stage
from geo_mlops.core.io.gate_io import GATE_MANIFEST_NAME


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a gating stage against a metrics file."
    )

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
        help="Unified task config YAML/JSON containing a `gating:` section.",
    )
    p.add_argument(
        "--gate-name",
        "--gate_name",
        dest="gate_name",
        type=str,
        required=True,
        help="Gate name in task config, e.g. gate_a or gate_b.",
    )
    p.add_argument(
        "--metrics-file-path",
        "--metrics_file_path",
        dest="metrics_file_path",
        type=Path,
        required=True,
        help="Metrics JSON/YAML file to evaluate.",
    )
    p.add_argument(
        "--gate-dir-path",
        "--gate_dir_path",
        dest="gate_dir_path",
        type=Path,
        required=True,
        help="Output directory for gate artifacts.",
    )

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    manifest_path, contract = run_gate_stage(
        task=args.task,
        gate_name=args.gate_name,
        task_cfg_path=args.task_cfg_path,
        metrics_file_path=args.metrics_file_path,
        gate_dir_path=args.gate_dir_path,
    )

    summary = contract.summary

    print(
        f"[gate] {contract.gate_name} | task={contract.task} | "
        f"decision={contract.decision} | passed={contract.passed}"
    )
    print(
        f"[gate] checks: total={summary.get('total_checks', len(contract.checks))} "
        f"passed={summary.get('passed_checks', 0)} "
        f"failed={summary.get('failed_checks', 0)}"
    )
    print(f"[gate] manifest={manifest_path}")

    if not contract.passed:
        print("[gate] result: FAILED")
        return 1

    print("[gate] result: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())