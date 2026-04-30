from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd
import torch
import yaml

from geo_mlops.core.contracts.eval_contract import (
    EVAL_SCHEMA_VERSION_V1,
    EvalContract,
)
from geo_mlops.core.io.eval_io import summarize_eval_contract, write_eval_contract
from geo_mlops.core.io.tile_io import load_tiles_contract
from geo_mlops.core.evaluation import load_groups_file, run_evaluation


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run evaluation on a held-out split (e.g. golden_test) using a trained model."
    )

    p.add_argument("--task", type=str, required=True, help="Task name, e.g. building_seg")
    p.add_argument("--tiles-manifest", type=Path, required=True, help="Path to tiles_manifest.json")
    p.add_argument("--train-manifest", type=Path, required=True, help="Path to train_manifest.json")
    p.add_argument("--train-cfg", type=Path, required=True, help="Path to training config YAML/JSON")

    p.add_argument(
        "--split-name",
        type=str,
        default="golden_test",
        help="Evaluation split name recorded in metrics/contract.",
    )
    p.add_argument(
        "--groups-file",
        type=Path,
        required=True,
        help="Newline-delimited text file listing the eval groups/regions/scenes.",
    )
    p.add_argument(
        "--group-col",
        type=str,
        default="region",
        help="Column in master CSV used to select eval rows.",
    )

    p.add_argument("--out-dir", type=Path, required=True, help="Output directory for evaluation artifacts")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=1337)

    return p


def _load_structured_file(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    suffix = p.suffix.lower()
    text = p.read_text(encoding="utf-8")

    if suffix == ".json":
        obj = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        obj = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported file extension '{suffix}' for {p}. Use .json/.yaml/.yml")

    if not isinstance(obj, dict):
        raise ValueError(f"Expected mapping/object at root of {p}, got {type(obj)}")

    return obj


def _build_upstream(
    *,
    tiles_manifest: Path,
    train_manifest: Path,
    train_cfg: Path,
) -> Dict[str, Any]:
    return {
        "tiles_manifest": str(tiles_manifest),
        "train_manifest": str(train_manifest),
        "train_cfg": str(train_cfg),
    }


def _print_summary(summary: Dict[str, Any]) -> None:
    print(
        f"[eval] task={summary['task']} | split={summary['split_name']} | "
        f"num_eval_tiles={summary['num_eval_tiles']}"
    )
    print(f"[eval] metrics={summary['metrics_path']}")
    print(f"[eval] model={summary['model_path']}")
    print(f"[eval] out_dir={summary['eval_dir']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    # -----------------------------
    # Load upstream artifacts/configs
    # -----------------------------
    tiles = load_tiles_contract(args.tiles_manifest)
    train_manifest = _load_structured_file(args.train_manifest)
    train_cfg = _load_structured_file(args.train_cfg)

    model_path = Path(train_manifest["model_path"])
    tiles_df = pd.read_csv(tiles.master_csv)
    groups = load_groups_file(args.groups_file)

    # -----------------------------
    # Run core evaluation engine
    # -----------------------------
    outputs = run_evaluation(
        tiles_df=tiles_df,
        train_cfg=train_cfg,
        model_path=model_path,
        group_col=args.group_col,
        groups=groups,
        split_name=args.split_name,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # -----------------------------
    # Write metrics artifact
    # -----------------------------
    metrics_path = args.out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(outputs.metrics, indent=2) + "\n", encoding="utf-8")

    # -----------------------------
    # Write canonical eval contract
    # -----------------------------
    contract = EvalContract(
        eval_dir=args.out_dir,
        schema_version=EVAL_SCHEMA_VERSION_V1,
        task=args.task,
        split_name=args.split_name,
        metrics_path=metrics_path,
        model_path=model_path,
        num_eval_tiles=int(outputs.num_eval_tiles),
        group_col=args.group_col,
        selection_source=args.groups_file,
        metrics=outputs.metrics,
        upstream=_build_upstream(
            tiles_manifest=args.tiles_manifest,
            train_manifest=args.train_manifest,
            train_cfg=args.train_cfg,
        ),
        meta={
            "num_groups": len(groups),
            "groups_preview": groups[:10],
            "seed": int(args.seed),
            "device": str(device),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
        },
    )

    contract_path = write_eval_contract(contract)
    summary = summarize_eval_contract(contract)

    _print_summary(summary)
    print(f"[eval] contract={contract_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())