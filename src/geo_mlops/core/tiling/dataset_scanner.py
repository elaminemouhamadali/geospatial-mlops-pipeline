from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from geo_mlops.core.data.types import DatasetLayout
from geo_mlops.core.data.scene_discovery import discover_sub_roi_scenes
from geo_mlops.core.tiling.engine import RoiTilingEngine
from geo_mlops.core.utils.dataframes import (
    _merge_stats,
    _sort_master,
    write_subdir_csv,
)


def _layout_from_engine(engine: RoiTilingEngine) -> DatasetLayout:
    return DatasetLayout(
        pan_dirname=engine.cfg.pan_dirname,
        gt_dirname=engine.cfg.gt_dirname,
        context_dirname=engine.cfg.context_dirname,
        preds_dirname=engine.cfg.preds_dirname,
    )


def _ensure_roi_columns(
    df: pd.DataFrame,
    *,
    roi: str,
    sub_roi: str | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    if "roi" not in df.columns:
        df["roi"] = roi
    else:
        df["roi"] = df["roi"].fillna(roi)

    if sub_roi is not None:
        if "sub_roi" not in df.columns:
            df["sub_roi"] = sub_roi
        else:
            df["sub_roi"] = df["sub_roi"].fillna(sub_roi)

    return df


def scan_sub_roi(
    sub_roi_dir: Path,
    engine: RoiTilingEngine,
    csv_name: str,
    *,
    force: bool = False,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Scan one sub-ROI directory.

    Expected layout:

        sub_roi_dir/
          PAN/
          Context/
          GT-Mask/
          Preds/ optional

    Responsibilities:
      - handle per-sub-ROI CSV cache
      - discover scenes
      - pass discovered scenes to engine.tile_scenes(...)
      - write per-sub-ROI CSV
      - return a dataframe + combined discovery/tiling stats
    """
    sub_roi_dir = Path(sub_roi_dir)

    if not sub_roi_dir.exists():
        raise FileNotFoundError(f"sub_roi_dir does not exist: {sub_roi_dir}")

    if not sub_roi_dir.is_dir():
        raise NotADirectoryError(f"sub_roi_dir is not a directory: {sub_roi_dir}")

    roi_name = sub_roi_dir.parent.name
    sub_roi_name = sub_roi_dir.name

    stats: Dict[str, Any] = dict(
        roi=roi_name,
        sub_roi=sub_roi_name,
        skipped_existing_csv=0,
        sub_roi_pred_missing=False,
        sub_roi_context_missing=False,
        scenes_discovered=0,
        scenes_processed=0,
        scenes_skipped_no_pan=0,
        scenes_skipped_no_context=0,
        scenes_read_error=0,
        tiles_considered=0,
        tiles_included=0,
        tiles_skipped=0,
        tiles_skipped_nodata=0,
    )

    existing_csv = sub_roi_dir / csv_name

    if existing_csv.exists() and not force:
        df_existing = pd.read_csv(existing_csv)
        df_existing = _ensure_roi_columns(
            df_existing,
            roi=roi_name,
            sub_roi=sub_roi_name,
        )

        stats["skipped_existing_csv"] = 1

        if verbose:
            tqdm.write(
                f"[SKIP] {roi_name}/{sub_roi_name}: "
                f"found existing {csv_name} ({len(df_existing)} rows)"
            )

        return _sort_master(df_existing), stats

    layout = _layout_from_engine(engine)

    discovered_scenes, discovery_stats = discover_sub_roi_scenes(
        sub_roi_dir=sub_roi_dir,
        layout=layout,
        stems_to_process_fn=lambda pan_map, gt_map: engine.adapter.stems_to_process(
            pan_map=pan_map,
            gt_map=gt_map,
        ),
        require_gt_dir=engine.adapter.require_gt_dir(),
        require_context_dir=engine.adapter.require_context_dir(),
        require_nonempty_gt_map=engine.adapter.require_nonempty_gt_map(),
        require_nonempty_context_map=engine.adapter.require_nonempty_context_map(),
        allow_missing_context_per_scene=engine.adapter.allow_missing_context_per_scene(),
    )

    _merge_stats(stats, discovery_stats)

    rows, tile_stats = engine.tile_scenes(
        discovered_scenes,
        sub_roi_pred_missing=bool(discovery_stats.get("sub_roi_pred_missing", False)),
    )

    _merge_stats(stats, tile_stats)

    for row in rows:
        row.setdefault("roi", roi_name)
        row.setdefault("sub_roi", sub_roi_name)

    csv_path = write_subdir_csv(
        sub_roi_dir,
        rows,
        csv_name,
    )

    if verbose:
        tqdm.write(
            f"[SUB-ROI] {roi_name}/{sub_roi_name}: "
            f"scenes={stats.get('scenes_discovered', 0)} "
            f"rows={len(rows)} "
            f"considered={stats.get('tiles_considered', 0)} "
            f"included={stats.get('tiles_included', 0)} "
            f"csv={csv_path if csv_path is not None else 'not written'}"
        )

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    df = _ensure_roi_columns(
        df,
        roi=roi_name,
        sub_roi=sub_roi_name,
    )

    return _sort_master(df), stats


def scan_roi(
    roi_dir: Path,
    engine: RoiTilingEngine,
    csv_name: str,
    *,
    force: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Scan one top-level ROI directory.

    Expected layout:

        ROI/
          000/
            PAN/
            Context/
            GT-Mask/
          001/
            PAN/
            Context/
            GT-Mask/

    Convention:
        roi     = top-level folder, e.g. AOI_Paris
        sub_roi = child folder, e.g. 000
    """
    roi_dir = Path(roi_dir)

    if not roi_dir.exists():
        raise FileNotFoundError(f"ROI directory does not exist: {roi_dir}")

    if not roi_dir.is_dir():
        raise NotADirectoryError(f"ROI path is not a directory: {roi_dir}")

    roi_name = roi_dir.name
    sub_roi_dirs = [p for p in sorted(roi_dir.iterdir()) if p.is_dir()]

    all_rows: List[Dict[str, Any]] = []

    roi_stats: Dict[str, Any] = defaultdict(int)
    roi_stats["roi"] = roi_name

    for sub_roi_dir in tqdm(sub_roi_dirs, desc=f"Sub-ROIs ({roi_name})"):
        df_sub_roi, sub_stats = scan_sub_roi(
            sub_roi_dir,
            engine=engine,
            csv_name=csv_name,
            force=force,
            verbose=verbose,
        )

        if not df_sub_roi.empty:
            all_rows.extend(df_sub_roi.to_dict(orient="records"))

        roi_stats["sub_roi_total"] += 1

        if sub_stats.get("skipped_existing_csv"):
            roi_stats["skipped_existing_csv"] += 1

        if sub_stats.get("sub_roi_pred_missing"):
            roi_stats["sub_roi_pred_missing_count"] += 1

        if sub_stats.get("sub_roi_context_missing"):
            roi_stats["sub_roi_context_missing_count"] += 1

        for k, v in sub_stats.items():
            if isinstance(v, int) and k not in {
                "roi",
                "sub_roi",
                "skipped_existing_csv",
                "sub_roi_pred_missing",
                "sub_roi_context_missing",
            }:
                roi_stats[k] += v

    if verbose:
        tqdm.write(
            f"[ROI] {roi_name} | "
            f"sub_rois={roi_stats.get('sub_roi_total', 0)} "
            f"skipped_existing_csv={roi_stats.get('skipped_existing_csv', 0)} | "
            f"scenes_discovered={roi_stats.get('scenes_discovered', 0)} "
            f"scenes_ok={roi_stats.get('scenes_processed', 0)} "
            f"no_pan={roi_stats.get('scenes_skipped_no_pan', 0)} "
            f"no_context={roi_stats.get('scenes_skipped_no_context', 0)} "
            f"io_err={roi_stats.get('scenes_read_error', 0)} | "
            f"tiles_considered={roi_stats.get('tiles_considered', 0)} "
            f"tiles_included={roi_stats.get('tiles_included', 0)} "
            f"tiles_skipped={roi_stats.get('tiles_skipped', 0)} "
            f"tiles_skipped_nodata={roi_stats.get('tiles_skipped_nodata', 0)}"
        )

    df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    df = _ensure_roi_columns(
        df,
        roi=roi_name,
    )

    return _sort_master(df)


def scan_full_dataset(
    dataset_root_path: Path,
    engine: RoiTilingEngine,
    csv_name: str,
    *,
    force: bool,
    verbose: bool,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Scan all top-level ROIs under dataset_root_path.

    Expected layout:

        dataset_root_path/
          AOI_Paris/
            000/
            001/
          AOI_Khartoum/
            000/
            001/

    Returns:
        master_df, roi_names
    """
    dataset_root_path = Path(dataset_root_path)

    if not dataset_root_path.exists():
        raise FileNotFoundError(f"dataset_root_path does not exist: {dataset_root_path}")

    if not dataset_root_path.is_dir():
        raise NotADirectoryError(f"dataset_root_path is not a directory: {dataset_root_path}")

    roi_dirs = [p for p in sorted(dataset_root_path.iterdir()) if p.is_dir()]
    roi_names = [roi_dir.name for roi_dir in roi_dirs]

    if not roi_dirs:
        raise ValueError(f"No ROI directories found under: {dataset_root_path}")

    dfs: List[pd.DataFrame] = []

    for roi_dir in roi_dirs:
        df_roi = scan_roi(
            roi_dir,
            engine=engine,
            csv_name=csv_name,
            force=force,
            verbose=verbose,
        )

        if not df_roi.empty:
            df_roi = _ensure_roi_columns(
                df_roi,
                roi=roi_dir.name,
            )

        dfs.append(df_roi)

    non_empty = [df for df in dfs if not df.empty]

    if not non_empty:
        return pd.DataFrame(), roi_names

    df = pd.concat(non_empty, ignore_index=True)
    return _sort_master(df), roi_names