# geo_mlops/core/datasets/scene_discovery.py

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from geo_mlops.core.data.types import DatasetLayout, DiscoveredScene
from geo_mlops.core.utils.resolve_paths import _relaxed_lookup, _tif_map


def discover_sub_roi_scenes(
    sub_roi_dir: Path,
    layout: DatasetLayout,
    stems_to_process_fn: Callable[[dict[str, Path], dict[str, Path]], list[str]] | None = None,
    require_gt_dir: bool = False,
    require_context_dir: bool = False,
    require_nonempty_gt_map: bool = False,
    require_nonempty_context_map: bool = False,
    allow_missing_context_per_scene: bool = True,
) -> tuple[list[DiscoveredScene], dict[str, int | bool | str]]:
    sub_roi_dir = Path(sub_roi_dir)

    roi = sub_roi_dir.parent.name
    sub_roi = sub_roi_dir.name

    stats: dict[str, int | bool | str] = {
        "roi": roi,
        "sub_roi": sub_roi,
        "sub_roi_pred_missing": False,
        "sub_roi_context_missing": False,
        "scenes_discovered": 0,
        "scenes_skipped_no_pan": 0,
        "scenes_skipped_no_context": 0,
    }

    pan_dir = sub_roi_dir / layout.pan_dirname
    if not pan_dir.is_dir():
        stats["scenes_skipped_no_pan"] = 1
        return [], stats

    gt_dir = sub_roi_dir / layout.gt_dirname if layout.gt_dirname else None
    ctx_dir = sub_roi_dir / layout.context_dirname if layout.context_dirname else None
    pred_dir = sub_roi_dir / layout.preds_dirname if layout.preds_dirname else None

    if require_gt_dir and (gt_dir is None or not gt_dir.is_dir()):
        return [], stats

    if require_context_dir and (ctx_dir is None or not ctx_dir.is_dir()):
        stats["sub_roi_context_missing"] = True
        return [], stats

    pan_map = _tif_map(pan_dir)
    gt_map = _tif_map(gt_dir) if gt_dir else {}
    ctx_map = _tif_map(ctx_dir) if ctx_dir else {}
    pred_map = _tif_map(pred_dir) if pred_dir else {}

    if require_nonempty_gt_map and not gt_map:
        return [], stats

    if require_nonempty_context_map and not ctx_map:
        stats["sub_roi_context_missing"] = True
        return [], stats

    if layout.preds_dirname is not None and (pred_dir is None or not pred_dir.is_dir()):
        stats["sub_roi_pred_missing"] = True

    stems = stems_to_process_fn(pan_map, gt_map) if stems_to_process_fn is not None else sorted(pan_map.keys())

    scenes: list[DiscoveredScene] = []

    for stem in stems:
        pan_path = _relaxed_lookup(stem, pan_map)
        if pan_path is None:
            stats["scenes_skipped_no_pan"] += 1
            continue

        gt_path = _relaxed_lookup(stem, gt_map) if gt_map else None
        ctx_path = _relaxed_lookup(stem, ctx_map) if ctx_map else None
        pred_path = _relaxed_lookup(stem, pred_map) if pred_map else None

        if ctx_path is None and not allow_missing_context_per_scene:
            stats["scenes_skipped_no_context"] += 1
            continue

        scene_id = f"{roi}__{sub_roi}__{pan_path.stem}"

        scenes.append(
            DiscoveredScene(
                region=roi,
                subregion=sub_roi,
                stem=stem,
                scene_id=scene_id,
                pan_path=pan_path,
                gt_path=gt_path,
                context_path=ctx_path,
                pred_path=pred_path,
                meta={
                    "roi": roi,
                    "sub_roi": sub_roi,
                    "stem": stem,
                },
            )
        )

        stats["scenes_discovered"] += 1

    return scenes, stats


def discover_roi_scenes(
    *,
    roi_dir: Path,
    layout: DatasetLayout,
    stems_to_process_fn: Callable[[dict[str, Path], dict[str, Path]], Sequence[str]] | None = None,
    require_gt_dir: bool = False,
    require_context_dir: bool = False,
    require_nonempty_gt_map: bool = False,
    require_nonempty_context_map: bool = False,
    allow_missing_context_per_scene: bool = True,
) -> tuple[list[DiscoveredScene], dict[str, Any]]:
    """
    Discover all scenes under one top-level ROI.

    Expected layout:

        roi_dir/
          000/
            PAN/
            Context/
            GT-Mask/
          001/
            PAN/
            Context/
            GT-Mask/
    """
    roi_dir = Path(roi_dir)

    if not roi_dir.exists():
        raise FileNotFoundError(f"ROI directory does not exist: {roi_dir}")

    if not roi_dir.is_dir():
        raise NotADirectoryError(f"ROI path is not a directory: {roi_dir}")

    roi_name = roi_dir.name
    sub_roi_dirs = [p for p in sorted(roi_dir.iterdir()) if p.is_dir()]

    all_scenes: list[DiscoveredScene] = []

    stats: dict[str, Any] = dict(
        roi=roi_name,
        roi_total=1,
        sub_roi_total=0,
        sub_roi_pred_missing_count=0,
        sub_roi_context_missing_count=0,
        scenes_discovered=0,
        scenes_skipped_no_pan=0,
        scenes_skipped_no_context=0,
    )

    for sub_roi_dir in sub_roi_dirs:
        scenes, sub_stats = discover_sub_roi_scenes(
            sub_roi_dir=sub_roi_dir,
            layout=layout,
            stems_to_process_fn=stems_to_process_fn,
            require_gt_dir=require_gt_dir,
            require_context_dir=require_context_dir,
            require_nonempty_gt_map=require_nonempty_gt_map,
            require_nonempty_context_map=require_nonempty_context_map,
            allow_missing_context_per_scene=allow_missing_context_per_scene,
        )

        all_scenes.extend(scenes)

        stats["sub_roi_total"] += 1
        stats["scenes_discovered"] += int(sub_stats.get("scenes_discovered", 0))
        stats["scenes_skipped_no_pan"] += int(sub_stats.get("scenes_skipped_no_pan", 0))
        stats["scenes_skipped_no_context"] += int(sub_stats.get("scenes_skipped_no_context", 0))

        if sub_stats.get("sub_roi_pred_missing"):
            stats["sub_roi_pred_missing_count"] += 1

        if sub_stats.get("sub_roi_context_missing"):
            stats["sub_roi_context_missing_count"] += 1

    return all_scenes, stats


def discover_dataset_scenes(
    *,
    dataset_root: Path,
    layout: DatasetLayout,
    roi_names: Sequence[str] | None = None,
    stems_to_process_fn: Callable[[dict[str, Path], dict[str, Path]], Sequence[str]] | None = None,
    require_gt_dir: bool = False,
    require_context_dir: bool = False,
    require_nonempty_gt_map: bool = False,
    require_nonempty_context_map: bool = False,
    allow_missing_context_per_scene: bool = True,
) -> tuple[list[DiscoveredScene], list[str], dict[str, Any]]:
    """
    Discover all scenes under a full dataset root.

    Expected layout:

        dataset_root/
          AOI_Paris/
            000/
              PAN/
              Context/
              GT-Mask/
          AOI_Khartoum/
            000/
              PAN/
              Context/
              GT-Mask/

    Convention:
        roi     = AOI_Paris / AOI_Khartoum
        sub_roi = 000 / 001 / ...
    """
    dataset_root = Path(dataset_root)

    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {dataset_root}")

    if not dataset_root.is_dir():
        raise NotADirectoryError(f"dataset_root is not a directory: {dataset_root}")

    if roi_names is None:
        roi_dirs = [p for p in sorted(dataset_root.iterdir()) if p.is_dir()]
    else:
        roi_dirs = [dataset_root / str(name) for name in roi_names]

    missing = [p for p in roi_dirs if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing ROI directory/directories: " + ", ".join(str(p) for p in missing))

    roi_dirs = [p for p in roi_dirs if p.is_dir()]
    resolved_roi_names = [p.name for p in roi_dirs]

    all_scenes: list[DiscoveredScene] = []

    stats: dict[str, Any] = dict(
        roi_total=0,
        sub_roi_total=0,
        scenes_discovered=0,
        scenes_skipped_no_pan=0,
        scenes_skipped_no_context=0,
        sub_roi_pred_missing_count=0,
        sub_roi_context_missing_count=0,
    )

    for roi_dir in roi_dirs:
        scenes, roi_stats = discover_roi_scenes(
            roi_dir=roi_dir,
            layout=layout,
            stems_to_process_fn=stems_to_process_fn,
            require_gt_dir=require_gt_dir,
            require_context_dir=require_context_dir,
            require_nonempty_gt_map=require_nonempty_gt_map,
            require_nonempty_context_map=require_nonempty_context_map,
            allow_missing_context_per_scene=allow_missing_context_per_scene,
        )

        all_scenes.extend(scenes)

        stats["roi_total"] += 1
        stats["sub_roi_total"] += int(roi_stats.get("sub_roi_total", 0))
        stats["scenes_discovered"] += int(roi_stats.get("scenes_discovered", 0))
        stats["scenes_skipped_no_pan"] += int(roi_stats.get("scenes_skipped_no_pan", 0))
        stats["scenes_skipped_no_context"] += int(roi_stats.get("scenes_skipped_no_context", 0))
        stats["sub_roi_pred_missing_count"] += int(roi_stats.get("sub_roi_pred_missing_count", 0))
        stats["sub_roi_context_missing_count"] += int(roi_stats.get("sub_roi_context_missing_count", 0))

    return all_scenes, resolved_roi_names, stats
