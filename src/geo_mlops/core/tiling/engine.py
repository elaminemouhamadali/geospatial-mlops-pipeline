from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio

from geo_mlops.core.tiling.adapters.base import (
    SceneArrays,
    SceneInputs,
    TaskAdapter,
    TileWindow,
    TilingPolicy,
)
from geo_mlops.core.data.types import DiscoveredScene
from geo_mlops.core.utils.geospatial import gsd_from_epsg4326
from geo_mlops.core.utils.scanning import gen_tiles_cover
from geo_mlops.core.utils.dataframes import _merge_stats


# -----------------------------------------
# Engine config (task-agnostic)
# -----------------------------------------
@dataclass(frozen=True)
class TilingEngineConfig:
    # optional discovery inputs
    preds_dirname: Optional[str]
    context_dirname: Optional[str]
    context_max_side_cap: Optional[int]

    # discovery
    pan_dirname: str
    gt_dirname: str

    # tiling geometry
    target_size_m: float
    overlap: float
    reflectance_max: int

    # shared behavior toggles
    verbose: bool
    skip_tiles_with_nodata: bool


# -----------------------------------------
# Unified scene tiling engine
# -----------------------------------------
class RoiTilingEngine:
    """
    Engine responsibilities ONLY:
      - tile already-discovered scenes
      - load PAN / GT / pred arrays
      - compute scene metadata
      - generate tile windows
      - apply nodata skip
      - apply policy
      - emit tile CSV row dictionaries
      - delegate task-specific columns to adapter

    Not responsible for:
      - dataset-root traversal
      - ROI / sub-ROI traversal
      - PAN / GT / Context / Pred discovery
      - CSV caching or writing
    """

    def __init__(
        self,
        *,
        cfg: TilingEngineConfig,
        adapter: TaskAdapter,
        policy: TilingPolicy,
    ):
        self.cfg = cfg
        self.adapter = adapter
        self.policy = policy

    # -----------------------------
    # Public tiling API
    # -----------------------------
    def tile_scenes(
        self,
        scenes: List[DiscoveredScene],
        *,
        sub_roi_pred_missing: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Tile a list of already-discovered scenes.

        Discovery belongs to dataset_scanner.py / scene_discovery.py.
        This method only performs scene -> tile rows.
        """
        stats: Dict[str, Any] = dict(
            scenes_processed=0,
            scenes_skipped_no_pan=0,
            scenes_skipped_no_context=0,
            scenes_read_error=0,
            tiles_considered=0,
            tiles_included=0,
            tiles_skipped=0,
            tiles_skipped_nodata=0,
        )

        rows: List[Dict[str, Any]] = []

        for scene in scenes:
            scene_rows, scene_stats = self.tile_scene(
                scene,
                sub_roi_pred_missing=sub_roi_pred_missing,
            )

            rows.extend(scene_rows)
            _merge_stats(stats, scene_stats)

        return rows, stats

    def tile_scene(
        self,
        scene: DiscoveredScene,
        *,
        sub_roi_pred_missing: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Tile one already-discovered scene.

        The scene already contains:
          - pan_path
          - gt_path
          - pred_path
          - context_path
          - roi
          - sub_roi
          - stem
          - scene_id
        """
        stats: Dict[str, Any] = dict(
            scenes_processed=0,
            scenes_skipped_no_pan=0,
            scenes_skipped_no_context=0,
            scenes_read_error=0,
            tiles_considered=0,
            tiles_included=0,
            tiles_skipped=0,
            tiles_skipped_nodata=0,
        )

        # -----------------------------
        # Read PAN + compute GSD
        # -----------------------------
        try:
            with rasterio.open(scene.pan_path) as src:
                pan_2d = src.read(1)

            H, W = pan_2d.shape
            gsd_mpp = gsd_from_epsg4326(scene.pan_path)

        except (FileNotFoundError, OSError, rasterio.errors.RasterioIOError):
            stats["scenes_read_error"] += 1
            return [], stats

        # -----------------------------
        # Read GT or fabricate zeros
        # -----------------------------
        if scene.gt_path is not None:
            try:
                with rasterio.open(scene.gt_path) as src:
                    gt2d = src.read()
                    gt2d = gt2d.astype(np.int64)
            except (FileNotFoundError, OSError, rasterio.errors.RasterioIOError):
                stats["scenes_read_error"] += 1
                return [], stats
        else:
            if self.adapter.allow_fabricated_zero_gt():
                gt2d = np.zeros((H, W), dtype=np.int64)
            else:
                gt2d = None

        # -----------------------------
        # Read pred optional
        # -----------------------------
        if scene.pred_path is not None:
            try:
                with rasterio.open(scene.pred_path) as src:
                    pred2d = src.read()
                    pred2d = pred2d.astype(np.int64)
            except (FileNotFoundError, OSError, rasterio.errors.RasterioIOError):
                stats["scenes_read_error"] += 1
                return [], stats
        else:
            pred2d = None

        scene_inputs = SceneInputs(
            roi=scene.roi,
            sub_roi=scene.sub_roi,
            stem=scene.stem,
            pan_path=scene.pan_path,
            gt_path=scene.gt_path,
            pred_path=scene.pred_path,
            context_path=scene.context_path,
            scene_id=scene.scene_id,
        )

        arr = SceneArrays(
            H=int(H),
            W=int(W),
            gsd_mpp=float(gsd_mpp),
            gt2d=gt2d,
            pred2d=pred2d,
        )

        # Adapter may populate derived layers / masks.
        self.adapter.build_convenience_layers(arr)

        # -----------------------------
        # Tiling params
        # -----------------------------
        tile_px = max(
            8,
            int(round(self.cfg.target_size_m / max(1e-6, arr.gsd_mpp))),
        )
        stride = max(
            1,
            int(round(tile_px * (1.0 - self.cfg.overlap))),
        )

        th = tw = tile_px
        sh = sw = stride

        rows: List[Dict[str, Any]] = []
        stats["scenes_processed"] += 1

        tile_idx = 0

        # -----------------------------
        # Tile loop
        # -----------------------------
        for x0, y0, x1, y1, r, c in gen_tiles_cover(
            int(H),
            int(W),
            th,
            tw,
            sh,
            sw,
        ):
            tot = int((y1 - y0) * (x1 - x0))
            stats["tiles_considered"] += 1

            win = TileWindow(
                x0=int(x0),
                y0=int(y0),
                x1=int(x1),
                y1=int(y1),
                r=int(r),
                c=int(c),
                tile_idx=int(tile_idx),
                tot=int(tot),
            )

            # Shared nodata skip.
            if self.cfg.skip_tiles_with_nodata and arr.pan_mask is not None:
                if arr.pan_mask[win.y0 : win.y1, win.x0 : win.x1].any():
                    stats["tiles_skipped_nodata"] += 1
                    stats["tiles_skipped"] += 1
                    tile_idx += 1
                    continue

            include, extra = self.policy.decide_include(
                adapter=self.adapter,
                scene=scene_inputs,
                arr=arr,
                tw=win,
                sub_roi_pred_missing=sub_roi_pred_missing,
            )

            if not include:
                stats["tiles_skipped"] += 1
                tile_idx += 1
                continue

            # -----------------------------
            # Core row fields
            # -----------------------------
            cx = 0.5 * (win.x0 + win.x1)
            cy = 0.5 * (win.y0 + win.y1)

            row: Dict[str, Any] = dict(
                roi=scene.roi,
                sub_roi=scene.sub_roi,
                scene_id=scene.scene_id,
                stem=scene.stem,
                image_src=str(scene.pan_path),
                gt_src=str(scene.gt_path) if scene.gt_path else "",
                pred_src=str(scene.pred_path) if scene.pred_path else "",
                context_src=str(scene.context_path) if scene.context_path else "",
                x0=win.x0,
                y0=win.y0,
                x1=win.x1,
                y1=win.y1,
                tile_w_px=int(win.x1 - win.x0),
                tile_h_px=int(win.y1 - win.y0),
                gsd_mpp=float(arr.gsd_mpp),
                scene_h=int(arr.H),
                scene_w=int(arr.W),
                tile_row=int(win.r),
                tile_col=int(win.c),
                tile_idx=int(win.tile_idx),
                tile_cx_norm=float(cx / max(1, arr.W)),
                tile_cy_norm=float(cy / max(1, arr.H)),
                tile_size_px=int(tile_px),
                stride_px=int(stride),
                overlap=float(self.cfg.overlap),
            )

            # Policy schema defaults + policy extra + task columns.
            row.update(self.policy.extra_row_fields())
            row.update(extra)
            row.update(
                self.adapter.build_task_row(
                    scene=scene_inputs,
                    arr=arr,
                    tw=win,
                )
            )

            rows.append(row)
            stats["tiles_included"] += 1
            tile_idx += 1

        return rows, stats
