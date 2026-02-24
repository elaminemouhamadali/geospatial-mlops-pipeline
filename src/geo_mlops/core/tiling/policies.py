from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from geo_mlops.core.tiling.adapters.base import (
    SceneArrays,
    SceneInputs,
    TaskAdapter,
    TileWindow,
)


# -------------------------
# AllPolicy
# -------------------------
@dataclass
class AllPolicy:
    """
    Include every tile unconditionally (except engine-level nodata skips).
    """

    def extra_row_fields(self) -> Dict[str, Any]:
        return {}

    def decide_include(
        self,
        *,
        adapter: TaskAdapter,
        scene: SceneInputs,
        arr: SceneArrays,
        tw: TileWindow,
        roi_pred_missing: bool,
    ) -> Tuple[bool, Dict[str, Any]]:
        _ = (adapter, scene, arr, tw, roi_pred_missing)
        return True, {}


# -------------------------
# RegularPolicy (GT presence)
# -------------------------
@dataclass
class RegularPolicy:
    """
    GT-based inclusion policy.
    Uses adapter.gt_presence(...) which returns PresenceResult(value, details).
    Include if presence.value >= gt_presence_threshold.
    For instance, if the water segmentation adapter defines the presence of water to be True
    if water fraction in chip is > 10% --> only chips with > 10% water pixels will be retained in the generated csv
    """

    gt_presence_threshold: float = 1e-6
    require_presence: bool = True

    # Prefix keys to avoid collisions with task columns
    details_prefix: str = "presence__"

    def extra_row_fields(self) -> Dict[str, Any]:
        key = f"{self.details_prefix}value"
        return {key: 0.0}

    def decide_include(
        self,
        *,
        adapter: TaskAdapter,
        scene: SceneInputs,
        arr: SceneArrays,
        tw: TileWindow,
        roi_pred_missing: bool,
    ) -> Tuple[bool, Dict[str, Any]]:
        _ = roi_pred_missing  # regular policy doesn't care about ROI-level pred state

        if not self.require_presence:
            return True, {}

        pres = adapter.gt_presence(scene=scene, arr=arr, tw=tw)
        include = float(pres.value) >= float(self.gt_presence_threshold)

        extra: Dict[str, Any] = {}
        # Bubble details into CSV
        if pres.details:
            if self.details_prefix:
                extra.update({f"{self.details_prefix}{k}": v for k, v in pres.details.items()})
            else:
                extra.update(pres.details)

        # Useful to always emit presence value
        extra.setdefault(f"{self.details_prefix}value", float(pres.value))

        return include, extra


# -------------------------
# HardMiningPolicy (Pred vs GT difficulty)
# -------------------------
@dataclass
class HardMiningPolicy:
    """
    Include tiles based on disagreement/difficulty between pred and gt.
    Strict mode:
      - No fallback behavior
      - If predictions are missing (ROI dir missing OR per-scene pred missing), raise immediately
    For instance, if the water segmentation adapter defines the difficulty of water to be True
    if Prediction is > 10% different from Ground Truth
    --> only chips with > 10% difference will be retained in the generated csv
    This policy is configured to always include chips containing class-of-interest
    irrespective of its difficulty, so if water if present in a chip -> included in csv
    """

    # thresholds on DifficultyResult.value (higher => harder)
    min_difficulty: float = 1e-6

    # include positives even if not hard
    include_if_gt_present: bool = True
    gt_presence_threshold: float = 1e-6

    presence_prefix: str = "presence__"
    difficulty_prefix: str = "difficulty__"

    def extra_row_fields(self) -> Dict[str, Any]:
        pres_key = f"{self.presence_prefix}value"
        diff_key = f"{self.difficulty_prefix}value"
        return {pres_key: 0.0, diff_key: 0.0}

    def decide_include(
        self,
        *,
        adapter: TaskAdapter,
        scene: SceneInputs,
        arr: SceneArrays,
        tw: TileWindow,
        roi_pred_missing: bool,
    ) -> Tuple[bool, Dict[str, Any]]:
        # -------------------------
        # Strict prediction requirement
        # -------------------------
        if roi_pred_missing:
            # ROI-level: preds directory missing (or not provided)
            raise FileNotFoundError(
                "HardMiningPolicy requires predictions, but preds directory is missing for ROI: "
                f"{scene.region}/{scene.subregion}"
            )

        # Per-scene: preds expected but not loaded for this scene
        if arr.pred2d is None:
            raise ValueError(
                "HardMiningPolicy requires predictions, but pred2d is None for scene/tile: "
                f"{scene.region}/{scene.subregion}"
            )

        extra: Dict[str, Any] = {}

        # -------------------------
        # Optionally include positives always
        # -------------------------
        if self.include_if_gt_present:
            pres = adapter.gt_presence(scene=scene, arr=arr, tw=tw)
            extra.update(self._pack_details(pres.value, pres.details, prefix=self.presence_prefix))

            if float(pres.value) >= float(self.gt_presence_threshold):
                # still compute difficulty for diagnostics
                diff = adapter.difficulty(scene=scene, arr=arr, tw=tw)
                extra.update(self._pack_details(diff.value, diff.details, prefix=self.difficulty_prefix))
                return True, extra

        # -------------------------
        # Otherwise include if difficulty exceeds threshold
        # -------------------------
        diff = adapter.difficulty(scene=scene, arr=arr, tw=tw)
        extra.update(self._pack_details(diff.value, diff.details, prefix=self.difficulty_prefix))
        include = float(diff.value) >= float(self.min_difficulty)
        return include, extra

    @staticmethod
    def _pack_details(value: float, details: Dict[str, Any] | None, *, prefix: str) -> Dict[str, Any]:
        key = f"{prefix}value" if prefix else "value"
        out: Dict[str, Any] = {key: float(value)}
        if details:
            if prefix:
                out.update({f"{prefix}{k}": v for k, v in details.items()})
            else:
                out.update(details)
        return out
