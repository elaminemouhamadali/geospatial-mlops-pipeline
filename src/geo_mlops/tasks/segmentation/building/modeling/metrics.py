from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
import torch


# -----------------------------------------------------------------------------
# Training-time validation metrics
# -----------------------------------------------------------------------------
def build_metrics_fn(train_cfg: Dict[str, Any]):
    metrics_cfg = train_cfg.get("metrics", {}) or {}

    threshold = float(metrics_cfg.get("threshold", 0.5))
    foreground_label = int(metrics_cfg.get("foreground_label", 1))
    eps = float(metrics_cfg.get("eps", 1e-7))

    def metrics_fn(outputs: torch.Tensor, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Lightweight binary segmentation metrics for training-time validation.

        This is intentionally batch-level and cheap. Formal golden evaluation
        uses BuildingSegmentationEvalAccumulator below.
        """
        if "mask" not in batch:
            raise KeyError("Building metrics expects batch['mask'].")

        if outputs.ndim != 4 or outputs.shape[1] != 1:
            raise ValueError(
                f"Building metrics expect outputs shaped [B,1,H,W], got {tuple(outputs.shape)}."
            )

        mask = batch["mask"].to(outputs.device)
        target = mask == foreground_label

        probs = torch.sigmoid(outputs[:, 0])
        pred = probs >= threshold

        counts = _torch_binary_counts(pred=pred, target=target)
        metrics = _metrics_from_counts(counts, eps=eps)

        return {
            "iou": metrics["iou"],
            "micro_precision": metrics["precision"],
            "micro_recall": metrics["recall"],
            "micro_f1": metrics["f1"],
            "pixel_accuracy": metrics["pixel_accuracy"],
        }

    return metrics_fn


class BuildingSegmentationEvalAccumulator:
    """
    Formal evaluation accumulator for full-scene building segmentation.

    Responsibilities:
      - consume already-loaded target/probability/mask arrays
      - compute per-scene metrics
      - aggregate global pixel-count micro metrics
      - aggregate per-scene macro metrics
      - aggregate per-ROI metrics
      - write Pareto/hardest-scene analytics tables

    File loading belongs outside this class.
    """

    def __init__(self, metrics_cfg: Optional[Dict[str, Any]] = None) -> None:
        metrics_cfg = metrics_cfg or {}

        self.eps = float(metrics_cfg.get("eps", 1e-7))
        self.pareto_top_k = int(metrics_cfg.get("pareto_top_k", 50))

        self.global_counts: Dict[str, int] = _empty_counts()
        self.rows: List[Dict[str, Any]] = []
        self.warnings: List[str] = []

    def update_from_arrays(
        self,
        *,
        scene_id: str,
        roi: str,
        sub_roi: str,
        target: np.ndarray,
        probability: np.ndarray,
        mask: np.ndarray,
        metadata: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        metadata = metadata or {}

        target_bool = _to_numpy_bool(target)
        pred_bool = _to_numpy_bool(mask)

        prob = np.asarray(probability).astype(np.float32)
        if prob.ndim == 3 and prob.shape[0] == 1:
            prob = prob[0]

        if pred_bool.shape != target_bool.shape:
            raise ValueError(
                f"Prediction/target shape mismatch for scene={scene_id!r}: "
                f"pred={pred_bool.shape}, target={target_bool.shape}"
            )

        if prob.shape != target_bool.shape:
            raise ValueError(
                f"Probability/target shape mismatch for scene={scene_id!r}: "
                f"prob={prob.shape}, target={target_bool.shape}"
            )

        counts = _numpy_binary_counts(pred=pred_bool, target=target_bool)
        self.global_counts = _add_counts(self.global_counts, counts)

        metrics = _metrics_from_counts(counts, eps=self.eps)
        prob_stats = _probability_stats(prob)

        row: Dict[str, Any] = {
            "scene_id": scene_id,
            "roi": roi,
            "sub_roi": sub_roi,
            "has_target": True,
            **counts,
            **metrics,
            "gt_foreground_pixels": int(target_bool.sum()),
            "pred_foreground_pixels": int(pred_bool.sum()),
            "gt_foreground_frac": float(target_bool.mean()),
            "pred_foreground_frac": float(pred_bool.mean()),
            "false_positive_pixels": int(counts["fp"]),
            "false_negative_pixels": int(counts["fn"]),
            "pan_path": str(metadata.get("pan_path", "")),
            "gt_path": str(metadata.get("gt_path", "")),
            "probability_path": str(metadata.get("probability_path", "")),
            "logits_path": str(metadata.get("logits_path", "")),
            **prob_stats,
        }

        self.rows.append(row)
        return row
    
    def ingest_rows(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            counts = {
                "tp": int(row.get("tp", 0)),
                "fp": int(row.get("fp", 0)),
                "fn": int(row.get("fn", 0)),
                "tn": int(row.get("tn", 0)),
            }
            self.global_counts = _add_counts(self.global_counts, counts)
            self.rows.append(dict(row))

    def finalize(self, *, out_dir: Path) -> Dict[str, Any]:
        out_dir = Path(out_dir)
        tables_dir = out_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self.rows)

        per_image_csv = tables_dir / "building_per_image_metrics.csv"
        df.to_csv(per_image_csv, index=False)

        roi_df = self._roi_metrics(df)
        roi_csv = tables_dir / "building_per_roi_metrics.csv"
        roi_df.to_csv(roi_csv, index=False)

        pareto_df = self._build_pareto_table(df)
        pareto_csv = tables_dir / "building_pareto_images.csv"
        pareto_df.to_csv(pareto_csv, index=False)

        micro = _metrics_from_counts(self.global_counts, eps=self.eps)
        scene_macro = self._macro_metrics(df)
        roi_macro = self._macro_metrics(roi_df)

        metrics: Dict[str, float] = {
            **{f"micro/{name}": float(value) for name, value in micro.items()},
            **{f"scene_macro/{name}": float(value) for name, value in scene_macro.items()},
            **{f"roi_macro/{name}": float(value) for name, value in roi_macro.items()},
        }

        analytics = {
            "num_images_with_target": int(df["has_target"].sum()) if "has_target" in df.columns else 0,
            "num_images_total": int(len(df)),
            "global_counts": dict(self.global_counts),
            "warnings": list(self.warnings),
        }

        return {
            "metrics": metrics,
            "artifacts": {
                "building_per_image_metrics_csv": str(per_image_csv),
                "building_per_roi_metrics_csv": str(roi_csv),
                "building_pareto_images_csv": str(pareto_csv),
            },
            "analytics": analytics,
        }

    def _macro_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        if df.empty:
            return {}

        if "has_target" in df.columns:
            df = df[df["has_target"] == True].copy()  # noqa: E712

        metric_cols = [
            "precision",
            "recall",
            "f1",
            "iou",
            "pixel_accuracy",
        ]

        out: Dict[str, float] = {}

        for col in metric_cols:
            if col in df.columns:
                values = pd.to_numeric(df[col], errors="coerce").dropna()
                if not values.empty:
                    out[col] = float(values.mean())

        return out

    def _roi_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        if "has_target" in df.columns:
            df = df[df["has_target"] == True].copy()  # noqa: E712

        if df.empty or "roi" not in df.columns:
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []

        for roi, g in df.groupby("roi"):
            counts = {
                "tp": int(g["tp"].sum()),
                "fp": int(g["fp"].sum()),
                "fn": int(g["fn"].sum()),
                "tn": int(g["tn"].sum()),
            }

            metrics = _metrics_from_counts(counts, eps=self.eps)

            rows.append(
                {
                    "roi": roi,
                    "num_scenes": int(len(g)),
                    **counts,
                    **metrics,
                    "gt_foreground_pixels": int(g["gt_foreground_pixels"].sum()),
                    "pred_foreground_pixels": int(g["pred_foreground_pixels"].sum()),
                    "false_positive_pixels": int(g["false_positive_pixels"].sum()),
                    "false_negative_pixels": int(g["false_negative_pixels"].sum()),
                }
            )

        return pd.DataFrame(rows)

    def _build_pareto_table(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        df = df.copy()

        if "has_target" in df.columns:
            df = df[df["has_target"] == True].copy()  # noqa: E712

        if df.empty:
            return df

        if "f1" in df.columns:
            df["rank_low_f1"] = pd.to_numeric(
                df["f1"], errors="coerce"
            ).rank(method="min", ascending=True)

        if "iou" in df.columns:
            df["rank_low_iou"] = pd.to_numeric(
                df["iou"], errors="coerce"
            ).rank(method="min", ascending=True)

        if "false_positive_pixels" in df.columns:
            df["rank_high_fp"] = pd.to_numeric(
                df["false_positive_pixels"], errors="coerce"
            ).rank(method="min", ascending=False)

        if "false_negative_pixels" in df.columns:
            df["rank_high_fn"] = pd.to_numeric(
                df["false_negative_pixels"], errors="coerce"
            ).rank(method="min", ascending=False)

        rank_cols = [c for c in df.columns if c.startswith("rank_")]

        if rank_cols:
            df["pareto_score"] = df[rank_cols].mean(axis=1)
            df.sort_values("pareto_score", inplace=True, ignore_index=True)

        return df.head(self.pareto_top_k)


def _empty_counts() -> Dict[str, int]:
    return {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
    }


def _add_counts(a: Mapping[str, int], b: Mapping[str, int]) -> Dict[str, int]:
    return {
        "tp": int(a.get("tp", 0)) + int(b.get("tp", 0)),
        "fp": int(a.get("fp", 0)) + int(b.get("fp", 0)),
        "fn": int(a.get("fn", 0)) + int(b.get("fn", 0)),
        "tn": int(a.get("tn", 0)) + int(b.get("tn", 0)),
    }


def _numpy_binary_counts(
    *,
    pred: np.ndarray,
    target: np.ndarray,
) -> Dict[str, int]:
    pred = pred.astype(bool)
    target = target.astype(bool)

    return {
        "tp": int(np.logical_and(pred, target).sum()),
        "fp": int(np.logical_and(pred, ~target).sum()),
        "fn": int(np.logical_and(~pred, target).sum()),
        "tn": int(np.logical_and(~pred, ~target).sum()),
    }


def _metrics_from_counts(
    counts: Mapping[str, int],
    *,
    eps: float = 1e-7,
) -> Dict[str, float]:
    tp = float(counts.get("tp", 0))
    fp = float(counts.get("fp", 0))
    fn = float(counts.get("fn", 0))
    tn = float(counts.get("tn", 0))

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    pixel_accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "pixel_accuracy": float(pixel_accuracy),
    }


def _to_numpy_bool(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()

    arr = np.asarray(x)

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    return arr.astype(bool)


def _probability_stats(probability: Any) -> Dict[str, float]:
    if probability is None:
        return {}

    if torch.is_tensor(probability):
        probability = probability.detach().cpu().numpy()

    arr = np.asarray(probability)

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    arr = arr.astype(np.float32, copy=False)

    return {
        "prob_mean": float(np.nanmean(arr)),
        "prob_std": float(np.nanstd(arr)),
        "prob_min": float(np.nanmin(arr)),
        "prob_max": float(np.nanmax(arr)),
    }


def _torch_binary_counts(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, int]:
    pred = pred.bool()
    target = target.bool()

    tp = torch.logical_and(pred, target).sum()
    fp = torch.logical_and(pred, ~target).sum()
    fn = torch.logical_and(~pred, target).sum()
    tn = torch.logical_and(~pred, ~target).sum()

    return {
        "tp": int(tp.detach().cpu()),
        "fp": int(fp.detach().cpu()),
        "fn": int(fn.detach().cpu()),
        "tn": int(tn.detach().cpu()),
    }
