from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from geo_mlops.core.contracts.train_contract import TrainContract
from geo_mlops.core.io.train_io import write_train_contract
from geo_mlops.core.training.callbacks import CallbackList, TrainingCallback
from geo_mlops.core.utils.cuda import _infer_batch_size
from geo_mlops.core.utils.dataclasses import _as_plain_dict
from geo_mlops.core.utils.random import _seed_everything


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    num_workers: int
    epochs: int
    lr: float
    seed: int
    selection_metric: str = "val/loss"
    selection_mode: str = "min"  # "min" or "max"


def _is_better(value: float, best: float | None, mode: str) -> bool:
    if best is None:
        return True
    if mode == "min":
        return value < best
    if mode == "max":
        return value > best
    raise ValueError(f"selection_mode must be 'min' or 'max', got {mode!r}")


def _prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}/{k}": float(v) for k, v in metrics.items()}


def train_one_run(
    *,
    model: torch.nn.Module,
    loss_fn: Callable[[Any, dict[str, Any]], torch.Tensor],
    train_ds,
    val_ds,
    train_dir_path: Path,
    device: torch.device,
    engine_cfg: TrainConfig,
    forward_fn: Callable[[torch.nn.Module, dict[str, Any], torch.device], Any],
    task: str,
    train_cfg: dict[str, Any],
    metrics_fn: Callable[[Any, dict[str, Any]], dict[str, float]] | None = None,
    optimizer_factory: Callable[[torch.nn.Module, TrainConfig], torch.optim.Optimizer] | None = None,
    callbacks: list[TrainingCallback] | None = None,
) -> tuple[Path, TrainContract]:
    train_dir_path = Path(train_dir_path)
    train_dir_path.mkdir(parents=True, exist_ok=True)

    _seed_everything(engine_cfg.seed)
    model.to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=engine_cfg.batch_size,
        shuffle=True,
        num_workers=engine_cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=engine_cfg.batch_size,
        shuffle=False,
        num_workers=engine_cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    opt = optimizer_factory(model, engine_cfg) if optimizer_factory is not None else torch.optim.AdamW(model.parameters(), lr=engine_cfg.lr)

    best_metric_value: float | None = None
    best_epoch: int | None = None
    history: dict[str, Any] = {}

    model_path = train_dir_path / "model.pt"
    metrics_path = train_dir_path / "metrics.json"

    callback_list = CallbackList(callbacks)

    engine_cfg_dict = _as_plain_dict(engine_cfg)

    callback_list.on_train_start(
        model=model,
        train_dir_path=train_dir_path,
        device=device,
        train_cfg=train_cfg,
        engine_cfg=engine_cfg_dict,
    )

    for epoch in range(1, engine_cfg.epochs + 1):
        model.train()

        train_loss_sum = 0.0
        n_train = 0

        for batch in train_loader:
            opt.zero_grad(set_to_none=True)

            outputs = forward_fn(model, batch, device)
            loss = loss_fn(outputs, batch)

            loss.backward()
            opt.step()

            batch_size = _infer_batch_size(batch)
            train_loss_sum += float(loss.item()) * batch_size
            n_train += batch_size

        train_loss = train_loss_sum / max(1, n_train)

        model.eval()
        val_loss_sum = 0.0
        n_val = 0
        val_metric_sums: dict[str, float] = {}

        with torch.no_grad():
            for batch in val_loader:
                outputs = forward_fn(model, batch, device)
                loss = loss_fn(outputs, batch)

                batch_size = _infer_batch_size(batch)
                val_loss_sum += float(loss.item()) * batch_size
                n_val += batch_size

                if metrics_fn is not None:
                    batch_metrics = metrics_fn(outputs, batch)
                    for name, value in batch_metrics.items():
                        val_metric_sums[name] = val_metric_sums.get(name, 0.0) + float(value) * batch_size

        val_loss = val_loss_sum / max(1, n_val)
        val_metrics = {name: total / max(1, n_val) for name, total in val_metric_sums.items()}

        epoch_metrics = {
            "train/loss": float(train_loss),
            "val/loss": float(val_loss),
            **_prefix_metrics("val", val_metrics),
        }

        callback_list.on_epoch_end(
            epoch=epoch,
            metrics=epoch_metrics,
        )

        history[f"epoch_{epoch}"] = epoch_metrics

        if engine_cfg.selection_metric not in epoch_metrics:
            raise KeyError(f"selection_metric={engine_cfg.selection_metric!r} not found. Available metrics: {sorted(epoch_metrics.keys())}")

        current_value = float(epoch_metrics[engine_cfg.selection_metric])

        print(f"[epoch {epoch}] " + " ".join(f"{k}={v:.4f}" for k, v in epoch_metrics.items()))

        if _is_better(current_value, best_metric_value, engine_cfg.selection_mode):
            best_metric_value = current_value
            best_epoch = epoch

            torch.save(model.state_dict(), model_path)

            callback_list.on_checkpoint_saved(
                checkpoint_path=model_path,
                model=model,
                epoch=epoch,
                metric_name=engine_cfg.selection_metric,
                metric_value=current_value,
            )

    metrics_payload = {
        "selection_metric": engine_cfg.selection_metric,
        "selection_mode": engine_cfg.selection_mode,
        "best_metric_value": best_metric_value,
        "best_epoch": best_epoch,
        "history": history,
    }

    metrics_path.write_text(json.dumps(metrics_payload, indent=2))

    callback_list.on_train_end(
        model=model,
        model_path=model_path,
        metrics_path=metrics_path,
    )

    train_contract = TrainContract(
        task=task,
        train_dir_path=train_dir_path,
        model_path=model_path,
        metrics_path=metrics_path,
        num_train_tiles=int(len(train_ds)),
        num_val_tiles=int(len(val_ds)),
        train_cfg=_as_plain_dict(train_cfg),
        best_metric_value=best_metric_value,
        best_epoch=best_epoch,
        tracking=callback_list.state_dict(),
    )

    manifest_path = write_train_contract(train_contract)

    return manifest_path, train_contract
