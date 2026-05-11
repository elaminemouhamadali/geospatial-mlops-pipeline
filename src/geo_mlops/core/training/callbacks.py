from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch


class TrainingCallback:
    def on_train_start(
        self,
        *,
        model: torch.nn.Module,
        train_dir_path: Path,
        device: torch.device,
        train_cfg: Dict[str, Any],
        engine_cfg: Dict[str, Any],
    ) -> None:
        pass

    def on_epoch_end(
        self,
        *,
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        pass

    def on_checkpoint_saved(
        self,
        *,
        checkpoint_path: Path,
        model: torch.nn.Module,
        epoch: int,
        metric_name: str,
        metric_value: float,
    ) -> None:
        pass

    def on_train_end(
        self,
        *,
        model: torch.nn.Module,
        model_path: Path,
        metrics_path: Path,
    ) -> None:
        pass

    def state_dict(self) -> Dict[str, Any]:
        return {}


class CallbackList:
    def __init__(self, callbacks: Optional[Iterable[TrainingCallback]] = None):
        self.callbacks = list(callbacks or [])

    def on_train_start(
        self,
        *,
        model: torch.nn.Module,
        train_dir_path: Path,
        device: torch.device,
        train_cfg: Dict[str, Any],
        engine_cfg: Dict[str, Any],
    ) -> None:
        for cb in self.callbacks:
            cb.on_train_start(
                model=model,
                train_dir_path=train_dir_path,
                device=device,
                train_cfg=train_cfg,
                engine_cfg=engine_cfg,
            )

    def on_epoch_end(
        self,
        *,
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(epoch=epoch, metrics=metrics)

    def on_checkpoint_saved(
        self,
        *,
        checkpoint_path: Path,
        model: torch.nn.Module,
        epoch: int,
        metric_name: str,
        metric_value: float,
    ) -> None:
        for cb in self.callbacks:
            cb.on_checkpoint_saved(
                checkpoint_path=checkpoint_path,
                model=model,
                epoch=epoch,
                metric_name=metric_name,
                metric_value=metric_value,
            )

    def on_train_end(
        self,
        *,
        model: torch.nn.Module,
        model_path: Path,
        metrics_path: Path,
    ) -> None:
        for cb in self.callbacks:
            cb.on_train_end(
                model=model,
                model_path=model_path,
                metrics_path=metrics_path,
            )

    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {}

        for cb in self.callbacks:
            cb_state = cb.state_dict()
            if cb_state:
                state.update(cb_state)

        return state