from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional


class TrainingCallback:
    def on_train_start(self, context: Dict[str, Any]) -> None:
        pass

    def on_epoch_end(
        self,
        epoch: int,
        metrics: Dict[str, float],
        context: Dict[str, Any],
    ) -> None:
        pass

    def on_checkpoint_saved(
        self,
        checkpoint_path: Path,
        context: Dict[str, Any],
    ) -> None:
        pass

    def on_train_end(
        self,
        outputs: Any,
        context: Dict[str, Any],
    ) -> None:
        pass

    def state_dict(self) -> Dict[str, Any]:
        """
        Optional callback state to persist into stage manifests.

        Example:
          MLflow callback can expose mlflow_run_id, run_name, tracking_uri, etc.
        """
        return {}


class CallbackList:
    def __init__(self, callbacks: Optional[Iterable[TrainingCallback]] = None):
        self.callbacks = list(callbacks or [])

    def on_train_start(self, context: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_train_start(context)

    def on_epoch_end(
        self,
        epoch: int,
        metrics: Dict[str, float],
        context: Dict[str, Any],
    ) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, metrics, context)

    def on_checkpoint_saved(
        self,
        checkpoint_path: Path,
        context: Dict[str, Any],
    ) -> None:
        for cb in self.callbacks:
            cb.on_checkpoint_saved(checkpoint_path, context)

    def on_train_end(self, outputs: Any, context: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_train_end(outputs, context)

    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {}

        for cb in self.callbacks:
            cb_state = cb.state_dict()
            if cb_state:
                state.update(cb_state)

        return state