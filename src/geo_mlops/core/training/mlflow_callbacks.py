from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import torch

from geo_mlops.core.training.callbacks import TrainingCallback


class MLflowTrainingCallback(TrainingCallback):
    def __init__(
        self,
        *,
        experiment_name: str,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        tags: dict[str, str] | None = None,
        log_checkpoints: bool = True,
        log_model: bool = True,
        model_artifact_path: str = "model",
        log_system_metrics: bool = True,
    ) -> None:
        self.experiment_name = experiment_name
        self.requested_run_name = run_name
        self.tracking_uri = tracking_uri
        self.tags = tags or {}
        self.log_checkpoints = log_checkpoints
        self.log_model = log_model
        self.model_artifact_path = model_artifact_path
        self.log_system_metrics = log_system_metrics

        self._active = False
        self.run_id: str | None = None
        self.run_name: str | None = run_name
        self.experiment_id: str | None = None

    def on_train_start(
        self,
        *,
        model: torch.nn.Module,
        train_dir_path: Path,
        device: torch.device,
        train_cfg: dict[str, Any],
        engine_cfg: dict[str, Any],
    ) -> None:
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        mlflow.set_experiment(self.experiment_name)

        run = mlflow.start_run(
            run_name=self.requested_run_name,
            log_system_metrics=self.log_system_metrics,
        )

        self._active = True
        self.run_id = run.info.run_id
        self.run_name = run.info.run_name
        self.experiment_id = run.info.experiment_id

        mlflow.set_tags(self.tags)

        mlflow.log_params(_flatten_dict(engine_cfg, prefix="engine"))
        mlflow.log_params(_flatten_dict(train_cfg.get("model", {}) or {}, prefix="model"))
        mlflow.log_params(_flatten_dict(train_cfg.get("dataset", {}) or {}, prefix="dataset"))
        mlflow.log_params(_flatten_dict(train_cfg.get("loss", {}) or {}, prefix="loss"))
        mlflow.log_params(_flatten_dict(train_cfg.get("metrics", {}) or {}, prefix="metrics"))
        mlflow.log_params(_flatten_dict(train_cfg.get("sampler", {}) or {}, prefix="sampler"))

    def on_epoch_end(
        self,
        *,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        mlflow.log_metrics(metrics, step=epoch)

    def on_checkpoint_saved(
        self,
        *,
        checkpoint_path: Path,
        model: torch.nn.Module,
        epoch: int,
        metric_name: str,
        metric_value: float,
    ) -> None:
        if self.log_checkpoints:
            mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")

    def on_train_end(
        self,
        *,
        model: torch.nn.Module,
        model_path: Path,
        metrics_path: Path,
    ) -> None:
        try:
            mlflow.log_artifact(str(metrics_path), artifact_path="training")
            mlflow.log_artifact(str(model_path), artifact_path="checkpoints")

            if self.log_model:
                was_training = model.training
                model.eval()

                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path=self.model_artifact_path,
                )

                if was_training:
                    model.train()

        finally:
            if self._active:
                mlflow.end_run()
                self._active = False

    def state_dict(self) -> dict[str, Any]:
        return {
            "mlflow_run_id": self.run_id,
            "mlflow_run_name": self.run_name,
            "mlflow_experiment_name": self.experiment_name,
            "mlflow_experiment_id": self.experiment_id,
            "mlflow_tracking_uri": self.tracking_uri,
            "mlflow_model_artifact_path": self.model_artifact_path,
            "mlflow_model_uri": (f"runs:/{self.run_id}/{self.model_artifact_path}" if self.run_id else None),
        }


def _flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}

    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)

        if isinstance(v, dict):
            out.update(_flatten_dict(v, prefix=key))
        elif isinstance(v, (str, int, float, bool)) or v is None:
            out[key] = v
        else:
            out[key] = str(v)

    return out
