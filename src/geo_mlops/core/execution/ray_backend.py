# src/geo_mlops/core/execution/ray_backend.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import ray


@dataclass(frozen=True)
class RayBackendConfig:
    address: Optional[str] = "auto"
    namespace: str = "geo-mlops"
    ignore_reinit_error: bool = True
    log_cluster_resources: bool = True
    runtime_env: Optional[dict[str, Any]] = None
    shutdown_on_exit: bool = False


def init_ray_backend(cfg: RayBackendConfig) -> None:
    if ray.is_initialized():
        return

    ray.init(
        address=cfg.address,
        namespace=cfg.namespace,
        runtime_env=cfg.runtime_env,
        ignore_reinit_error=cfg.ignore_reinit_error,
    )

    if cfg.log_cluster_resources:
        print("Ray cluster resources:")
        print(ray.cluster_resources())


def shutdown_ray_backend(cfg: RayBackendConfig) -> None:
    if cfg.shutdown_on_exit and ray.is_initialized():
        ray.shutdown()