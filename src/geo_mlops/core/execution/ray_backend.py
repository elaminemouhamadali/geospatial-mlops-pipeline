# src/geo_mlops/core/execution/ray_backend.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ray


@dataclass(frozen=True)
class RayBackendConfig:
    """
    Configuration for connecting to a Ray runtime.

    Common modes:
      - address="auto":
          Connect to an already-running Ray cluster.
          This is the normal mode inside a Ray head container or Ray driver pod.

      - address=None:
          Start a local in-process Ray runtime.
          Useful for debugging outside an existing cluster.

      - address="ray://...":
          Ray Client mode. Not recommended as the default for this repo's
          training/evaluation jobs, but possible later for interactive use.
    """

    address: str | None = "auto"
    namespace: str = "geo-mlops"
    ignore_reinit_error: bool = True
    log_cluster_resources: bool = True
    runtime_env: dict[str, Any] | None = None
    shutdown_on_exit: bool = False


def init_ray_backend(cfg: RayBackendConfig) -> None:
    """
    Initialize Ray according to the supplied backend config.

    This function is intentionally small and reusable across stages:
      - evaluation
      - training
      - inference
      - tiling
      - future embedding/mining jobs

    Stage-specific code should call this instead of calling ray.init(...)
    directly.
    """

    if ray.is_initialized():
        if cfg.log_cluster_resources:
            print("[ray-backend] Ray is already initialized.")
            print_ray_cluster_resources()
        return

    ray.init(
        address=cfg.address,
        namespace=cfg.namespace,
        runtime_env=cfg.runtime_env,
        ignore_reinit_error=cfg.ignore_reinit_error,
    )

    if cfg.log_cluster_resources:
        print_ray_cluster_resources()


def shutdown_ray_backend(cfg: RayBackendConfig) -> None:
    """
    Optionally shut down this Ray driver connection.

    When connected to an existing cluster, shutdown_on_exit should usually stay
    False. ray.shutdown() disconnects this driver; it does not stop the whole
    external Ray cluster.
    """

    if cfg.shutdown_on_exit and ray.is_initialized():
        ray.shutdown()


def require_ray_initialized() -> None:
    """
    Raise a clear error if Ray has not been initialized yet.
    """

    if not ray.is_initialized():
        raise RuntimeError("Ray is not initialized. Call init_ray_backend(...) before using Ray execution utilities.")


def get_ray_cluster_resources() -> dict[str, Any]:
    """
    Return Ray cluster resources after validating that Ray is initialized.
    """

    require_ray_initialized()
    return dict(ray.cluster_resources())


def get_ray_available_resources() -> dict[str, Any]:
    """
    Return currently available Ray resources after validating Ray is initialized.

    cluster_resources() reports total capacity.
    available_resources() reports what is currently free.
    """

    require_ray_initialized()
    return dict(ray.available_resources())


def get_ray_default_num_shards(*, resource: str = "CPU", fallback: int = 1) -> int:
    """
    Choose a reasonable default number of shards from Ray cluster resources.

    By default this returns the total number of CPUs in the Ray cluster.

    Args:
        resource:
            Ray resource key to inspect. Usually "CPU".
        fallback:
            Value used if the resource does not exist or Ray reports zero.

    Returns:
        Positive integer shard count.
    """

    require_ray_initialized()

    resources = ray.cluster_resources()
    value = int(resources.get(resource, fallback))

    return max(1, value)


def print_ray_cluster_resources() -> None:
    """
    Print total and currently available Ray resources.
    """

    require_ray_initialized()

    print("[ray-backend] Ray cluster resources:")
    print(ray.cluster_resources())

    print("[ray-backend] Ray available resources:")
    print(ray.available_resources())
