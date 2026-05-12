# infra/ray-local/smoke_test.py

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

import ray


@ray.remote(num_cpus=1)
def inspect_worker(task_id: int, output_dir: str) -> dict:
    import rasterio
    import torch
    import transformers

    import geo_mlops

    host = socket.gethostname()
    node_id = ray.get_runtime_context().get_node_id()

    out_path = Path(output_dir) / f"ray_smoke_task_{task_id}_{host}.json"
    payload = {
        "task_id": task_id,
        "hostname": host,
        "node_id": node_id,
        "pid": os.getpid(),
        "geo_mlops_file": getattr(geo_mlops, "__file__", None),
        "rasterio_version": rasterio.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "transformers_version": transformers.__version__,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    time.sleep(1)
    return payload


def main() -> None:
    output_dir = "/outputs/ray-smoke-test"

    ray.init(address="auto")

    print("Connected to Ray.")
    print("Cluster resources:")
    print(json.dumps(ray.cluster_resources(), indent=2, default=str))

    refs = [inspect_worker.remote(i, output_dir) for i in range(12)]
    results = ray.get(refs)

    hosts = sorted({r["hostname"] for r in results})
    node_ids = sorted({r["node_id"] for r in results})

    print("\nTask results:")
    for r in results:
        print(json.dumps(r, indent=2))

    print("\nSummary:")
    print(f"Unique hosts used: {hosts}")
    print(f"Unique Ray node IDs used: {node_ids}")
    print(f"Output dir: {output_dir}")

    if len(hosts) < 2:
        raise RuntimeError("Smoke test only used one host. Ray may not be distributing tasks across workers.")

    print("\nRay smoke test PASSED.")


if __name__ == "__main__":
    main()
