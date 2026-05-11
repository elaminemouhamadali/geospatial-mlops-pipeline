from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from geo_mlops.core.io.train_io import load_train_contract
from geo_mlops.core.registry.task_registry import get_task
from geo_mlops.core.data.scene_discovery import discover_dataset_scenes


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run full-scene sliding-window inference on a golden dataset using "
            "a trained task model."
        )
    )
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--task-cfg-path", "--task_cfg_path", dest="task_cfg_path", type=Path, required=True)
    p.add_argument("--dataset-root-path", "--dataset_root_path", dest="dataset_root_path", type=Path, required=True)
    p.add_argument("--train-manifest-path", "--train_manifest_path", dest="train_manifest_path", type=Path, required=True)
    p.add_argument("--inference-dir-path", "--inference_dir_path", dest="inference_dir_path", type=Path, required=True)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--execution-backend", choices=["local", "ray"], default="local")

    p.add_argument("--ray-address", type=str, default=None)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--items-per-shard", "--items_per_shard", dest="items_per_shard", type=int, default=None)
    p.add_argument("--num-gpus-per-worker", type=float, default=1.0)
    p.add_argument("--num-cpus-per-worker", type=int, default=4)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    args.inference_dir_path.mkdir(parents=True, exist_ok=True)

    task_plugin = get_task(args.task)

    train_cfg = task_plugin.build_training_cfg(args.task_cfg_path)

    train_contract = load_train_contract(args.train_manifest_path)
    checkpoint_path = train_contract.model_path

    inference_cfg = task_plugin.build_inference_cfg(args.task_cfg_path)
    inference_engine_cfg = task_plugin.build_inference_engine_cfg(inference_cfg)

    scene_layout = task_plugin.resolve_inference_layout(inference_cfg)

    scenes, roi_names, discovery_stats = discover_dataset_scenes(
        dataset_root=args.dataset_root_path,
        layout=scene_layout,
    )

    if args.execution_backend == "ray":
        from geo_mlops.core.execution.ray_backend import (
            RayBackendConfig,
            init_ray_backend,
            shutdown_ray_backend,
        )
        from geo_mlops.core.inference.ray_runner import run_full_scene_inference_ray

        ray_cfg = RayBackendConfig(
            address=args.ray_address,
            namespace="geo-mlops",
        )

        init_ray_backend(ray_cfg)

        try:
            manifest_path, contract = run_full_scene_inference_ray(
                task=args.task,
                scenes=scenes,
                inference_out_dir=args.inference_dir_path,
                inference_cfg=inference_engine_cfg,
                model_factory_fn=lambda: task_plugin.build_model(train_cfg),
                load_checkpoint_fn=task_plugin.load_checkpoint,
                checkpoint_path=checkpoint_path,
                device=args.device,
                load_scene_fn=lambda scene: task_plugin.load_inference_scene(scene, inference_cfg),
                forward_fn=task_plugin.get_forward_fn(),
                postprocess_fn=task_plugin.build_inference_postprocessor(inference_cfg),
                save_prediction_fn=task_plugin.save_inference_prediction,
                num_workers=args.num_workers,
                items_per_shard=args.items_per_shard,
                num_cpus_per_worker=args.num_cpus_per_worker,
                num_gpus_per_worker=args.num_gpus_per_worker,
            )
        finally:
            shutdown_ray_backend(ray_cfg)

    else:
        from geo_mlops.core.utils.cuda import _resolve_device
        from geo_mlops.core.inference.engine import run_full_scene_inference

        device = _resolve_device(args.device)

        model = task_plugin.build_model(train_cfg)
        model = task_plugin.load_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            device=device,
        )

        manifest_path, contract = run_full_scene_inference(
            task=args.task,
            model=model,
            device=device,
            scenes=scenes,
            inference_out_dir=args.inference_dir_path,
            inference_cfg=inference_engine_cfg,
            load_scene_fn=lambda scene: task_plugin.load_inference_scene(scene, inference_cfg),
            forward_fn=task_plugin.get_forward_fn(),
            postprocess_fn=task_plugin.build_inference_postprocessor(inference_cfg),
            save_prediction_fn=task_plugin.save_inference_prediction,
            checkpoint_path=checkpoint_path,
        )

    print("[inference] done")
    print(f"[inference] backend={args.execution_backend}")
    print(f"[inference] rois={roi_names}")
    print(f"[inference] discovered_scenes={discovery_stats.get('scenes_discovered')}")
    print(f"[inference] scenes={contract.num_scenes}")
    print(f"[inference] manifest={manifest_path}")
    print(f"[inference] prediction_table={contract.prediction_table_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())