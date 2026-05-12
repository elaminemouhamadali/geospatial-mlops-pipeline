# Geospatial MLOps Pipeline

A contract-driven MLOps framework for remote-sensing task-agnostic workflows, built to move geospatial ML tasks from local experimentation to reproducible Docker, MLflow, Ray-distributed, and Kubernetes-ready execution. Enhances development lifecycles and experimentations by 10x

[![CI](https://github.com/maelamine-genai/geospatial-mlops-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/maelamine-genai/geospatial-mlops-pipeline/actions/workflows/ci.yml)
[![Docker Publish](https://github.com/maelamine-genai/geospatial-mlops-pipeline/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/maelamine-genai/geospatial-mlops-pipeline/actions/workflows/docker-publish.yml)
[![Demo Pipeline Local](https://github.com/maelamine-genai/geospatial-mlops-pipeline/actions/workflows/demo-pipeline-local.yml/badge.svg)](https://github.com/maelamine-genai/geospatial-mlops-pipeline/actions/workflows/demo-pipeline-local.yml)
[![Demo Ray Local Cluster](https://github.com/maelamine-genai/geospatial-mlops-pipeline/actions/workflows/demo-pipeline-ray.yml/badge.svg)](https://github.com/maelamine-genai/geospatial-mlops-pipeline/actions/workflows/demo-pipeline-ray.yml)

## Verified Execution Paths

| Workflow | What it proves |
|---|---|
| [CI](https://github.com/maelamine-genai/geospatial-mlops-pipeline/actions/workflows/ci.yml) | Linting, formatting, and tests pass |
| [Docker Publish](https://github.com/maelamine-genai/geospatial-mlops-pipeline/actions/workflows/docker-publish.yml) | Runtime images are built and published to GHCR |
| [Demo Pipeline Local](https://github.com/maelamine-genai/geospatial-mlops-pipeline/actions/workflows/demo-pipeline-local.yml) | Full tiny building-segmentation lifecycle runs with MLflow tracking |
| [Demo Ray Local Cluster](https://github.com/maelamine-genai/geospatial-mlops-pipeline/actions/workflows/demo-pipeline-ray.yml) | Docker Compose Ray cluster starts and executes distributed worker tasks |

The demo workflows upload pipeline outputs, MLflow metadata, Ray logs, and smoke-test artifacts as GitHub Actions artifacts.

The screenshot below shows the GitHub Actions runs used as executable proof that the repository builds, tests, publishes images, and validates the demo execution paths.

![GitHub Actions proof](docs/assets/github_actions_proof.png)


## System overview

![Geospatial MLOps Pipeline Architecture](docs/assets/pipeline_architecture.png)

The diagram above shows the full system: task plugins, execution backends, MLflow governance, Docker/CI, and the end-to-end ML lifecycle.

For a more implementation-focused view, the pipeline is organized around stage contracts:
### Contract artifacts

Each pipeline stage writes a small JSON contract that records the inputs, outputs, configuration, and artifact paths needed by downstream stages. This keeps the pipeline reproducible and makes local, Docker, Ray, and future Kubernetes execution use the same interface.

| Stage | Contract / artifact | Purpose |
|---|---|---|
| Tiling | `tiles_manifest.json` | Records generated chip inventory, task config, dataset roots, tiling policy, and master CSV path |
| Splitting | `split_manifest.json` | Records deterministic train/val/test split outputs and grouping/stratification metadata |
| Training | `train_manifest.json` | Records model checkpoint path, training config, metrics, and optional MLflow run metadata |
| Gate A | `gate_a_manifest.json` | Records validation-gate decision used to accept/reject candidate registration |
| MLflow Candidate Registry | `registry_candidate.json` | Tracks candidates model lifecycle state |
| Inference | `inference_manifest.json` `prediction_inventory.csv` | Records golden-set inference configuration and tabular inventory of saved predictions, probabilities, source scenes, and GT paths|
| Evaluation | `eval_manifest.json` | Records scoring configuration, metrics path, per-scene metrics table, and analytics artifacts |
| Evaluation | `metrics.json` | Machine-readable aggregate metrics used by promotion gates |
| Gate B | `gate_b_manifest.json` | Records golden-set KPI decision used to promote/reject the candidate model |
| MLflow Production Registry | `registry_production.json` | Tracks production model lifecycle state |


The key design choice is that pipeline stages communicate through explicit contracts rather than hidden in-memory state. A downstream stage does not need to know whether the previous stage ran locally, inside Docker, or through Ray. It only needs the previous stage’s contract artifact.

This is what allows the same pipeline to support:

- local development
- Dockerized execution
- Ray-distributed inference and evaluation
- KubeRay deployment templates
- Argo or Vertex orchestration


![Execution Modes](docs/assets/execution_modes.png)

The orchestrator can change without changing the core stage contracts. Argo, KubeRay, and Vertex are deployment/orchestration targets around the same CLI and manifest interface.

## Tiny building-segmentation demo

The repository includes a small committed GeoTIFF fixture under `examples/tiny_building_seg/` so the pipeline can be executed without downloading a full remote-sensing dataset.

The demo is intentionally small, but it exercises the same production-style lifecycle used by the full pipeline:

1. Tile training scenes
2. Create deterministic train/validation splits
3. Train a task model
4. Apply Gate A on validation metrics
5. Register a candidate model in MLflow
6. Run inference on a golden test set
7. Score predictions against ground truth
8. Apply Gate B on golden-set KPIs
9. Promote the model in MLflow when policy checks pass

### Demo data layout

```text
examples/tiny_building_seg/
  train_data/
    Khartoum/000
    Paris/000

  test_data/
    Khartoum/001
```

The demo uses the same task plugin and configuration interface as larger geospatial datasets. The tiny files are only used to make the repository self-validating and easy to run in CI.

### Start MLflow locally

```bash
mlflow server \
  --backend-store-uri sqlite:////tmp/geo_mlops_mlflow.db \
  --default-artifact-root /tmp/geo_mlops_mlruns \
  --host 127.0.0.1 \
  --port 5000
```

### Run the local demo
In another terminal, from the repository root:
```bash
bash examples/run_full_pipeline_local.sh
```


## MLflow governance

![MLflow Experiments](docs/assets/mlflow_experiments.png)

![MLflow Models](docs/assets/mlflow_models.png)

## Ray distributed execution

![Ray Distributed Inference](docs/assets/ray_inference.png)

![Ray Nodes](docs/assets/ray_workers.png)

## Docker images

- [Local Images](docker/README.md)

## Kubernetes / KubeRay readiness

- [KubeRay Configurations](infra/kuberay/raycluster-cpu.yaml)
- [Local Ray Configurations](infra/ray-local/docker-compose.yml)

## Orchestration targets (Argo)

![Argo Example](docs/assets/argo_example.png)

## Documentation

- [Pipeline Walkthrough (Step by Step)](src/geo_mlops/cli/README.md)
- [Task Plugins](src/geo_mlops/tasks/README.md)
- [Core Engines](src/geo_mlops/core/README.md)

## Repository structure

```text
.github/wokflows
  ci.yml
  demo-local.yml
  demo-ray.yml
  docker-publish.yml

docker/
  Dockerfile.*

infra
  kuberay
    raycluster.yml
  ray-local
    docker-compose.yml

src/geo_mlops/
  cli/
    tile.py
    split.py
    train.py
    gate.py
    register.py
    inference.py
    evaluate.py
    run_pipeline.py

  core/
    config/
    contracts/
    data/
    evaluation/
    execution/
    gating/
    io/
    registry/
    splitting/
    tiling/
    training/
    utils/

  models/
    backbones/
    fusion/

  tasks/
    segmentation/
      segmentation_adapter.py
      building/
        config/
        data/
        evaluation/
        inference/
        modeling/
        tiling/
        task.py

tests/test_imports_*
```

## Technical highlights

Bullet list.