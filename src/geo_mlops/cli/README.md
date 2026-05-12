# CLI Layer

`geo_mlops.cli` contains thin command-line wrappers around the core pipeline stages.

## Commands

| Command | Purpose |
|---|---|
| `tile.py` | Generate tile records and `tiles_manifest.json`. |
| `split.py` | Create deterministic group-aware train/validation splits. |
| `train.py` | Train a task model and write training artifacts. |
| `gate.py` | Evaluate KPI thresholds and write `gate_decision.json`. |
| `register.py` | Register candidate models and promote production versions in MLflow. |
| `inference.py` | Run full-scene golden inference with sliding-window inference. |
| `evaluate.py` | Run full-scene golden evaluation (scoring) |
| `run_pipeline.py` | Orchestrate the full A-to-Z workflow. |

## 0. Start MLflow locally

```bash
mlflow server \
  --backend-store-uri sqlite:////tmp/geo_mlops_mlflow.db \
  --default-artifact-root /tmp/geo_mlops_mlruns \
  --host 127.0.0.1 \
  --port 5000
```
## 1. One-command pipeline
```bash
python -m geo_mlops.cli.run_pipeline \
  --task <task> \
  --task-cfg <path/to/task/cfg> \
  --dataset-root </path/to/train_val_dataset_root> \
  --golden-root </path/to/golden_full_scene_dataset_root> \
  --run-dir </path/to/output/run> \
  --mlflow \
  --mlflow-tracking-uri http://127.0.0.1:5000 \
  --mlflow-experiment <experiment-name>
```

Each stage can also run on its own, assuming the artifacts of the stages it depends on exist

## Design rule

Do not put heavy logic here. Heavy logic belongs in `core/` if it's task-agnostic or in task plugins under `tasks/` if it's task specific.
