#!/usr/bin/env bash
set -euo pipefail

# Example local full pipeline run.
# Update these paths for your machine.

TASK="building_seg"
TASK_CFG="src/geo_mlops/tasks/segmentation/building/config/default.yaml"
DATASET_ROOT="/mnt/d/SpaceNet/Regions"
GOLDEN_ROOT="/mnt/d/SpaceNet/Golden"
RUN_DIR="/mnt/d/SpaceNet/Run-Full-May"
MLFLOW_URI="http://127.0.0.1:5000"

python -m geo_mlops.cli.run_pipeline \
  --task "${TASK}" \
  --task-cfg "${TASK_CFG}" \
  --dataset-root "${DATASET_ROOT}" \
  --golden-root "${GOLDEN_ROOT}" \
  --run-dir "${RUN_DIR}" \
  --csv-name bldg_tiles_regular.csv \
  --mlflow \
  --mlflow-tracking-uri "${MLFLOW_URI}" \
  --mlflow-experiment building_seg_debug \
  --force-tiling
