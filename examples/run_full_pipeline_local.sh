#!/usr/bin/env bash
set -euo pipefail

# Run from repository root regardless of caller location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TASK="${TASK:-building_seg}"
TASK_CFG_PATH="${TASK_CFG_PATH:-src/geo_mlops/tasks/segmentation/building/config/default.yaml}"
DATASET_ROOT_PATH="${DATASET_ROOT_PATH:-examples/tiny_building_seg/train_data}"
GOLDEN_ROOT_PATH="${GOLDEN_ROOT_PATH:-examples/tiny_building_seg/test_data}"
DEVICE="${DEVICE:-cpu}"
RUN_DIR_PATH="${RUN_DIR_PATH:-outputs/tiny_building_seg_local}"
EXECUTION_BACKEND="${EXECUTION_BACKEND:-local}"

EXTRA_ARGS=()

if [[ "${ENABLE_MLFLOW:-0}" == "1" ]]; then
  EXTRA_ARGS+=(
    --mlflow
    --mlflow-tracking-uri "${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
    --mlflow-experiment "${MLFLOW_EXPERIMENT:-tiny_building_seg_demo}"
  )
fi

if [[ "${EXECUTION_BACKEND}" == "ray" ]]; then
  EXTRA_ARGS+=(
    --execution-backend ray
    --ray-address "${RAY_ADDRESS:-auto}"
    --items-per-shard "${ITEMS_PER_SHARD:-2}"
    --num-gpus-per-worker "${NUM_GPUS_PER_WORKER:-0}"
    --num-cpus-per-worker "${NUM_CPUS_PER_WORKER:-2}"
  )
else
  EXTRA_ARGS+=(
    --execution-backend local
  )
fi

echo "[tiny-demo] task=${TASK}"
echo "[tiny-demo] backend=${EXECUTION_BACKEND}"
echo "[tiny-demo] dataset_root=${DATASET_ROOT_PATH}"
echo "[tiny-demo] golden_root=${GOLDEN_ROOT_PATH}"
echo "[tiny-demo] run_dir=${RUN_DIR_PATH}"
echo "[tiny-demo] device=${DEVICE}"

python -m geo_mlops.cli.run_pipeline \
  --task "${TASK}" \
  --task-cfg-path "${TASK_CFG_PATH}" \
  --dataset-root-path "${DATASET_ROOT_PATH}" \
  --golden-root-path "${GOLDEN_ROOT_PATH}" \
  --run-dir-path "${RUN_DIR_PATH}" \
  --device "${DEVICE}" \
  --force-tiling \
  "${EXTRA_ARGS[@]}"