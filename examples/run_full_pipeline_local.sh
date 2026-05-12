#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root robustly, regardless of where this script lives.
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

TASK="${TASK:-building_seg}"
TASK_CFG_PATH="${TASK_CFG_PATH:-examples/tiny_building_seg/config/default.yaml}"
DATASET_ROOT_PATH="${DATASET_ROOT_PATH:-examples/tiny_building_seg/train_data}"
GOLDEN_ROOT_PATH="${GOLDEN_ROOT_PATH:-examples/tiny_building_seg/test_data}"
DEVICE="${DEVICE:-cpu}"
RUN_DIR_PATH="${RUN_DIR_PATH:-outputs/tiny_building_seg_local}"
EXECUTION_BACKEND="${EXECUTION_BACKEND:-local}"
ENABLE_MLFLOW="${ENABLE_MLFLOW:-1}"

EXTRA_ARGS=()

if [[ "${ENABLE_MLFLOW}" == "1" ]]; then
  MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
  MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT:-tiny_building_seg_demo}"

  echo "[tiny-demo] mlflow enabled"
  echo "[tiny-demo] mlflow_tracking_uri=${MLFLOW_TRACKING_URI}"
  echo "[tiny-demo] mlflow_experiment=${MLFLOW_EXPERIMENT}"

  if ! curl -fsS "${MLFLOW_TRACKING_URI}/health" >/dev/null 2>&1; then
    echo "[tiny-demo] ERROR: MLflow server is not reachable at ${MLFLOW_TRACKING_URI}"
    echo "[tiny-demo] Start one locally with:"
    echo "  mlflow server \\"
    echo "    --host 127.0.0.1 \\"
    echo "    --port 5000 \\"
    echo "    --backend-store-uri sqlite:///mlflow.db \\"
    echo "    --default-artifact-root ./mlruns"
    exit 1
  fi
  
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

echo "[tiny-demo] repo_root=${REPO_ROOT}"
echo "[tiny-demo] task=${TASK}"
echo "[tiny-demo] backend=${EXECUTION_BACKEND}"
echo "[tiny-demo] task_cfg=${TASK_CFG_PATH}"
echo "[tiny-demo] dataset_root=${DATASET_ROOT_PATH}"
echo "[tiny-demo] golden_root=${GOLDEN_ROOT_PATH}"
echo "[tiny-demo] run_dir=${RUN_DIR_PATH}"
echo "[tiny-demo] device=${DEVICE}"

test -f "${TASK_CFG_PATH}" || {
  echo "[tiny-demo] ERROR: task config not found: ${TASK_CFG_PATH}"
  pwd
  ls -lah src/geo_mlops/tasks/segmentation/building/config || true
  exit 1
}

python -m geo_mlops.cli.run_pipeline \
  --task "${TASK}" \
  --task-cfg-path "${TASK_CFG_PATH}" \
  --dataset-root-path "${DATASET_ROOT_PATH}" \
  --golden-root-path "${GOLDEN_ROOT_PATH}" \
  --run-dir-path "${RUN_DIR_PATH}" \
  --device "${DEVICE}" \
  --force-tiling \
  "${EXTRA_ARGS[@]}"