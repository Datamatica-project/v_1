#!/bin/bash
set -e

echo "🚀 V1 Auto Labeling Worker starting..."

export PYTHONPATH=/workspace
: "${WORKER_HOST:=0.0.0.0}"
: "${WORKER_PORT:=8011}"

: "${WORKER_JOB_DIR:=/workspace/auto_labeling/v_1/logs/jobs}"

echo "✅ WORKER_HOST=${WORKER_HOST}"
echo "✅ WORKER_PORT=${WORKER_PORT}"
echo "✅ WORKER_JOB_DIR=${WORKER_JOB_DIR}"

exec python -m uvicorn auto_labeling.v_1.worker.server:app \
  --host "${WORKER_HOST}" \
  --port "${WORKER_PORT}"