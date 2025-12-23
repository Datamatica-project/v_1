#!/bin/bash
set -e

echo "🚀 V1 Auto Labeling API Server starting..."

export PYTHONPATH=/workspace

uvicorn auto_labeling.v_1.api.server:app \
  --host 0.0.0.0 \
  --port 8010 \
  --log-level info