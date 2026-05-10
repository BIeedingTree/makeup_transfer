#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/DATA/yantongliu/FinalProj/Stable-Makeup"

conda run -n finalproj python "${PROJECT_DIR}/tools/infer_one_id.py" \
  --project-dir "${PROJECT_DIR}" \
  --id-name "5.jpg"
