#!/bin/bash
DATA_ROOT="${DATA_ROOT:-/path/to/datasets}"

OUT_DIR="outputs/ashmil"

python train.py \
  --image_dir "${DATA_ROOT}/CXR8/images" \
  --prior_dir "${DATA_ROOT}/CXR8/cxas_mask" \
  --output_dir "${OUT_DIR}"
