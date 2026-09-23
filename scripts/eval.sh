#!/usr/bin/env bash
DATA_ROOT="${DATA_ROOT:-/path/to/datasets}"

NAME="ashmil"
CKPT="outputs/${NAME}/checkpoint_best.pth"
OUT="outputs/${NAME}_eval_best"

echo "[EVAL] ${NAME} (best) -> ${OUT}"
python tools/final_eval_ashmil.py \
  --ckpt "${CKPT}" \
  --output_dir "${OUT}" \
  --nih_ann "data/cxr8/coco_det/annotations/instances_test2017.json" \
  --nih_image_dir "$DATA_ROOT/CXR8/images" \
  --nih_prior_dir "$DATA_ROOT/CXR8/cxas_mask" \
  --mimic_ann "data/mimic/annotations/221v2hiqualnihsplit.json" \
  --mimic_prior_dir "$DATA_ROOT/mimic-cxr-jpg/cxas_mask" \
  --mimic_files_root "$DATA_ROOT/mimic-cxr-jpg/files" \
  --mimic_metadata_csv "$DATA_ROOT/mimic-cxr-jpg/mimic-cxr-2.0.0-metadata.csv" \
  --iou_thrs 0.1,0.2,0.3,0.4,0.5 \
  "$@"
