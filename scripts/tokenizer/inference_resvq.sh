#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data_debug.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results_debug/inference}"
VQ_CKPT="${VQ_CKPT:-${REPO_ROOT}/pretrained_model/vq_ds16_t2i.pt}"
RDA_MODEL="${RDA_MODEL:-${RESVQ_CKPT:-CSU-JPG/RDA_llamagen}}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-1}"
GPU="${GPU:-0}"

CMD=(
  "${REPO_ROOT}/tokenizer/tokenizer_image/resvq_inference.py"
  --dataset json_data
  --data-path "${DATA_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --resvq-codebook-embed-dim 16
  --vq-ckpt "${VQ_CKPT}"
  --rda-model-path "${RDA_MODEL}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --gpu "${GPU}"
)

if [[ -n "${IMAGE_SIZE:-}" ]]; then
  CMD+=(--image-size "${IMAGE_SIZE}")
fi

if [[ -n "${NPROC_PER_NODE:-}" ]]; then
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${CMD[@]}"
else
  python "${CMD[@]}"
fi
