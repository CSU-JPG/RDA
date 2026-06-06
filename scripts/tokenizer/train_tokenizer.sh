#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

DATA_PATH="${DATA_PATH:-${REPO_ROOT}/examples/inference/sample_images.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/rda_train}"
VQ_CKPT="${VQ_CKPT:-${REPO_ROOT}/pretrained_model/vq_ds16_t2i.pt}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-16}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
EPOCHS="${EPOCHS:-20}"
CKPT_EVERY="${CKPT_EVERY:-200}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  "${REPO_ROOT}/tokenizer/tokenizer_image/resvq_train.py" \
  --dataset json_data \
  --data-path "${DATA_PATH}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --resvq-codebook-embed-dim 16 \
  --log-every 1 \
  --lr 1e-4 \
  --num-workers "${NUM_WORKERS}" \
  --cloud-save-path "${OUTPUT_DIR}" \
  --image-size "${IMAGE_SIZE}" \
  --vq-ckpt "${VQ_CKPT}" \
  --reconstruction-loss l1 \
  --freq_loss \
  --freq_loss_weight 1.0 \
  --freq_q 0.1 \
  --res_p_loss \
  --sum_p_loss \
  --sum_rec_loss \
  --sobel_recon_loss \
  --epochs "${EPOCHS}" \
  --disc-start 200000 \
  --ckpt-every "${CKPT_EVERY}"
