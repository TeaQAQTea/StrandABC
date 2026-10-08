#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
PYTHON="${PYTHON:-/home/zwang/.conda/envs/torch25/bin/python}"
MANIFEST="${MANIFEST:?set MANIFEST to a completed three-cell TSV}"
OUT_DIR="${OUT_DIR:?set OUT_DIR to a new training output directory}"
GPU="${GPU:-0}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" train_multicell_fusion_compartment_range.py \
  --manifest "${MANIFEST}" --output-dir "${OUT_DIR}" \
  --train-chroms chr1,chr2,chr3,chr4,chr5,chr7,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22 \
  --val-chroms chr6 --input-mode bp --range-size 1650000 --bin-size 2200 \
  --stride 550000 --supervised-center-size 1100000 \
  --bp-stem-type conv_pool --bp-downsample-strides 5,5,4,11,2 \
  --bp-stem-kernels 15,9,7,5,3 --fusion-method simple_concat \
  --loss-type mse --sign-loss-weight 0 --track-noise-std 0 \
  --d-model 128 --depth 3 --num-heads 4 --dropout 0.12 \
  --decoder-layers 1 --decoder-kernel-size 3 --decoder-type conv \
  --lr 0.0001 --weight-decay 0.03 --no-encoder-checkpoint \
  --no-validate-ranges --no-per-range-zscore --batch-size 4 --num-workers 10 \
  --seed 43 --epochs 80 --eval-every 3 --early-stopping-patience 8 \
  --target-threshold 0 "$@"
