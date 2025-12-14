#!/bin/bash
set -e

echo "Running baseline experiments..."

python app/train/train_baseline.py \
  --config configs/experiments/baseline_logreg.yaml

python app/train/train_baseline.py \
  --config configs/experiments/baseline_logreg_v2.yaml


echo "Running ModernBERT frozen experiments..."

python app/train/train.py \
  --config configs/experiments/modernbert_frozen.yaml

python app/train/train.py \
  --config configs/experiments/modernbert_frozen_v2.yaml


echo "Running ModernBERT finetuned experiments..."

python app/train/train.py \
  --config configs/experiments/modernbert_tuned.yaml

python app/train/train.py \
  --config configs/experiments/modernbert_tuned_v2.yaml


echo "All experiments finished successfully."