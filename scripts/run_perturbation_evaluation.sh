#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0


python -m saliency_evaluation.perturbation_eval \
    --explanation_dir="results/perturbation_bcos_hatexplain" \
    --model_dir="models/bcos_distilbert_hatexplain" \
    --num_labels=4 \
    --batch_size=16 \
    --max_length=512 \
    --num_examples=-1 \
