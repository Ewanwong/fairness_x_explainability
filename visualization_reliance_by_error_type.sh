#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0

python -m fairness_x_explainability.visualize_reliance_error_type \
    --explanation_dir="/scratch/yifwang/fairness_x_explainability/results/baseline_bert_civil" \
    --split="test" \
    --num_labels=2 \
    --bias_type="race" \
    #-- methods

