#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0

python -m fairness_x_explainability.bias_rejection \
    --explanation_dir="/scratch/yifwang/fairness_x_explainability/results/baseline_bert_civil" \
    --split="test" \
    --num_labels=2 \
    --bias_type="race" \
    --counterfactual \
    --rejection_ratios="0.01, 0.05, 0.1, 0.2, 0.5" \
    --rejection_direction="undirected" \
    #-- methods

python -m fairness_x_explainability.bias_rejection \
    --explanation_dir="/scratch/yifwang/fairness_x_explainability/results/baseline_bert_civil" \
    --split="test" \
    --num_labels=2 \
    --bias_type="race" \
    --counterfactual \
    --rejection_ratios="0.01, 0.05, 0.1, 0.2, 0.5" \
    --rejection_direction="min" \

python -m fairness_x_explainability.bias_rejection \
    --explanation_dir="/scratch/yifwang/fairness_x_explainability/results/baseline_bert_civil" \
    --split="test" \
    --num_labels=2 \
    --bias_type="race" \
    --counterfactual \
    --rejection_ratios="0.01, 0.05, 0.1, 0.2, 0.5" \
    --rejection_direction="max" \