#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0

python -m fairness_x_explainability.fairness_eval \
    --explanation_dir="results/bcos_bert_civil" \
    --split="test" \
    --num_labels=2 \
    --bias_type="race" \
    --counterfactual \

python -m fairness_x_explainability.fairness_eval \
    --explanation_dir="results/baseline_bert_civil" \
    --split="test" \
    --num_labels=2 \
    --bias_type="race" \
    --counterfactual \
    