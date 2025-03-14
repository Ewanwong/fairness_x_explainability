#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0

python -m fairness_x_explainability.sensitive_reliance_by_error_type \
    --explanation_dir="/scratch/yifwang/fairness_x_explainability/results/bcos_bert_civil" \
    --num_labels=2 \
    --split="test" \
    --bias_type="race" \
    --counterfactual \
    --methods "Bcos" \

python -m fairness_x_explainability.sensitive_reliance_by_error_type \
    --explanation_dir="/scratch/yifwang/fairness_x_explainability/results/baseline_bert_civil" \
    --num_labels=2 \
    --split="test" \
    --bias_type="race" \
    --counterfactual \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion" \