#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0

python -m fairness_x_explainability.bias_correlation \
    --explanation_dir="results/bcos_bert_civil" \
    --split="test" \
    --num_labels=2 \
    --bias_type="race" \
    --methods "Bcos" \

python -m fairness_x_explainability.bias_correlation \
    --explanation_dir="results/baseline_bert_civil" \
    --split="test" \
    --num_labels=2 \
    --bias_type="race" \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion" \