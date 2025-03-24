#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0


python -m fairness_x_explainability.bias_correlation \
    --explanation_dir="/scratch/yifwang/fairness_x_explainability/results/baseline_bert_civil" \
    --split="test" \
    --num_labels=2 \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion" \
    --normalization_factor="norm" \

python -m fairness_x_explainability.bias_correlation \
    --explanation_dir="/scratch/yifwang/fairness_x_explainability/results/bcos_bert_civil" \
    --split="test" \
    --num_labels=2 \
    --bias_type="race" \
    --methods="Bcos" \
    --normalization_factor="norm" \

