#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0

python -m fairness_x_explainability.extract_sensitive_attribution \
    --explanation_dir="results/baseline_bert_civil" \
    --split="test" \
    --bias_type="race" \
    --counterfactual \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion" \


python -m fairness_x_explainability.extract_sensitive_attribution \
    --explanation_dir="results/bcos_bert_civil" \
    --split="test" \
    --bias_type="race" \
    --counterfactual \
    --methods "Bcos" \
