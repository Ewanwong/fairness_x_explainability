#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0

python -m saliency_generation.gen_explanations \
    --dataset_name="lighteval/civil_comments_helm" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/models/bcos_bert_civil_seed_42" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=-1 \
    --methods "Bcos" \
    --output_dir="results/bcos_bert_civil" \
    --baseline="pad" \
    --split_ratio="1, 1" \
    --split="test" \
    --bias_type="race" \
    --seed=42 \
    --counterfactual \
    #--only_predicted_classes \


python -m saliency_generation.gen_explanations \
    --dataset_name="lighteval/civil_comments_helm" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/models/baseline_bert_civil_seed_42" \
    --num_labels=2 \
    --batch_size=4 \
    --max_length=512 \
    --num_examples=-1 \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion" \
    --output_dir="results/baseline_bert_civil" \
    --baseline="pad" \
    --split_ratio="1, 1" \
    --split="test" \
    --bias_type="race" \
    --seed=42 \
    --counterfactual \
    #--only_predicted_classes \
