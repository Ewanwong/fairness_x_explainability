#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0

python -m saliency_generation.gen_explanations \
    --dataset_name="agvidit1/hateXplain_processed_dataset" \
    --model_dir="models/bcos_distilbert_hatexplain" \
    --num_labels=2 \
    --batch_size=16 \
    --max_length=512 \
    --num_examples=-1 \
    --methods "Bcos" \
    --output_dir="results/perturbation_bcos_hatexplain" \
    --only_predicted_classes \
    --baseline="pad" \
    --seed=42 \