#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0

python -m model_finetuning \
    --model_name_or_path "distilbert/distilbert-base-uncased" \
    --dataset_name="agvidit1/hateXplain_processed_dataset" \
    --num_labels=2 \
    --output_dir "models/bcos_distilbert_hatexplain" \
    --batch_size=16 \
    --max_seq_length=512 \
    --learning_rate=2e-5 \
    --warmup_steps_or_ratio=0.1 \
    --num_train_epochs=5 \
    --early_stopping_patience=-1 \
    --seed=42 \
    --b 1.5 \
    --bcos \
    --bce \
    --seed=42 \