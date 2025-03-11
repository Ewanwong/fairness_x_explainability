#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0

python -m saliency_evaluation.create_pointing_game_examples \
    --dataset_name="agvidit1/hateXplain_processed_dataset" \
    --model_dir="models/conventional_distilbert_hatexplain" \
    --load_pointing_game_examples_path='pointing_game_examples/distilbert_hatexplain.json' \
    --save_pointing_game_examples_path='pointing_game_examples/distilbert_hatexplain.json' \
    --num_labels=2 \
    --batch_size=16 \
    --max_length=25 \
    --num_examples=-1 \
    --seed=42 \