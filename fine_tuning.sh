#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0

python -m model_finetuning \
    --model_name_or_path "distilbert/distilbert-base-uncased" \
    --dataset_name="lighteval/civil_comments_helm" \
    --num_labels=2 \
    --output_dir "/scratch/yifwang/fairness_x_explainability/models/baseline_distilbert_civil_seed_42" \
    --batch_size=32 \
    --max_seq_length=512 \
    --learning_rate=2e-5 \
    --warmup_steps_or_ratio=0.1 \
    --num_train_epochs=3 \
    --early_stopping_patience=-1 \
    --eval_steps=1000 \
    --save_steps=1000 \
    --split_ratio="1, 1" \
    --seed=42 \

python -m model_finetuning \
    --model_name_or_path "bert-base-uncased" \
    --dataset_name="lighteval/civil_comments_helm" \
    --num_labels=2 \
    --output_dir "/scratch/yifwang/fairness_x_explainability/models/baseline_bert_civil_seed_42" \
    --batch_size=32 \
    --max_seq_length=512 \
    --learning_rate=2e-5 \
    --warmup_steps_or_ratio=0.1 \
    --num_train_epochs=3 \
    --early_stopping_patience=-1 \
    --eval_steps=1000 \
    --save_steps=1000 \
    --split_ratio="1, 1" \
    --seed=42 \

python -m model_finetuning \
    --model_name_or_path "FacebookAI/roberta-base" \
    --dataset_name="lighteval/civil_comments_helm" \
    --num_labels=2 \
    --output_dir "/scratch/yifwang/fairness_x_explainability/models/baseline_roberta_civil_seed_42" \
    --batch_size=32 \
    --max_seq_length=512 \
    --learning_rate=2e-5 \
    --warmup_steps_or_ratio=0.1 \
    --num_train_epochs=3 \
    --early_stopping_patience=-1 \
    --eval_steps=1000 \
    --save_steps=1000 \
    --split_ratio="1, 1" \
    --seed=42 \

python -m model_finetuning \
    --model_name_or_path "distilbert/distilbert-base-uncased" \
    --dataset_name="stanfordnlp/sst2" \
    --num_labels=2 \
    --output_dir "/scratch/yifwang/fairness_x_explainability/models/baseline_distilbert_sst2_seed_42" \
    --batch_size=16 \
    --max_seq_length=512 \
    --learning_rate=2e-5 \
    --warmup_steps_or_ratio=0.1 \
    --num_train_epochs=5 \
    --early_stopping_patience=-1 \
    --eval_steps=1000 \
    --save_steps=1000 \
    --split_ratio="4, 3, 3" \
    --seed=42 \

python -m model_finetuning \
    --model_name_or_path "bert-base-uncased" \
    --dataset_name="stanfordnlp/sst2" \
    --num_labels=2 \
    --output_dir "/scratch/yifwang/fairness_x_explainability/models/baseline_bert_sst2_seed_42" \
    --batch_size=16 \
    --max_seq_length=512 \
    --learning_rate=2e-5 \
    --warmup_steps_or_ratio=0.1 \
    --num_train_epochs=5 \
    --early_stopping_patience=-1 \
    --eval_steps=1000 \
    --save_steps=1000 \
    --split_ratio="4, 3, 3" \
    --seed=42 \

python -m model_finetuning \
    --model_name_or_path "FacebookAI/roberta-base" \
    --dataset_name="stanfordnlp/sst2" \
    --num_labels=2 \
    --output_dir "/scratch/yifwang/fairness_x_explainability/models/baseline_roberta_sst2_seed_42" \
    --batch_size=16 \
    --max_seq_length=512 \
    --learning_rate=2e-5 \
    --warmup_steps_or_ratio=0.1 \
    --num_train_epochs=5 \
    --early_stopping_patience=-1 \
    --eval_steps=1000 \
    --save_steps=1000 \
    --split_ratio="4, 3, 3" \
    --seed=42 \

python -m model_finetuning \
    --model_name_or_path "distilbert/distilbert-base-uncased" \
    --dataset_name="LabHC/bias_in_bios" \
    --num_labels=28 \
    --output_dir "/scratch/yifwang/fairness_x_explainability/models/baseline_distilbert_bios_seed_42" \
    --batch_size=32 \
    --max_seq_length=512 \
    --learning_rate=2e-5 \
    --warmup_steps_or_ratio=0.1 \
    --num_train_epochs=3 \
    --early_stopping_patience=-1 \
    --eval_steps=1000 \
    --save_steps=1000 \
    --seed=42 \

python -m model_finetuning \
    --model_name_or_path "bert-base-uncased" \
    --dataset_name="LabHC/bias_in_bios" \
    --num_labels=28 \
    --output_dir "/scratch/yifwang/fairness_x_explainability/models/baseline_bert_bios_seed_42" \
    --batch_size=32 \
    --max_seq_length=512 \
    --learning_rate=2e-5 \
    --warmup_steps_or_ratio=0.1 \
    --num_train_epochs=3 \
    --early_stopping_patience=-1 \
    --eval_steps=1000 \
    --save_steps=1000 \
    --seed=42 \

python -m model_finetuning \
    --model_name_or_path "FacebookAI/roberta-base" \
    --dataset_name="LabHC/bias_in_bios" \
    --num_labels=32 \
    --output_dir "/scratch/yifwang/fairness_x_explainability/models/baseline_roberta_bios_seed_42" \
    --batch_size=32 \
    --max_seq_length=512 \
    --learning_rate=2e-5 \
    --warmup_steps_or_ratio=0.1 \
    --num_train_epochs=3 \
    --early_stopping_patience=-1 \
    --eval_steps=1000 \
    --save_steps=1000 \
    --seed=42 \