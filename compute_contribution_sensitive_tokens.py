## conclusion:
## white/black words should be given positive attribution for positive predictions
## and negative attribution for negative predictions in both baseline and bcos models
## (in >95%/99% of the cases)

import argparse
import json
import logging
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from transformers import (AutoTokenizer, AutoConfig, AdamW,
                          get_linear_schedule_with_warmup)
from bcos_lm.models.modeling_bert import BertForSequenceClassification
from bcos_lm.models.modeling_roberta import RobertaForSequenceClassification
from bcos_lm.models.modeling_distilbert import DistilBertForSequenceClassification
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import random
import os
from tqdm import tqdm

SHOULD_CONTAIN = {"white": ["white", "caucasian", "europe"], 
                  "black": ["black", "africa"],
                  "male": [],
                  "female": []}

SHOULD_NOT_CONTAIN = {"white": [],
                      "black": ["nigg", "negro", "niger"],
                      "male": [],
                      "female": []}

# TODO: a better filter function
def filter_text(text, should_contain, should_not_contain):
    contain_flag = False
    for word in should_contain:
        if word in text.lower():
            contain_flag = True
            break
    if not contain_flag:
        return False
    for word in should_not_contain:
        if word in text.lower():
            return False
    return True

if __name__ == "__main__":
    """
    model_dir = "/scratch/yifwang/fairness_x_explainability/models/baseline_bert_civil_seed_42"
    black_dataset = "results/baseline_bert_civil/Saliency_black_test_explanations.json"
    white_dataset = "results/baseline_bert_civil/Saliency_white_test_explanations.json"
    aggregation = "Saliency_mean"
    """
    model_dir = "/scratch/yifwang/fairness_x_explainability/models/bcos_bert_civil_seed_42"
    black_dataset = "results/bcos_bert_civil/Bcos_black_test_explanations.json"
    white_dataset = "results/bcos_bert_civil/Bcos_white_test_explanations.json"
    aggregation = "Bcos_absolute_ixg_mean"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load pre-trained BERT model and tokenizer
    if "distilbert" in model_dir.lower():
        Model = DistilBertForSequenceClassification
    elif "roberta" in model_dir.lower():
        Model = RobertaForSequenceClassification
    elif "bert" in model_dir.lower():
        Model = BertForSequenceClassification
    config = AutoConfig.from_pretrained(model_dir, num_labels=2)
    #config.bcos = args.bcos
    #config.b = args.b

    config.output_attentions = True
    config.num_labels = 2
    #print(config)
    model = Model.load_from_pretrained(model_dir, config=config, output_attentions=True)
    model.eval()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    black_positive_examples = []
    black_negative_examples = []
    white_positive_examples = []
    white_negative_examples = []
    black_positive_confidences = []
    black_negative_confidences = []
    white_positive_confidences = []
    white_negative_confidences = []
    # extract examples
    with open(black_dataset, "r") as f:
        black_data = json.load(f)[aggregation]
    
    for example in black_data:
        text = example[0]['text']
        if filter_text(text.lower(), SHOULD_CONTAIN["black"], SHOULD_NOT_CONTAIN["black"]):
            masked_text = text.lower().replace("african", "[MASK]").replace("africa", "[MASK]").replace("blacks", "[MASK]").replace("black", "[MASK]")
            if example[0]['predicted_class'] == 1:
                black_positive_examples.append(masked_text)
                black_positive_confidences.append(example[0]['predicted_class_confidence'])
            else:
                black_negative_examples.append(masked_text)
                black_negative_confidences.append(example[0]['predicted_class_confidence'])
    print(len(black_positive_examples), len(black_negative_examples))

    with open(white_dataset, "r") as f:
        white_data = json.load(f)[aggregation]

    for example in white_data:
        text = example[0]['text']
        if filter_text(text.lower(), SHOULD_CONTAIN["white"], SHOULD_NOT_CONTAIN["white"]):
            masked_text = text.lower().replace("european", "[MASK]").replace("europe", "[MASK]").replace("caucasian", "[MASK]").replace("whites", "[MASK]").replace("white", "[MASK]")
            if example[0]['predicted_class'] == 1:    
                white_positive_examples.append(masked_text)
                white_positive_confidences.append(example[0]['predicted_class_confidence'])
            else:
                white_negative_examples.append(masked_text)
                white_negative_confidences.append(example[0]['predicted_class_confidence'])
    print(len(white_positive_examples), len(white_negative_examples))
    masked_black_positive_confidences = []
    masked_black_negative_confidences = []
    masked_white_positive_confidences = []
    masked_white_negative_confidences = []
    for example in black_positive_examples:
        inputs = tokenizer(example, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        confidence = torch.nn.functional.softmax(outputs.logits, dim=1)
        masked_black_positive_confidences.append(confidence[0][1].item())

    for example in black_negative_examples:
        inputs = tokenizer(example, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        confidence = torch.nn.functional.softmax(outputs.logits, dim=1)
        masked_black_negative_confidences.append(confidence[0][0].item())
        
    for example in white_positive_examples:
        inputs = tokenizer(example, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        confidence = torch.nn.functional.softmax(outputs.logits, dim=1)
        masked_white_positive_confidences.append(confidence[0][1].item())
        
    for example in white_negative_examples:
        inputs = tokenizer(example, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        confidence = torch.nn.functional.softmax(outputs.logits, dim=1)
        masked_white_negative_confidences.append(confidence[0][0].item())

    # compute the difference
    black_positive_confidences = np.array(black_positive_confidences)
    black_negative_confidences = np.array(black_negative_confidences)
    white_positive_confidences = np.array(white_positive_confidences)
    white_negative_confidences = np.array(white_negative_confidences)
    masked_black_positive_confidences = np.array(masked_black_positive_confidences)
    masked_black_negative_confidences = np.array(masked_black_negative_confidences)
    masked_white_positive_confidences = np.array(masked_white_positive_confidences)
    masked_white_negative_confidences = np.array(masked_white_negative_confidences)

    black_positive_diff = masked_black_positive_confidences - black_positive_confidences
    black_negative_diff = masked_black_negative_confidences - black_negative_confidences
    white_positive_diff = masked_white_positive_confidences - white_positive_confidences
    white_negative_diff = masked_white_negative_confidences - white_negative_confidences

    print("Black positive diff: ", black_positive_diff.mean()) # baseline/bcos -0.269 / -0.349
    print("Black negative diff: ", black_negative_diff.mean()) # baseline/bcos 0.110 / 0.089
    print("White positive diff: ", white_positive_diff.mean()) # baseline/bcos -0.150 / -0.225
    print("White negative diff: ", white_negative_diff.mean()) # baseline/bcos 0.0741 / 0.066

    black_positive_diff_positive_ratio = (black_positive_diff > 0).sum() / len(black_positive_diff)
    black_negative_diff_positive_ratio = (black_negative_diff > 0).sum() / len(black_negative_diff)
    white_positive_diff_positive_ratio = (white_positive_diff > 0).sum() / len(white_positive_diff)
    white_negative_diff_positive_ratio = (white_negative_diff > 0).sum() / len(white_negative_diff)

    print("Black positive diff positive ratio: ", black_positive_diff_positive_ratio) # baseline/bcos 0.009 / 0.001
    print("Black negative diff positive ratio: ", black_negative_diff_positive_ratio) # baseline/bcos 0.973 / 0.998
    print("White positive diff positive ratio: ", white_positive_diff_positive_ratio) # baseline/bcos 0.061 / 0.003
    print("White negative diff positive ratio: ", white_negative_diff_positive_ratio) # baseline/bcos 0.929 / 0.995