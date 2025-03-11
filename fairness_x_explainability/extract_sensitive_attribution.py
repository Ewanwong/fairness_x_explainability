import argparse
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
import os
import random
from tqdm import tqdm
from utils.utils import filter_text
from utils.utils import BIAS_TYPES, EXPLANATION_METHODS, SENSITIVE_TOKENS, SHOULD_CONTAIN, SHOULD_NOT_CONTAIN

# TODO: a better way to extract sensitive token reliance
def extract_sensitive_attributions(explanations, sensitive_tokens):
    results = {}
    for explanations in explanations:
        index = explanations[0]["index"]
        results[index] = {}
        for target_class in range(len(explanations)):           
            attribution_scores = explanations[target_class]["attribution"]
            results[index][f"class_{target_class}"] = {"sensitive_attribution":[], "total_attribution":attribution_scores}
            for attribution_score in attribution_scores:
                for sensitive_token in sensitive_tokens:
                    if sensitive_token in attribution_score[0]:
                        results[index][f"class_{target_class}"]["sensitive_attribution"].append(attribution_score)
                        
        results[index][f"predicted_class"] = results[index][f"class_{explanations[0]['predicted_class']}"].copy()
    return results

def main(args):

    if len(BIAS_TYPES[args.bias_type]) != 2:
        raise ValueError("Only binary bias types are supported")
    
    if args.methods is not None:
        methods = args.methods.strip().split(",")
    else:
        methods = EXPLANATION_METHODS

    for method in methods:
        for group in BIAS_TYPES[args.bias_type]:
            sensitive_tokens = SENSITIVE_TOKENS[group]           
            group_explanation_file = os.path.join(args.explanation_dir, f"{method}_{group}_{args.split}_explanations.json")
            if not os.path.exists(group_explanation_file):
                #print(f"File {group_explanation_file} does not exist. Skipping...")
                continue
            print(f"Extracting sensitive attribution from {group_explanation_file}")
            sensitive_attribution_results = {}
            with open(group_explanation_file) as f:
                orig_data = json.load(f)
            aggregations = list(orig_data.keys())
            
            for aggregation in aggregations:
                sensitive_attribution_results[aggregation] = {}
                orig_explanations = orig_data[aggregation]
                orig_explanations = [explanations for explanations in orig_explanations if filter_text(explanations[0]["text"], SHOULD_CONTAIN[group], SHOULD_NOT_CONTAIN[group])]
                orig_predictions = [explanations[0]["predicted_class"] for explanations in orig_explanations]
                orig_labels = [explanations[0]["true_label"] for explanations in orig_explanations]

                sensitive_attribution_results[aggregation] = extract_sensitive_attributions(orig_explanations, sensitive_tokens)
                explanation_indexes = list(sensitive_attribution_results[aggregations[0]].keys()) 
                if "predicted_classes" not in sensitive_attribution_results:
                    sensitive_attribution_results[f"predicted_classes"] = {index:orig_predictions[i] for i, index in enumerate(explanation_indexes)} 
                if "true_labels" not in sensitive_attribution_results:
                    sensitive_attribution_results[f"true_labels"] = {index:orig_labels[i] for i, index in enumerate(explanation_indexes)}
            assert len(explanation_indexes) == len(orig_predictions)       
            output_file = os.path.join(args.explanation_dir, f"{method}_{group}_{args.split}_sensitive_attribution.json")
            with open(output_file, 'w') as f:
                json.dump(sensitive_attribution_results, f, indent=4)
            print(f"Sensitive attribution results saved to {output_file}")  


            if args.counterfactual:
                counterfactual_group = BIAS_TYPES[args.bias_type][1 - BIAS_TYPES[args.bias_type].index(group)]
                counterfactual_sensitive_tokens = SENSITIVE_TOKENS[counterfactual_group]
                counterfactual_explanation_file = os.path.join(args.explanation_dir, f"{method}_{group}_counterfactual_{args.split}_explanations.json")
                if not os.path.exists(counterfactual_explanation_file):
                    #print(f"File {counterfactual_explanation_file} does not exist. Skipping...")
                    continue
                print(f"Extracting sensitive attribution from {counterfactual_explanation_file}")
                counterfactual_sensitive_attribution_results = {}
                with open(counterfactual_explanation_file) as f:
                    counterfactual_data = json.load(f)
                aggregations = list(counterfactual_data.keys())
                for aggregation in aggregations:
                    counterfactual_sensitive_attribution_results[aggregation] = {}
                    counterfactual_explanations = counterfactual_data[aggregation]
                    counterfactual_explanations = [explanations for explanations in counterfactual_explanations if explanations[0]["index"] in explanation_indexes]
                    counterfactual_predictions = [explanations[0]["predicted_class"] for explanations in counterfactual_explanations]
                    counterfactual_sensitive_attribution_results[aggregation] = extract_sensitive_attributions(counterfactual_explanations, counterfactual_sensitive_tokens)
                    if "predicted_classes" not in counterfactual_sensitive_attribution_results:
                        counterfactual_sensitive_attribution_results[f"predicted_classes"] = {index:counterfactual_predictions[i] for i, index in enumerate(explanation_indexes)}
                    if "true_labels" not in counterfactual_sensitive_attribution_results:
                        counterfactual_sensitive_attribution_results[f"true_labels"] = {index:orig_labels[i] for i, index in enumerate(explanation_indexes)}
                counterfactual_output_file = os.path.join(args.explanation_dir, f"{method}_{group}_counterfactual_{args.split}_sensitive_attribution.json")
                with open(counterfactual_output_file, 'w') as f:
                    json.dump(counterfactual_sensitive_attribution_results, f, indent=4)
                print(f"Sensitive attribution results saved to {counterfactual_output_file}")                                                                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BERT Attribution with Captum')

    parser.add_argument('--explanation_dir', type=str, default='baseline_saliency_results/all_methods_1000_examples_512', help='Path to the saliency data')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use (e.g., train, test)')
    #parser.add_argument('--num_labels', type=int, default=2, help='Number of labels in the classification')
    parser.add_argument('--methods', type=str, default=None, help='List of attribution methods to use separated by commas')
    parser.add_argument('--bias_type', type=str, default='race', help='Bias type to explain')
    parser.add_argument('--counterfactual', action='store_true', help='Apply counterfactual perturbation')

    args = parser.parse_args()
    main(args)