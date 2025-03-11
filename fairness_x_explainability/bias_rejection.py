import argparse
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
import os
import random
from tqdm import tqdm
import scipy.stats
from utils.utils import compute_reliance_score, compute_reliance_score_by_class_comparison, compute_metrics
from utils.utils import EXPLANATION_METHODS, BIAS_TYPES

RELIANCE_KEYS = ["normalized", "normalized_by_class_comparison"]
# RELIANCE_KEYS = ["raw", "normalized", "raw_by_class_comparison", "normalized_by_class_comparison"]

def find_highest_reliances(reliance_scores_group1, reliance_scores_group2, num_highest_reliances, rejection_direction="undirected"):
    # return the indexes of the num_highest_reliances highest reliance scores by magnitute
    labeled_reliances_scores_group1 = [('group1', i, x) for i, x in enumerate(reliance_scores_group1)]
    labeled_reliances_scores_group2 = [('group2', i, x) for i, x in enumerate(reliance_scores_group2)]
    all_reliances = labeled_reliances_scores_group1 + labeled_reliances_scores_group2
    if rejection_direction == "undirected":
        all_reliances_sort = sorted(
            all_reliances, key=lambda t: abs(t[2]), reverse=True
        )
    elif rejection_direction == "min":
        all_reliances_sort = sorted(
            all_reliances, key=lambda t: t[2]
        )
    elif rejection_direction == "max":
        all_reliances_sort = sorted(
            all_reliances, key=lambda t: t[2], reverse=True
        )
    else:
        raise ValueError("Rejection direction not recognized")
    top_k = all_reliances_sort[:num_highest_reliances]
    indexes_group1 = [x[1] for x in top_k if x[0] == 'group1']
    indexes_group2 = [x[1] for x in top_k if x[0] == 'group2']
    return indexes_group1, indexes_group2

def compute_fairness_after_rejection(
        indexes_group1, 
        indexes_group2,
        predictions_group1,
        predictions_group2,
        labels_group1,
        labels_group2,
        group1_to_group2_confidence_diff_group1: dict,
        group1_to_group2_confidence_diff_group2: dict,
        group1="black",
        group2="white",
        num_classes=2,
    ):

    fairness_results = {}

    predictions_after_rejection_group1 = [pred for i, pred in enumerate(predictions_group1) if i not in indexes_group1]
    predictions_after_rejection_group2 = [pred for i, pred in enumerate(predictions_group2) if i not in indexes_group2]
    labels_after_rejection_group1 = [label for i, label in enumerate(labels_group1) if i not in indexes_group1]
    labels_after_rejection_group2 = [label for i, label in enumerate(labels_group2) if i not in indexes_group2]

    metrics_after_rejection_group1 = compute_metrics(labels_after_rejection_group1, predictions_after_rejection_group1, num_classes)
    metrics_after_rejection_group2 = compute_metrics(labels_after_rejection_group2, predictions_after_rejection_group2, num_classes)
    metrics_after_rejection_overall = compute_metrics(
        labels_after_rejection_group1 + labels_after_rejection_group2,
        predictions_after_rejection_group1 + predictions_after_rejection_group2,
        num_classes,
    )

    # overall metrics
    fairness_results[f"num_examples"] = len(predictions_after_rejection_group1) + len(predictions_after_rejection_group2)
    fairness_results[f"num_rejections"] = len(indexes_group1) + len(indexes_group2)
    fairness_results[f"accuracy"] = metrics_after_rejection_overall["accuracy"]
    fairness_results[f"f1"] = metrics_after_rejection_overall["f1"]
    for i in range(num_classes):
        for metric in ["tpr", "fpr", "tnr", "fnr"]:
            fairness_results[f"class_{i}_{metric}"] = metrics_after_rejection_overall[f"class_{i}"][metric]

    # group1 metrics
    fairness_results[f"{group1}_num_examples"] = len(predictions_after_rejection_group1)
    fairness_results[f"{group1}_num_rejections"] = len(indexes_group1)
    fairness_results[f"{group1}_accuracy"] = metrics_after_rejection_group1["accuracy"]
    fairness_results[f"{group1}_f1"] = metrics_after_rejection_group1["f1"]
    for i in range(num_classes):
        for metric in ["tpr", "fpr", "tnr", "fnr"]:
            fairness_results[f"{group1}_class_{i}_{metric}"] = metrics_after_rejection_group1[f"class_{i}"][metric]

    # group2 metrics
    fairness_results[f"{group2}_num_examples"] = len(predictions_after_rejection_group2)
    fairness_results[f"{group2}_num_rejections"] = len(indexes_group2)
    fairness_results[f"{group2}_accuracy"] = metrics_after_rejection_group2["accuracy"]
    fairness_results[f"{group2}_f1"] = metrics_after_rejection_group2["f1"]
    for i in range(num_classes):
        for metric in ["tpr", "fpr", "tnr", "fnr"]:
            fairness_results[f"{group2}_class_{i}_{metric}"] = metrics_after_rejection_group2[f"class_{i}"][metric]

    # num rejections by group and predicted class
    for i in range(num_classes):
        fairness_results[f"{group1}_num_rejections_class_{i}"] = len([pred for idx, pred in enumerate(predictions_group1) if idx in indexes_group1 and pred == i])
        fairness_results[f"{group2}_num_rejections_class_{i}"] = len([pred for idx, pred in enumerate(predictions_group2) if idx in indexes_group2 and pred == i])

    # group fairness: difference in metrics between group1 and group2
    for metric in ["accuracy", "f1"]:
        fairness_results[f"{group1}_to_{group2}_{metric}_diff"] = metrics_after_rejection_group2[metric] - metrics_after_rejection_group1[metric]
        fairness_results[f"{group2}_to_{group1}_{metric}_diff"] = metrics_after_rejection_group1[metric] - metrics_after_rejection_group2[metric]
    
    for i in range(num_classes):
        for metric in ["tpr", "fpr", "tnr", "fnr"]:
            fairness_results[f"{group1}_to_{group2}_class_{i}_{metric}_diff"] = metrics_after_rejection_group2[f"class_{i}"][metric] - metrics_after_rejection_group1[f"class_{i}"][metric]
            fairness_results[f"{group2}_to_{group1}_class_{i}_{metric}_diff"] = metrics_after_rejection_group1[f"class_{i}"][metric] - metrics_after_rejection_group2[f"class_{i}"][metric]
    
    # individual fairness: difference in metrics between group1 and group2 for each class and predicted class
    if group1_to_group2_confidence_diff_group1 is not None and group1_to_group2_confidence_diff_group2 is not None:
        group1_to_group2_confidence_diff_group1_predicted_class = group1_to_group2_confidence_diff_group1["predicted_class"]
        group1_to_group2_confidence_diff_group2_predicted_class = group1_to_group2_confidence_diff_group2["predicted_class"]

        group1_to_group2_confidence_diff_group1_predicted_class_after_rejection = [x for i, x in enumerate(group1_to_group2_confidence_diff_group1_predicted_class) if i not in indexes_group1]
        group1_to_group2_confidence_diff_group2_predicted_class_after_rejection = [x for i, x in enumerate(group1_to_group2_confidence_diff_group2_predicted_class) if i not in indexes_group2]
        fairness_results[f"counterfactually_augmented_{group1}_to_{group2}_predicted_class_confidence_diff_avg"] = np.mean(group1_to_group2_confidence_diff_group1_predicted_class_after_rejection + group1_to_group2_confidence_diff_group2_predicted_class_after_rejection)
        fairness_results[f"counterfactually_augmented_{group2}_to_{group1}_predicted_class_confidence_diff_avg"] = -fairness_results[f"counterfactually_augmented_{group1}_to_{group2}_predicted_class_confidence_diff_avg"]
        fairness_results[f"counterfactually_augmented_predicted_class_confidence_diff_abs_avg"] = np.mean(np.abs(group1_to_group2_confidence_diff_group2_predicted_class_after_rejection + group1_to_group2_confidence_diff_group1_predicted_class_after_rejection))

        for i in range(num_classes):
            group1_to_group2_confidence_diff_group1_class_i = group1_to_group2_confidence_diff_group1[f"class_{i}"]
            group1_to_group2_confidence_diff_group2_class_i = group1_to_group2_confidence_diff_group2[f"class_{i}"]

            group1_to_group2_confidence_diff_group1_class_i_after_rejection = [x for i, x in enumerate(group1_to_group2_confidence_diff_group1_class_i) if i not in indexes_group1]
            group1_to_group2_confidence_diff_group2_class_i_after_rejection = [x for i, x in enumerate(group1_to_group2_confidence_diff_group2_class_i) if i not in indexes_group2]
            fairness_results[f"counterfactually_augmented_{group1}_to_{group2}_class_{i}_confidence_diff_avg"] = np.mean(group1_to_group2_confidence_diff_group2_class_i_after_rejection + group1_to_group2_confidence_diff_group1_class_i_after_rejection)
            fairness_results[f"counterfactually_augmented_{group2}_to_{group1}_class_{i}_confidence_diff_avg"] = -fairness_results[f"counterfactually_augmented_{group1}_to_{group2}_class_{i}_confidence_diff_avg"]
            fairness_results[f"counterfactually_augmented_class_{i}_confidence_diff_abs_avg"] = np.mean(np.abs(group1_to_group2_confidence_diff_group2_class_i_after_rejection + group1_to_group2_confidence_diff_group1_class_i_after_rejection))

    return fairness_results


def main(args):

    if len(BIAS_TYPES[args.bias_type]) != 2:
        raise ValueError("Only binary bias types are supported")
    
    # identify model type to extract fairness results
    if "baseline" in args.explanation_dir:
        model_type = "baseline"
    elif "bcos" in args.explanation_dir:
        model_type = "bcos"
    else:
        raise ValueError("Model type not recognized") 

    if args.methods is not None:
        methods = args.methods.strip().split(",")
    else:
        methods = EXPLANATION_METHODS

    rejection_ratios = [float(ratio) for ratio in args.rejection_ratios.strip().split(",") if float(ratio) > 0 and float(ratio) < 1]
    
    # load fairness results
    fairness_file = os.path.join(args.explanation_dir, f"fairness_{model_type}_{args.bias_type}_{args.split}_results.json")
    if not os.path.exists(fairness_file):
        raise ValueError(f"File {fairness_file} does not exist")
        
    with open(fairness_file) as f:
        fairness_results = json.load(f)   

    # load predictions, labels, confidence_differences
    predictions = {group: fairness_results[f"{group}_predictions"] for group in BIAS_TYPES[args.bias_type]}
    labels = {group: fairness_results[f"{group}_labels"] for group in BIAS_TYPES[args.bias_type]}
    if args.counterfactual:
        group1_to_group2_confidence_diff = {group: {"predicted_class": list(fairness_results[f"{group}_counterfactual_{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_predicted_class_confidence_diff_list"].values())} for group in BIAS_TYPES[args.bias_type]}
        for group in BIAS_TYPES[args.bias_type]:
            group1_to_group2_confidence_diff[group].update({f"class_{i}": list(fairness_results[f"{group}_counterfactual_{BIAS_TYPES[args.bias_type][1]}_to_{BIAS_TYPES[args.bias_type][0]}_class_{i}_confidence_diff_list"].values()) for i in range(args.num_labels)})
    else:
        group1_to_group2_confidence_diff = {group: None for group in BIAS_TYPES[args.bias_type]}

    orig_fairness = compute_fairness_after_rejection(
        [],
        [],
        predictions[BIAS_TYPES[args.bias_type][0]],
        predictions[BIAS_TYPES[args.bias_type][1]],
        labels[BIAS_TYPES[args.bias_type][0]],
        labels[BIAS_TYPES[args.bias_type][1]],
        group1_to_group2_confidence_diff[BIAS_TYPES[args.bias_type][0]],
        group1_to_group2_confidence_diff[BIAS_TYPES[args.bias_type][1]],
        BIAS_TYPES[args.bias_type][0],
        BIAS_TYPES[args.bias_type][1],
        args.num_labels,
    )

    for method in methods:
        fairness_after_rejection = {}

        attribution_file = os.path.join(args.explanation_dir, f"{method}_{BIAS_TYPES[args.bias_type][0]}_{args.split}_sensitive_attribution.json")
        if not os.path.exists(attribution_file):
            #print(f"File {sensitive_attribution_file} does not exist. Skipping...")
            continue
        
        print(f"Computing fairness after rejection for {method}...")
        with open(attribution_file) as f:
            attribution_data = json.load(f)

        # mean or L2
        aggregations = list(attribution_data.keys())
        aggregations = [aggr for aggr in aggregations if "predicted_class" not in aggr and "true_label" not in aggr]
    
        # not use class comparison for attention methods
        reliance_keys = RELIANCE_KEYS
        if "Attention" in method:
            reliance_keys = [reliance for reliance in reliance_keys if "by_class_comparison" not in reliance]

        reliances = {aggregation: {group: {} for group in BIAS_TYPES[args.bias_type]} for aggregation in aggregations}

        for group in BIAS_TYPES[args.bias_type]:           
            
            attribution_file = os.path.join(args.explanation_dir, f"{method}_{group}_{args.split}_sensitive_attribution.json")
            if not os.path.exists(attribution_file):
                #print(f"File {sensitive_attribution_file} does not exist. Skipping...")
                continue

            with open(attribution_file) as f:
                attribution_data = json.load(f)

            for aggregation in aggregations:
                attribution_results = attribution_data[aggregation]
                # make sure the attribution and fairness files have the same examples
                assert list(attribution_data["predicted_classes"].values()) == predictions[group]

                fairness_after_rejection[aggregation] = {ratio: {} for ratio in rejection_ratios}

                # extract reliance scores
                predicted_class_sensitive_attribution_results = [attribution["predicted_class"]["sensitive_attribution"] for attribution in list(attribution_results.values())]
                predicted_class_total_attribution_results = [attribution["predicted_class"]["total_attribution"] for attribution in list(attribution_results.values())]
                if "raw" in reliance_keys:
                    raw_reliance_scores_predicted_class = [compute_reliance_score(predicted_class_sensitive_attribution_result, predicted_class_total_attribution_result, method="raw") for predicted_class_sensitive_attribution_result, predicted_class_total_attribution_result in zip(predicted_class_sensitive_attribution_results, predicted_class_total_attribution_results)]
                    reliances[aggregation][group]["raw"] = raw_reliance_scores_predicted_class
                if "normalized" in reliance_keys:
                    normalized_reliance_scores_predicted_class = [compute_reliance_score(predicted_class_sensitive_attribution_result, predicted_class_total_attribution_result, method="normalize") for predicted_class_sensitive_attribution_result, predicted_class_total_attribution_result in zip(predicted_class_sensitive_attribution_results, predicted_class_total_attribution_results)]
                    reliances[aggregation][group]["normalized"] = normalized_reliance_scores_predicted_class

                # if class comparison is used, extract reliance scores for each class
                if "normalized_by_class_comparison" in reliance_keys or "raw_by_class_comparison" in reliance_keys:
                    total_attribution_results_all_classes = []
                    sensitive_attribution_results_all_classes = []
                    
                    for target_class in range(args.num_labels):
                        class_sensitive_attribution_results = [attribution[f"class_{target_class}"]["sensitive_attribution"] for attribution in list(attribution_results.values())]
                        class_total_attribution_results = [attribution[f"class_{target_class}"]["total_attribution"] for attribution in list(attribution_results.values())]
                        sensitive_attribution_results_all_classes.append(class_sensitive_attribution_results)
                        total_attribution_results_all_classes.append(class_total_attribution_results)
                    predicted_classes = predictions[group]
                    raw_reliance_scores_by_class_comparison_predicted_class = []
                    normalized_reliance_scores_by_class_comparison_predicted_class = []
                    for i in range(len(predicted_classes)):
                        predicted_class = predicted_classes[i]
                        predicted_class_sensitive_attribution_result = predicted_class_sensitive_attribution_results[i]
                        predicted_class_total_attribution_result = predicted_class_total_attribution_results[i]
                        other_class_sensitive_attribution_results = [sensitive_attribution_results_all_classes[j][i] for j in range(len(sensitive_attribution_results_all_classes)) if j != predicted_class]
                        other_class_total_attribution_results = [total_attribution_results_all_classes[j][i] for j in range(len(total_attribution_results_all_classes)) if j != predicted_class]
                        raw_reliance_score_by_class_comparison_predicted_class = compute_reliance_score_by_class_comparison(predicted_class_sensitive_attribution_result, predicted_class_total_attribution_result, other_class_sensitive_attribution_results, other_class_total_attribution_results, method="raw")
                        normalized_reliance_score_by_class_comparison_predicted_class = compute_reliance_score_by_class_comparison(predicted_class_sensitive_attribution_result, predicted_class_total_attribution_result, other_class_sensitive_attribution_results, other_class_total_attribution_results, method="normalize")
                        raw_reliance_scores_by_class_comparison_predicted_class.append(raw_reliance_score_by_class_comparison_predicted_class)
                        normalized_reliance_scores_by_class_comparison_predicted_class.append(normalized_reliance_score_by_class_comparison_predicted_class)
                    if "raw_by_class_comparison" in reliance_keys:
                        reliances[aggregation][group]["raw_by_class_comparison"] = raw_reliance_scores_by_class_comparison_predicted_class
                    if "normalized_by_class_comparison" in reliance_keys:
                        reliances[aggregation][group]["normalized_by_class_comparison"] = normalized_reliance_scores_by_class_comparison_predicted_class
                
        # compute metrics after rejection
        num_all_examples = len(predictions[BIAS_TYPES[args.bias_type][0]]) + len(predictions[BIAS_TYPES[args.bias_type][1]])
        for aggregation in aggregations:
            for ratio in rejection_ratios:
                for reliance_key in reliance_keys:
                    indexes_group1, indexes_group2 = find_highest_reliances(reliances[aggregation][BIAS_TYPES[args.bias_type][0]][reliance_key], reliances[aggregation][BIAS_TYPES[args.bias_type][1]][reliance_key], int(num_all_examples * ratio), rejection_direction=args.rejection_direction)
                    fairness_after_rejection[aggregation][ratio][reliance_key] = compute_fairness_after_rejection(
                        indexes_group1,
                        indexes_group2,
                        predictions[BIAS_TYPES[args.bias_type][0]],
                        predictions[BIAS_TYPES[args.bias_type][1]],
                        labels[BIAS_TYPES[args.bias_type][0]],
                        labels[BIAS_TYPES[args.bias_type][1]],
                        group1_to_group2_confidence_diff[BIAS_TYPES[args.bias_type][0]],
                        group1_to_group2_confidence_diff[BIAS_TYPES[args.bias_type][1]],
                        BIAS_TYPES[args.bias_type][0],
                        BIAS_TYPES[args.bias_type][1],
                        args.num_labels,
                    )

            fairness_after_rejection[aggregation]["orig"] = orig_fairness

        # save fairness results
        fairness_after_rejection_file = os.path.join(args.explanation_dir, f"fairness_after_rejection_{model_type}_{args.bias_type}_{args.split}_{method}_{args.rejection_direction}_results.json")
        with open(fairness_after_rejection_file, "w") as f:
            json.dump(fairness_after_rejection, f, indent=4)
        
        print(f"Fairness after rejection for {method} saved to {fairness_after_rejection_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BERT Attribution with Captum')

    parser.add_argument('--explanation_dir', type=str, default='baseline_saliency_results/all_methods_1000_examples_512', help='Path to the saliency data')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use (e.g., train, test)')
    parser.add_argument('--num_labels', type=int, default=2, help='Number of labels in the classification')
    parser.add_argument('--methods', type=str, default=None, help='List of attribution methods to use separated by commas')
    parser.add_argument('--bias_type', type=str, default='race', help='Bias type to explain')
    parser.add_argument('--counterfactual', action='store_true', help='Apply counterfactual perturbation')
    parser.add_argument('--rejection_ratios', type=str, default='0.1,0.2,0.3,0.4,0.5', help='List of rejection ratios separated by commas')
    parser.add_argument('--rejection_direction', type=str, default='undirected', choices=['undirected', 'min', 'max'], help='Direction of rejection')
    args = parser.parse_args()
    main(args)