import argparse
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
import os
import random
from tqdm import tqdm
from utils.utils import filter_text, compute_metrics
from utils.utils import BIAS_TYPES, EXPLANATION_METHODS, SHOULD_CONTAIN, SHOULD_NOT_CONTAIN


def main(args):
    if "baseline" in args.explanation_dir:
        model_type = "baseline"
    elif "bcos" in args.explanation_dir:
        model_type = "bcos"
    else:
        raise ValueError("Model type not recognized")

    if model_type == "bcos":
        method = "Bcos"
    else:
        method = None
        for m in EXPLANATION_METHODS:
            if os.path.exists(os.path.join(args.explanation_dir, f"{m}_{BIAS_TYPES[args.bias_type][0]}_{args.split}_explanations.json")):
                method = m
                break
        if method is None:
            raise ValueError("No explanation files found")
        
    results = {}
    all_predictions = []
    all_labels = []
    if len(BIAS_TYPES[args.bias_type]) != 2:
        raise ValueError("Only binary bias types are supported")

    for group in BIAS_TYPES[args.bias_type]:
        results[group] = {}
        group_explanation_file = os.path.join(args.explanation_dir, f"{method}_{group}_{args.split}_explanations.json")
        with open(group_explanation_file) as f:
            orig_data = json.load(f)
        aggregations = list(orig_data.keys())
        orig_explanations = orig_data[aggregations[0]]

        # filter out explanations without group mentions or with toxic mentions
        orig_explanations = [explanations for explanations in orig_explanations if filter_text(explanations[0]["text"], SHOULD_CONTAIN[group], SHOULD_NOT_CONTAIN[group])]
        all_orig_predictions = [explanations[0]["predicted_class"] for explanations in orig_explanations]
        all_orig_labels = [explanations[0]["true_label"] for explanations in orig_explanations]
        all_predictions.extend(all_orig_predictions)
        all_labels.extend(all_orig_labels)
        orig_metrics_dict = compute_metrics(all_orig_labels, all_orig_predictions, num_classes=args.num_labels)
        
        # metrics on the original group-specific data
        results[group]["num_examples"] = len(orig_explanations)
        results[group]["accuracy"] =  orig_metrics_dict["accuracy"]
        results[group]["f1"] = orig_metrics_dict["f1"]
        for i in range(args.num_labels):
            for metric in ["tpr", "fpr", "tnr", "fnr"]:
                results[group][f"class_{i}_{metric}"] = orig_metrics_dict[f"class_{i}"][metric]
        results[group]["predictions"] = all_orig_predictions
        results[group]["labels"] = all_orig_labels

        if args.counterfactual:
            counterfactual_group = BIAS_TYPES[args.bias_type][0] if group == BIAS_TYPES[args.bias_type][1] else BIAS_TYPES[args.bias_type][1]
            all_group_idxs = [explanations[0]["index"] for explanations in orig_explanations]
            all_group_class_confidences = {i: [explanations[i]["target_class_confidence"] for explanations in orig_explanations] for i in range(args.num_labels)}
            all_group_predicted_class_confidences = [explanations[0]["predicted_class_confidence"] for explanations in orig_explanations]

            counterfactual_explanation_file = os.path.join(args.explanation_dir, f"{method}_{group}_counterfactual_{args.split}_explanations.json")
            with open(counterfactual_explanation_file) as f:
                counterfactual_data = json.load(f)
            counterfactual_explanations = counterfactual_data[aggregations[0]]

            # choose the same examples as the group explanations
            counterfactual_explanations = [explanations for explanations in counterfactual_explanations if explanations[0]["index"] in all_group_idxs]
            all_counterfactual_group_predictions = [explanations[0]["predicted_class"] for explanations in counterfactual_explanations]
            all_counterfactual_group_labels = [explanations[0]["true_label"] for explanations in counterfactual_explanations]
            counterfactual_group = BIAS_TYPES[args.bias_type][0] if group == BIAS_TYPES[args.bias_type][1] else BIAS_TYPES[args.bias_type][1]
            results[group]["counterfactual_predictions"] = all_counterfactual_group_predictions
            #results[group]["counterfactual_labels"] = all_counterfactual_group_labels

            # compute metrics on the counterfactual group-specific data
            counterfactual_metrics_dict = compute_metrics(all_counterfactual_group_labels, all_counterfactual_group_predictions, num_classes=args.num_labels)

            results[group]["counterfactual_accuracy"] = counterfactual_metrics_dict["accuracy"]
            results[group]["counterfactual_f1"] = counterfactual_metrics_dict["f1"]
            for i in range(args.num_labels):
                for metric in ["tpr", "fpr", "tnr", "fnr"]:
                    results[group][f"counterfactual_class_{i}_{metric}"] = counterfactual_metrics_dict[f"class_{i}"][metric]

            # compute differences between the original and counterfactual group-specific data
            results[group][f"counterfactual_{group}_to_{counterfactual_group}_accuracy_diff"] = counterfactual_metrics_dict["accuracy"] - orig_metrics_dict["accuracy"]
            results[group][f"counterfactual_{group}_to_{counterfactual_group}_f1_diff"] = counterfactual_metrics_dict["f1"] - orig_metrics_dict["f1"]
            for i in range(args.num_labels):
                for metric in ["tpr", "fpr", "tnr", "fnr"]:
                    results[group][f"counterfactual_{group}_to_{counterfactual_group}_class_{i}_{metric}_diff"] = counterfactual_metrics_dict[f"class_{i}"][metric] - orig_metrics_dict[f"class_{i}"][metric]

            results[group][f"counterfactual_{counterfactual_group}_to_{group}_accuracy_diff"] = orig_metrics_dict["accuracy"] - counterfactual_metrics_dict["accuracy"]
            results[group][f"counterfactual_{counterfactual_group}_to_{group}_f1_diff"] = orig_metrics_dict["f1"] - counterfactual_metrics_dict["f1"]
            for i in range(args.num_labels):
                for metric in ["tpr", "fpr", "tnr", "fnr"]:
                    results[group][f"counterfactual_{counterfactual_group}_to_{group}_class_{i}_{metric}_diff"] = orig_metrics_dict[f"class_{i}"][metric] - counterfactual_metrics_dict[f"class_{i}"][metric]

            # compute confidence differences for each class 
            all_counterfactual_group_class_confidences = {i: [explanations[i]["target_class_confidence"] for explanations in counterfactual_explanations] for i in range(args.num_labels)}
            all_counterfactual_group_predicted_class_confidences = [explanations[predicted_class]["target_class_confidence"] for explanations, predicted_class in zip(counterfactual_explanations, all_orig_predictions)]

            results[group][f"counterfactual_{group}_to_{counterfactual_group}_predicted_class_confidence_diff"] = {}
            for i in range(args.num_labels):
                results[group][f"counterfactual_{group}_to_{counterfactual_group}_class_{i}_confidence_diff"] = {}
            for i in range(len(all_group_idxs)):
                results[group][f"counterfactual_{group}_to_{counterfactual_group}_predicted_class_confidence_diff"][all_group_idxs[i]] = all_counterfactual_group_predicted_class_confidences[i] - all_group_predicted_class_confidences[i]
                for j in range(args.num_labels):
                    results[group][f"counterfactual_{group}_to_{counterfactual_group}_class_{j}_confidence_diff"][all_group_idxs[i]] = all_counterfactual_group_class_confidences[j][i] - all_group_class_confidences[j][i]
            
            results[group][f"counterfactual_{counterfactual_group}_to_{group}_predicted_class_confidence_diff"] = {}
            for i in range(args.num_labels):
                results[group][f"counterfactual_{counterfactual_group}_to_{group}_class_{i}_confidence_diff"] = {}
            for i in range(len(all_group_idxs)):
                results[group][f"counterfactual_{counterfactual_group}_to_{group}_predicted_class_confidence_diff"][all_group_idxs[i]] = all_group_predicted_class_confidences[i] - all_counterfactual_group_predicted_class_confidences[i]
                for j in range(args.num_labels):
                    results[group][f"counterfactual_{counterfactual_group}_to_{group}_class_{j}_confidence_diff"][all_group_idxs[i]] = all_group_class_confidences[j][i] - all_counterfactual_group_class_confidences[j][i]

            
            results[group][f"counterfactual_{group}_to_{counterfactual_group}_predicted_class_confidence_diff_avg"] = np.mean(list(results[group][f"counterfactual_{group}_to_{counterfactual_group}_predicted_class_confidence_diff"].values()))
            results[group][f"counterfactual_{counterfactual_group}_to_{group}_predicted_class_confidence_diff_avg"] = np.mean(list(results[group][f"counterfactual_{counterfactual_group}_to_{group}_predicted_class_confidence_diff"].values()))
            results[group]["counterfactual_predicted_class_confidence_diff_avg_abs"] = np.mean(np.abs(list(results[group][f"counterfactual_{group}_to_{counterfactual_group}_predicted_class_confidence_diff"].values())))

            for i in range(args.num_labels):
                
                results[group][f"counterfactual_{group}_to_{counterfactual_group}_class_{i}_confidence_diff_avg"] = np.mean(list(results[group][f"counterfactual_{group}_to_{counterfactual_group}_class_{i}_confidence_diff"].values()))
                results[group][f"counterfactual_{counterfactual_group}_to_{group}_class_{i}_confidence_diff_avg"] = np.mean(list(results[group][f"counterfactual_{counterfactual_group}_to_{group}_class_{i}_confidence_diff"].values()))
                results[group][f"counterfactual_class_{i}_confidence_diff_avg_abs"] = np.mean(np.abs(list(results[group][f"counterfactual_{group}_to_{counterfactual_group}_class_{i}_confidence_diff"].values())))

    # compute metrics for both groups
    overall_metrics_dict = compute_metrics(all_labels, all_predictions, num_classes=args.num_labels)
    
    results["overall"] = {}
    results["overall"]["num_examples"] = len(all_labels)
    results["overall"]["accuracy"] = overall_metrics_dict["accuracy"]
    results["overall"]["f1"] = overall_metrics_dict["f1"]
    

    # compute metrics difference in both directions (group fairness)
    results["overall"][f"{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_accuracy_diff"] = results[BIAS_TYPES[args.bias_type][1]]["accuracy"] - results[BIAS_TYPES[args.bias_type][0]]["accuracy"]
    results["overall"][f"{BIAS_TYPES[args.bias_type][1]}_to_{BIAS_TYPES[args.bias_type][0]}_accuracy_diff"] = results[BIAS_TYPES[args.bias_type][0]]["accuracy"] - results[BIAS_TYPES[args.bias_type][1]]["accuracy"]
    results["overall"][f"{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_f1_diff"] = results[BIAS_TYPES[args.bias_type][1]]["f1"] - results[BIAS_TYPES[args.bias_type][0]]["f1"]
    results["overall"][f"{BIAS_TYPES[args.bias_type][1]}_to_{BIAS_TYPES[args.bias_type][0]}_f1_diff"] = results[BIAS_TYPES[args.bias_type][0]]["f1"] - results[BIAS_TYPES[args.bias_type][1]]["f1"]
    for i in range(args.num_labels):
        for metric in ["tpr", "fpr", "tnr", "fnr"]:
            results["overall"][f"class_{i}_{metric}"] = overall_metrics_dict[f"class_{i}"][metric]

    for i in range(args.num_labels):
        for metric in ["tpr", "fpr", "tnr", "fnr"]:
            results["overall"][f"{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_class_{i}_{metric}_diff"] = results[BIAS_TYPES[args.bias_type][1]][f"class_{i}_{metric}"] - results[BIAS_TYPES[args.bias_type][0]][f"class_{i}_{metric}"]
            results["overall"][f"{BIAS_TYPES[args.bias_type][1]}_to_{BIAS_TYPES[args.bias_type][0]}_class_{i}_{metric}_diff"] = results[BIAS_TYPES[args.bias_type][0]][f"class_{i}_{metric}"] - results[BIAS_TYPES[args.bias_type][1]][f"class_{i}_{metric}"]
    

    
    if args.counterfactual:
        
        for group in BIAS_TYPES[args.bias_type]:
            counterfactual_group = BIAS_TYPES[args.bias_type][0] if group == BIAS_TYPES[args.bias_type][1] else BIAS_TYPES[args.bias_type][1]
            counterfactually_augmented_metrics_dict = compute_metrics(results[group]["labels"]+results[counterfactual_group]["labels"], results[group]["predictions"]+results[counterfactual_group]["predictions"], num_classes=args.num_labels)
            results["overall"][f"counterfactually_augmented_{group}_accuracy"] = counterfactually_augmented_metrics_dict["accuracy"]
            results["overall"][f"counterfactually_augmented_{group}_f1"] = counterfactually_augmented_metrics_dict["f1"]
            for i in range(args.num_labels):
                for metric in ["tpr", "fpr", "tnr", "fnr"]:
                    results["overall"][f"counterfactually_augmented_{group}_class_{i}_{metric}"] = counterfactually_augmented_metrics_dict[f"class_{i}"][metric]
                    results["overall"][f"counterfactually_augmented_{group}_to_{counterfactual_group}_class_{i}_{metric}_diff"] = counterfactually_augmented_metrics_dict[f"class_{i}"][metric] - results[group][f"class_{i}_{metric}"]                    
            results["overall"][f"counterfactually_augmented_{group}_to_{counterfactual_group}_accuracy_diff"] = counterfactually_augmented_metrics_dict["accuracy"] - results[group]["accuracy"]
            results["overall"][f"counterfactually_augmented_{group}_to_{counterfactual_group}_f1_diff"] = counterfactually_augmented_metrics_dict["f1"] - results[group]["f1"]
            for i in range(args.num_labels):
                for metric in ["tpr", "fpr", "tnr", "fnr"]:
                    results["overall"][f"counterfactually_augmented_{group}_class_{i}_{metric}"] = counterfactually_augmented_metrics_dict[f"class_{i}"][metric]
                    results["overall"][f"counterfactually_augmented_{group}_to_{counterfactual_group}_class_{i}_{metric}_diff"] = counterfactually_augmented_metrics_dict[f"class_{i}"][metric] - results[group][f"class_{i}_{metric}"]
        
        

        results["overall"][f"counterfactually_augmented_{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_predicted_class_confidence_diff_avg"] = np.mean(list(results[BIAS_TYPES[args.bias_type][0]][f"counterfactual_{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_predicted_class_confidence_diff"].values()) + list(results[BIAS_TYPES[args.bias_type][1]][f"counterfactual_{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_predicted_class_confidence_diff"].values()))
        results["overall"][f"counterfactually_augmented_{BIAS_TYPES[args.bias_type][1]}_to_{BIAS_TYPES[args.bias_type][0]}_predicted_class_confidence_diff_avg"] = -results["overall"][f"counterfactually_augmented_{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_predicted_class_confidence_diff_avg"]
        results["overall"]["counterfactually_augmented_predicted_class_confidence_diff_avg_abs"] = np.mean(np.concatenate([np.abs(list(results[BIAS_TYPES[args.bias_type][0]][f"counterfactual_{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_predicted_class_confidence_diff"].values())), np.abs(list(results[BIAS_TYPES[args.bias_type][1]][f"counterfactual_{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_predicted_class_confidence_diff"].values()))]))

        for i in range(args.num_labels):           
            results["overall"][f"counterfactually_augmented_{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_class_{i}_confidence_diff_avg"] = np.mean(list(results[BIAS_TYPES[args.bias_type][0]][f"counterfactual_{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_class_{i}_confidence_diff"].values()) + list(results[BIAS_TYPES[args.bias_type][1]][f"counterfactual_{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_class_{i}_confidence_diff"].values()))
            results["overall"][f"counterfactually_augmented_{BIAS_TYPES[args.bias_type][1]}_to_{BIAS_TYPES[args.bias_type][0]}_class_{i}_confidence_diff_avg"] = -results["overall"][f"counterfactually_augmented_{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_class_{i}_confidence_diff_avg"]
            results["overall"][f"counterfactually_augmented_class_{i}_confidence_diff_avg_abs"] = np.mean(np.concatenate([np.abs(list(results[BIAS_TYPES[args.bias_type][0]][f"counterfactual_{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_class_{i}_confidence_diff"].values())), np.abs(list(results[BIAS_TYPES[args.bias_type][1]][f"counterfactual_{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_class_{i}_confidence_diff"].values()))]))

    
    # drop positive_confidence_diff and predicted_confidence_diff for ease of visualization
    for group in BIAS_TYPES[args.bias_type]:
        counterfactual_group = BIAS_TYPES[args.bias_type][0] if group == BIAS_TYPES[args.bias_type][1] else BIAS_TYPES[args.bias_type][1]
        results[f"{group}_counterfactual_{group}_to_{counterfactual_group}_predicted_class_confidence_diff_list"] = results[group][f"counterfactual_{group}_to_{counterfactual_group}_predicted_class_confidence_diff"]
        results[group].pop(f"counterfactual_{group}_to_{counterfactual_group}_predicted_class_confidence_diff", None)
        results[f"{group}_counterfactual_{counterfactual_group}_to_{group}_predicted_class_confidence_diff_list"] = results[group][f"counterfactual_{counterfactual_group}_to_{group}_predicted_class_confidence_diff"]
        results[group].pop(f"counterfactual_{counterfactual_group}_to_{group}_predicted_class_confidence_diff", None)
        for i in range(args.num_labels):
            results[f"{group}_counterfactual_{group}_to_{counterfactual_group}_class_{i}_confidence_diff_list"] = results[group][f"counterfactual_{group}_to_{counterfactual_group}_class_{i}_confidence_diff"]
            results[group].pop(f"counterfactual_{group}_to_{counterfactual_group}_class_{i}_confidence_diff", None)
            results[f"{group}_counterfactual_{counterfactual_group}_to_{group}_class_{i}_confidence_diff_list"] = results[group][f"counterfactual_{counterfactual_group}_to_{group}_class_{i}_confidence_diff"]
            results[group].pop(f"counterfactual_{counterfactual_group}_to_{group}_class_{i}_confidence_diff", None)
        results[f"{group}_predictions"] = results[group]["predictions"]
        results[group].pop("predictions", None)
        results[f"{group}_labels"] = results[group]["labels"]
        results[group].pop("labels", None)
        if args.counterfactual:
            results[f"{group}_counterfactual_predictions"] = results[group]["counterfactual_predictions"]
            results[group].pop("counterfactual_predictions", None)


    # save results          
    output_file = os.path.join(args.explanation_dir, f"fairness_{model_type}_{args.bias_type}_{args.split}_results.json") 
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_file}")                                                                      

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BERT Attribution with Captum')

    parser.add_argument('--explanation_dir', type=str, default='baseline_saliency_results/all_methods_1000_examples_512', help='Path to the saliency data')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use (e.g., train, test)')
    parser.add_argument('--num_labels', type=int, default=2, help='Number of labels in the classification')
    #parser.add_argument('--methods', type=str, default=None, help='List of attribution methods to use separated by commas')
    parser.add_argument('--bias_type', type=str, default='race', help='Bias type to explain')
    parser.add_argument('--counterfactual', action='store_true', help='Apply counterfactual perturbation')

    args = parser.parse_args()
    main(args)