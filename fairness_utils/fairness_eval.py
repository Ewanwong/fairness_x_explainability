import argparse
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
import os
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

EXPLANATION_METHODS = ["Bcos",
                       "Attention",
                       "Saliency",
                       "DeepLift",
                       "GuidedBackprop",
                       "InputXGradient",
                       "IntegratedGradients",
                       "SIG",
                       "Occlusion",
                       "KernelShap",
                       "ShapleyValue",                       
                       "Lime",
                       "Decompx",]


BIAS_TYPES = {
    "gender": ["female", "male"],
    "race": ["black", "white"],
}

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

# TODO: add support for multi-class error rate
def compute_metrics(labels, predictions, num_classes=2):
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    if num_classes == 2:
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn + 1e-12)
        tnr = tn / (tn + fp + 1e-12)
        fpr = fp / (fp + tn + 1e-12)
        fnr = fn / (fn + tp + 1e-12)
    else:
        print("Multi-class error rate computations are not supported yet, set to -1 by default")
        tpr = -1
        tnr = -1
        fpr = -1
        fnr = -1
    return accuracy, f1, tpr, tnr, fpr, fnr


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
    if args.counterfactual:
        all_predictions_by_group = {group:[] for group in BIAS_TYPES[args.bias_type]}
        all_labels_by_group = {group:[] for group in BIAS_TYPES[args.bias_type]}
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
        all_predictions_by_group[group].extend(all_orig_predictions)
        all_labels_by_group[group].extend(all_orig_labels)
        accuracy, f1, tpr, tnr, fpr, fnr = compute_metrics(all_orig_labels, all_orig_predictions, num_classes=args.num_labels)
        
        results[group]["num_examples"] = len(orig_explanations)
        results[group]["accuracy"] = accuracy
        results[group]["f1"] = f1
        results[group]["tpr"] = tpr
        results[group]["fpr"] = fpr
        results[group]["tnr"] = tnr
        results[group]["fnr"] = fnr
        results[group]["predictions"] = all_orig_predictions
        results[group]["labels"] = all_orig_labels

        if args.counterfactual:
            counterfactual_group = BIAS_TYPES[args.bias_type][0] if group == BIAS_TYPES[args.bias_type][1] else BIAS_TYPES[args.bias_type][1]
            all_group_idxs = [explanations[0]["index"] for explanations in orig_explanations]
            all_group_class_confidences = {i: [explanations[i]["target_class_confidence"] for explanations in orig_explanations] for i in range(args.num_labels)}
            #if args.num_labels == 2:
            #    all_group_positive_confidence = [explanations[1]["target_class_confidence"] for explanations in orig_explanations]
            all_group_predicted_class_confidence_ = [explanations[0]["predicted_class_confidence"] for explanations in orig_explanations]

            counterfactual_explanation_file = os.path.join(args.explanation_dir, f"{method}_{group}_counterfactual_{args.split}_explanations.json")
            with open(counterfactual_explanation_file) as f:
                counterfactual_data = json.load(f)
            counterfactual_explanations = counterfactual_data[aggregations[0]]

            # choose the same examples as the group explanations
            counterfactual_explanations = [explanations for explanations in counterfactual_explanations if explanations[0]["index"] in all_group_idxs]
            all_counterfactual_group_predictions = [explanations[0]["predicted_class"] for explanations in counterfactual_explanations]
            all_counterfactual_group_labels = [explanations[0]["true_label"] for explanations in counterfactual_explanations]
            counterfactual_group = BIAS_TYPES[args.bias_type][0] if group == BIAS_TYPES[args.bias_type][1] else BIAS_TYPES[args.bias_type][1]
            all_predictions_by_group[counterfactual_group].extend(all_counterfactual_group_predictions)
            all_labels_by_group[counterfactual_group].extend(all_counterfactual_group_labels)
            #all_counterfactual_predictions.extend(all_counterfactual_group_predictions)
            #all_counterfactual_labels.extend(all_counterfactual_group_labels)

            counterfactual_accuracy, counterfactual_f1, counterfactual_tpr, counterfactual_tnr, counterfactual_fpr, counterfactual_fnr = compute_metrics(all_counterfactual_group_labels, all_counterfactual_group_predictions, num_classes=args.num_labels)

            results[group]["counterfactual_accuracy"] = counterfactual_accuracy
            results[group]["counterfactual_f1"] = counterfactual_f1
            results[group]["counterfactual_tpr"] = counterfactual_tpr
            results[group]["counterfactual_fpr"] = counterfactual_fpr
            results[group]["counterfactual_tnr"] = counterfactual_tnr
            results[group]["counterfactual_fnr"] = counterfactual_fnr
            results[group]["counterfactual_predictions"] = all_counterfactual_group_predictions
            results[group]["counterfactual_labels"] = all_counterfactual_group_labels
            
            
            all_counterfactual_group_class_confidences = {i: [explanations[i]["target_class_confidence"] for explanations in counterfactual_explanations] for i in range(args.num_labels)}
            #if args.num_labels == 2:
            #    all_counterfactual_group_positive_confidence = [explanations[1]["target_class_confidence"] for explanations in counterfactual_explanations]
            all_counterfactual_group_predicted_class_confidence_ = [explanations[predicted_class]["target_class_confidence"] for explanations, predicted_class in zip(counterfactual_explanations, all_orig_predictions)]


            results[group][f"counterfactual_{group}_to_{counterfactual_group}_accuracy_diff"] = counterfactual_accuracy - accuracy
            results[group][f"counterfactual_{group}_to_{counterfactual_group}_f1_diff"] = counterfactual_f1 - f1
            results[group][f"counterfactual_{group}_to_{counterfactual_group}_tpr_diff"] = counterfactual_tpr - tpr
            results[group][f"counterfactual_{group}_to_{counterfactual_group}_fpr_diff"] = counterfactual_fpr - fpr
            results[group][f"counterfactual_{group}_to_{counterfactual_group}_tnr_diff"] = counterfactual_tnr - tnr
            results[group][f"counterfactual_{group}_to_{counterfactual_group}_fnr_diff"] = counterfactual_fnr - fnr

            results[group][f"counterfactual_{counterfactual_group}_to_{group}_accuracy_diff"] = accuracy - counterfactual_accuracy
            results[group][f"counterfactual_{counterfactual_group}_to_{group}_f1_diff"] = f1 - counterfactual_f1
            results[group][f"counterfactual_{counterfactual_group}_to_{group}_tpr_diff"] = tpr - counterfactual_tpr
            results[group][f"counterfactual_{counterfactual_group}_to_{group}_fpr_diff"] = fpr - counterfactual_fpr
            results[group][f"counterfactual_{counterfactual_group}_to_{group}_tnr_diff"] = tnr - counterfactual_tnr
            results[group][f"counterfactual_{counterfactual_group}_to_{group}_fnr_diff"] = fnr - counterfactual_fnr
            
            results[group][f"counterfactual_{group}_to_{counterfactual_group}_predicted_class_confidence_diff"] = {}
            for i in range(args.num_labels):
                results[group][f"counterfactual_{group}_to_{counterfactual_group}_class_{i}_confidence_diff"] = {}
            for i in range(len(all_group_idxs)):
                results[group][f"counterfactual_{group}_to_{counterfactual_group}_predicted_class_confidence_diff"][all_group_idxs[i]] = all_counterfactual_group_predicted_class_confidence_[i] - all_group_predicted_class_confidence_[i]
                for j in range(args.num_labels):
                    results[group][f"counterfactual_{group}_to_{counterfactual_group}_class_{j}_confidence_diff"][all_group_idxs[i]] = all_counterfactual_group_class_confidences[j][i] - all_group_class_confidences[j][i]
            
            results[group][f"counterfactual_{counterfactual_group}_to_{group}_predicted_class_confidence_diff"] = {}
            for i in range(args.num_labels):
                results[group][f"counterfactual_{counterfactual_group}_to_{group}_class_{i}_confidence_diff"] = {}
            for i in range(len(all_group_idxs)):
                results[group][f"counterfactual_{counterfactual_group}_to_{group}_predicted_class_confidence_diff"][all_group_idxs[i]] = all_group_predicted_class_confidence_[i] - all_counterfactual_group_predicted_class_confidence_[i]
                for j in range(args.num_labels):
                    results[group][f"counterfactual_{counterfactual_group}_to_{group}_class_{j}_confidence_diff"][all_group_idxs[i]] = all_group_class_confidences[j][i] - all_counterfactual_group_class_confidences[j][i]

            
            results[group][f"counterfactual_{group}_to_{counterfactual_group}_predicted_class_confidence_diff_avg"] = np.mean(list(results[group][f"counterfactual_{group}_to_{counterfactual_group}_predicted_class_confidence_diff"].values()))
            results[group][f"counterfactual_{counterfactual_group}_to_{group}_predicted_class_confidence_diff_avg"] = np.mean(list(results[group][f"counterfactual_{counterfactual_group}_to_{group}_predicted_class_confidence_diff"].values()))
            
            results[group]["counterfactual_predicted_class_confidence_diff_avg_abs"] = np.mean(np.abs(list(results[group][f"counterfactual_{group}_to_{counterfactual_group}_predicted_class_confidence_diff"].values())))
            #if args.num_labels == 2:
            for i in range(args.num_labels):
                results[group][f"counterfactual_{group}_to_{counterfactual_group}_class_{i}_confidence_diff_avg"] = np.mean(list(results[group][f"counterfactual_{group}_to_{counterfactual_group}_class_{i}_confidence_diff"].values()))
                results[group][f"counterfactual_{counterfactual_group}_to_{group}_class_{i}_confidence_diff_avg"] = np.mean(list(results[group][f"counterfactual_{counterfactual_group}_to_{group}_class_{i}_confidence_diff"].values()))
                results[group][f"counterfactual_class_{i}_confidence_diff_avg_abs"] = np.mean(np.abs(list(results[group][f"counterfactual_{group}_to_{counterfactual_group}_class_{i}_confidence_diff"].values())))


    overall_accuracy = accuracy_score(all_labels, all_predictions)
    overall_f1 = f1_score(all_labels, all_predictions, average='macro')
    overall_tpr = sum([1 for pred, label in zip(all_predictions, all_labels) if pred==label and pred==1]) / sum([1 for label in all_labels if label==1])
    overall_fpr = sum([1 for pred, label in zip(all_predictions, all_labels) if pred!=label and pred==1]) / sum([1 for label in all_labels if label==0])
    overall_tnr = sum([1 for pred, label in zip(all_predictions, all_labels) if pred==label and pred==0]) / sum([1 for label in all_labels if label==0])
    overall_fnr = sum([1 for pred, label in zip(all_predictions, all_labels) if pred!=label and pred==0]) / sum([1 for label in all_labels if label==1])
    


    results["overall"] = {}
    results["overall"]["num_examples"] = len(all_labels)
    results["overall"]["accuracy"] = overall_accuracy
    results["overall"]["f1"] = overall_f1
    results["overall"]["tpr"] = overall_tpr
    results["overall"]["fpr"] = overall_fpr
    results["overall"]["tnr"] = overall_tnr
    results["overall"]["fnr"] = overall_fnr

    for metric in ["accuracy", "f1", "tpr", "fpr", "tnr", "fnr"]:
        
        results["overall"][f"{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_{metric}_diff"] = results[BIAS_TYPES[args.bias_type][1]][metric] - results[BIAS_TYPES[args.bias_type][0]][metric]
        results["overall"][f"{BIAS_TYPES[args.bias_type][1]}_to_{BIAS_TYPES[args.bias_type][0]}_{metric}_diff"] = results[BIAS_TYPES[args.bias_type][0]][metric] - results[BIAS_TYPES[args.bias_type][1]][metric]

    
    if args.counterfactual:
        
        for group in BIAS_TYPES[args.bias_type]:
            counterfactual_augmentation_accuracy, counterfactual_augmentation_f1, counterfactual_augmentation_tpr, counterfactual_augmentation_tnr, counterfactual_augmentation_fpr, counterfactual_augmentation_fnr = compute_metrics(all_labels_by_group[group], all_predictions_by_group[group], num_classes=args.num_labels)
            results["overall"][f"counterfactually_augmented_{group}_accuracy"] = counterfactual_augmentation_accuracy
            results["overall"][f"counterfactually_augmented_{group}_f1"] = counterfactual_augmentation_f1
            results["overall"][f"counterfactually_augmented_{group}_tpr"] = counterfactual_augmentation_tpr
            results["overall"][f"counterfactually_augmented_{group}_fpr"] = counterfactual_augmentation_fpr
            results["overall"][f"counterfactually_augmented_{group}_tnr"] = counterfactual_augmentation_tnr
            results["overall"][f"counterfactually_augmented_{group}_fnr"] = counterfactual_augmentation_fnr
        
        for metric in ["accuracy", "f1", "tpr", "fpr", "tnr", "fnr"]:
            results["overall"][f"counterfactually_augmented_{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_{metric}_diff"] = results["overall"][f"counterfactually_augmented_{BIAS_TYPES[args.bias_type][1]}_{metric}"] - results["overall"][f"counterfactually_augmented_{BIAS_TYPES[args.bias_type][0]}_{metric}"]
            results["overall"][f"counterfactually_augmented_{BIAS_TYPES[args.bias_type][1]}_to_{BIAS_TYPES[args.bias_type][0]}_{metric}_diff"] = results["overall"][f"counterfactually_augmented_{BIAS_TYPES[args.bias_type][0]}_{metric}"] - results["overall"][f"counterfactually_augmented_{BIAS_TYPES[args.bias_type][1]}_{metric}"]

        results["overall"][f"counterfactually_augmented_{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_predicted_class_confidence_diff_avg"] = np.mean(list(results[BIAS_TYPES[args.bias_type][0]][f"counterfactual_{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_predicted_class_confidence_diff"].values()) + list(results[BIAS_TYPES[args.bias_type][1]][f"counterfactual_{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_predicted_class_confidence_diff"].values()))
        results["overall"][f"counterfactually_augmented_{BIAS_TYPES[args.bias_type][1]}_to_{BIAS_TYPES[args.bias_type][0]}_predicted_class_confidence_diff_avg"] = -results["overall"][f"counterfactually_augmented_{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_predicted_class_confidence_diff_avg"]
        results["overall"]["counterfactually_augmented_predicted_class_confidence_diff_avg_abs"] = np.mean(np.concatenate([np.abs(list(results[BIAS_TYPES[args.bias_type][0]][f"counterfactual_{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_predicted_class_confidence_diff"].values())), np.abs(list(results[BIAS_TYPES[args.bias_type][1]][f"counterfactual_{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_predicted_class_confidence_diff"].values()))]))
        #if args.num_labels == 2:
        for i in range(args.num_labels):
            results["overall"][f"counterfactually_augmented_{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_class_{i}_confidence_diff_avg"] = np.mean(list(results[BIAS_TYPES[args.bias_type][0]][f"counterfactual_{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_class_{i}_confidence_diff"].values()) + list(results[BIAS_TYPES[args.bias_type][1]][f"counterfactual_{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_class_{i}_confidence_diff"].values()))
            results["overall"][f"counterfactually_augmented_{BIAS_TYPES[args.bias_type][1]}_to_{BIAS_TYPES[args.bias_type][0]}_class_{i}_confidence_diff_avg"] = -results["overall"][f"counterfactually_augmented_{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_class_{i}_confidence_diff_avg"]
            results["overall"][f"counterfactually_augmented_class_{i}_confidence_diff_avg_abs"] = np.mean(np.concatenate([np.abs(list(results[BIAS_TYPES[args.bias_type][0]][f"counterfactual_{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_class_{i}_confidence_diff"].values())), np.abs(list(results[BIAS_TYPES[args.bias_type][1]][f"counterfactual_{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_class_{i}_confidence_diff"].values()))]))

        
            
    output_file = os.path.join(args.explanation_dir, f"fairness_{model_type}_{args.bias_type}_{args.split}_results.json") 

    # drop positive_confidence_diff and predicted_confidence_diff for ease of visualization
    # TODO: aggregate to directions, rather than social groups?
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
            results[f"{group}_counterfactual_labels"] = results[group]["counterfactual_labels"]
            results[group].pop("counterfactual_labels", None)
        

    # save results  
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