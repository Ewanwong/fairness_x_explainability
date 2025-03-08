## we measure the correlation between sensitive token reliance and confidence change (in the same direction)
## the examples are split according to their predicted class and sensitive groups, the correlation is measured within each group
## the correlation is also measured for all subgroups of the sensitive group
## the correlation is also measured for all examples


import argparse
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
import os
import random
from tqdm import tqdm
import scipy.stats
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


FAIRNESS_ABS = True
RELIANCE_ABS = True
RELIANCE_KEYS = ["normalized", "normalized_by_class_comparison"]
# RELIANCE_KEYS = ["raw", "normalized", "raw_by_class_comparison", "normalized_by_class_comparison"]


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

def compute_reliance_score(sensitive_attribution, total_attribution, method="normalize"):
    # method: raw, normalize
        # TODO: make sure sensitive attribution scores are not empty
    if len(sensitive_attribution) == 0:
        return 0.0
    
    sensitive_attribution_scores = np.array([attribution_score[1] for attribution_score in sensitive_attribution])
    total_attribution_scores = np.array([attribution_score[1] for attribution_score in total_attribution])
    
    # select the sensitive attribution score with the largest magnitude
    sensitive_attribution_score =  sensitive_attribution_scores[np.argmax(np.abs(sensitive_attribution_scores))]
    if method == "raw":
        return sensitive_attribution_score
    # TODO: consider length as well for normalization
    elif method == "normalize":
        #mean_total_attribution_magnitute = np.mean(total_attribution_scores)
        norm_total_attribution_scores = np.linalg.norm(total_attribution_scores)

        normalized_sensitive_attribution_score = sensitive_attribution_score / norm_total_attribution_scores
        return normalized_sensitive_attribution_score
    else:
        raise ValueError("Method not recognized")
    
def compute_reliance_score_by_class_comparison(sensitive_attribution, total_attribution, other_class_sensitive_attributions, other_class_total_attributions, method="normalize"):
    # method: raw, normalize
    sensitive_attribution_scores = [attribution_score[1] for attribution_score in sensitive_attribution]
    other_sensitive_attribution_scores = [[attribution_score[1] for attribution_score in other_class_sensitive_attribution] for other_class_sensitive_attribution in other_class_sensitive_attributions]

    # TODO: make sure sensitive attribution scores are not empty
    if len(sensitive_attribution_scores) == 0 or 0 in [len(other_sensitive_attribution) for other_sensitive_attribution in other_class_sensitive_attributions]:
        return 0.0
    
    total_attribution_scores = [attribution_score[1] for attribution_score in total_attribution]
    other_total_attribution_scores = [[attribution_score[1] for attribution_score in other_class_total_attribution] for other_class_total_attribution in other_class_total_attributions]

    # take the mean score for each instance across other classes
    other_sensitive_attribution_scores = np.mean(other_sensitive_attribution_scores, axis=0)
    other_total_attribution_scores = np.mean(other_total_attribution_scores, axis=0)

    # compute the difference between the attribution score of the class of interest and the mean attribution score of the other classes
    sensitive_attribution_scores_diff = np.array(sensitive_attribution_scores) - other_sensitive_attribution_scores
    total_attribution_scores_diff = np.array(total_attribution_scores) - other_total_attribution_scores
    sensitive_attribution_by_class_comparison = [[token, score] for token, score in zip(sensitive_attribution, sensitive_attribution_scores_diff)]
    total_attribution_by_class_comparison = [[token, score] for token, score in zip(total_attribution, total_attribution_scores_diff)]

    return compute_reliance_score(sensitive_attribution_by_class_comparison, total_attribution_by_class_comparison, method=method)

    
def compute_all_correlations(fairness_list, reliance_scores_dict, keys=RELIANCE_KEYS, fairness_abs=True, reliance_abs=True):

    correlation_results = {"confidence_diff": {}}
    for key in keys:
        reliance_score = reliance_scores_dict[key]
        corr = scipy.stats.pearsonr(fairness_list, reliance_score)
        correlation_results["confidence_diff"][key] = corr
        if reliance_abs:
            abs_reliance_score = [abs(score) for score in reliance_score]
            abs_corr = scipy.stats.pearsonr(fairness_list, abs_reliance_score)
            correlation_results["confidence_diff"][f"{key}_abs"] = abs_corr
    
    if fairness_abs:
        abs_fairness_list = [abs(score) for score in fairness_list]
        correlation_results["abs_confidence_diff"] = {}
        for key in keys:
            reliance_score = reliance_scores_dict[key]
            abs_corr = scipy.stats.pearsonr(abs_fairness_list, reliance_score)
            correlation_results["abs_confidence_diff"][key] = abs_corr
            if reliance_abs:
                abs_reliance_score = [abs(score) for score in reliance_score]
                abs_corr = scipy.stats.pearsonr(abs_fairness_list, abs_reliance_score)
                correlation_results["abs_confidence_diff"][f"{key}_abs"] = abs_corr
    return correlation_results
    
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

    # load fairness results
    fairness_results = {}
    
    fairness_file = os.path.join(args.explanation_dir, f"fairness_{model_type}_{args.bias_type}_{args.split}_results.json")
    if not os.path.exists(fairness_file):
        raise ValueError(f"File {fairness_file} does not exist")
        
    with open(fairness_file) as f:
        fairness_results = json.load(f)   
    
    if args.methods is not None:
        methods = args.methods.strip().split(",")
    else:
        methods = EXPLANATION_METHODS

    for method in methods:
        correlation_results = {}
        attribution_file = os.path.join(args.explanation_dir, f"{method}_{BIAS_TYPES[args.bias_type][0]}_{args.split}_sensitive_attribution.json")
        if not os.path.exists(attribution_file):
            #print(f"File {sensitive_attribution_file} does not exist. Skipping...")
            continue
        
        # not use class comparison for attention methods
        reliance_keys = RELIANCE_KEYS
        if "Attention" in method:
            reliance_keys = [reliance for reliance in reliance_keys if "by_class_comparison" not in reliance]

        # collect results for all groups to compute overall correlation
        overall_fairness = {}
        overall_reliance_scores = {}

        for group in BIAS_TYPES[args.bias_type]:           
            
            attribution_file = os.path.join(args.explanation_dir, f"{method}_{group}_{args.split}_sensitive_attribution.json")
            if not os.path.exists(attribution_file):
                #print(f"File {sensitive_attribution_file} does not exist. Skipping...")
                continue
            
            # collect all predictions of the group (original)
            group_predictions = fairness_results[f"{group}_predictions"]

            with open(attribution_file) as f:
                attribution_data = json.load(f)

            # mean or L2
            aggregations = list(attribution_data.keys())
            aggregations = [aggr for aggr in aggregations if "predicted_class" not in aggr and "true_label" not in aggr]

            for aggregation in aggregations:
                attribution_results = attribution_data[aggregation]
                # make sure the attribution and fairness files have the same examples
                assert list(attribution_data["predicted_classes"].values()) == group_predictions

                

                # create correlation results dictionary
                if aggregation not in correlation_results:
                    correlation_results[aggregation] = {}
                correlation_results[aggregation][group] = {}
                correlation_results[aggregation][group][f"{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_confidence_change"] = {}
                correlation_results[aggregation][group][f"{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_abs_confidence_change"] = {}
                correlation_results[aggregation][group]["reliance_score"] = {}
                correlation_results[aggregation][group]["fairness_score"] = {}

                # initialize overall fairness and reliance scores
                if aggregation not in overall_fairness:
                    overall_fairness[aggregation] = {f"class_{target_class}": [] for target_class in range(args.num_labels)}
                    overall_fairness[aggregation]["predicted_class"] = []
                if aggregation not in overall_reliance_scores:
                    overall_reliance_scores[aggregation] = {f"class_{target_class}": {"raw":[], "normalized":[], "raw_by_class_comparison":[], "normalized_by_class_comparison":[]} for target_class in range(args.num_labels)}
                    overall_reliance_scores[aggregation]["predicted_class"] = {"raw":[], "normalized":[], "raw_by_class_comparison":[], "normalized_by_class_comparison":[]}

                # fairness and sensitive_attribution results for all classes             
                # we split the examples by the predicted group, and compute the correlation within the group for the predicted class
                group_counterfactual_fairness_results_all_classes = []
                total_attribution_results_all_classes = []
                sensitive_attribution_results_all_classes = []
                
                for target_class in range(args.num_labels):

                    class_group_counterfactual_fairness_results = list(fairness_results[f"{group}_counterfactual_{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_class_{target_class}_confidence_diff_list"].values())
                    group_counterfactual_fairness_results_all_classes.append(class_group_counterfactual_fairness_results)
                    
                    class_sensitive_attribution_results = [attribution[f"class_{target_class}"]["sensitive_attribution"] for attribution in list(attribution_results.values())]
                    class_total_attribution_results = [attribution[f"class_{target_class}"]["total_attribution"] for attribution in list(attribution_results.values())]
                    sensitive_attribution_results_all_classes.append(class_sensitive_attribution_results)
                    total_attribution_results_all_classes.append(class_total_attribution_results)
                # print(len(sensitive_attribution_results_all_classes[0]), len(total_attribution_results_all_classes[0]), len(group_counterfactual_fairness_results_all_classes[0]))
                assert len(sensitive_attribution_results_all_classes[0]) == len(group_counterfactual_fairness_results_all_classes[0])
                
                # compute correlation for each subgroup by prediction
                for target_class in range(args.num_labels):
                    subset_indexes = [i for i in range(len(group_predictions)) if group_predictions[i] == target_class]
                    # fairness scores
                    class_group_counterfactual_fairness_results = group_counterfactual_fairness_results_all_classes[target_class]
                    class_group_counterfactual_fairness_results = [class_group_counterfactual_fairness_results[i] for i in subset_indexes]
                    
                    # sensitive/total attributions
                    class_sensitive_attribution_results = sensitive_attribution_results_all_classes[target_class]
                    class_sensitive_attribution_results = [class_sensitive_attribution_results[i] for i in subset_indexes]
                    class_total_attribution_results = total_attribution_results_all_classes[target_class]
                    class_total_attribution_results = [class_total_attribution_results[i] for i in subset_indexes]
                    other_class_sensitive_attribution_results = [sensitive_attribution_results_all_classes[i] for i in range(len(sensitive_attribution_results_all_classes)) if i != target_class]
                    other_class_sensitive_attribution_results = [[other_class_sensitive_attribution_results[i][j] for j in subset_indexes] for i in range(len(other_class_sensitive_attribution_results))]
                    other_class_total_attribution_results = [total_attribution_results_all_classes[i] for i in range(len(total_attribution_results_all_classes)) if i != target_class]
                    other_class_total_attribution_results = [[other_class_total_attribution_results[i][j] for j in subset_indexes] for i in range(len(other_class_total_attribution_results))]

                    # compute reliance scores and correlations with different methods
                    raw_reliance_scores = [compute_reliance_score(class_sensitive_attribution_result, class_total_attribution_result, method="raw") for class_sensitive_attribution_result, class_total_attribution_result in zip(class_sensitive_attribution_results, class_total_attribution_results)]
                    normalized_reliance_scores = [compute_reliance_score(class_sensitive_attribution_result, class_total_attribution_result, method="normalize") for class_sensitive_attribution_result, class_total_attribution_result in zip(class_sensitive_attribution_results, class_total_attribution_results)]
                    raw_reliance_scores_by_class_comparison = [compute_reliance_score_by_class_comparison(class_sensitive_attribution_result, class_total_attribution_result, other_class_sensitive_attribution_result, other_class_total_attribution_result, method="raw") for class_sensitive_attribution_result, class_total_attribution_result, other_class_sensitive_attribution_result, other_class_total_attribution_result in zip(class_sensitive_attribution_results, class_total_attribution_results, zip(*other_class_sensitive_attribution_results), zip(*other_class_total_attribution_results))]
                    normalized_reliance_scores_by_class_comparison = [compute_reliance_score_by_class_comparison(class_sensitive_attribution_result, class_total_attribution_result, other_class_sensitive_attribution_result, other_class_total_attribution_result, method="normalize") for class_sensitive_attribution_result, class_total_attribution_result, other_class_sensitive_attribution_result, other_class_total_attribution_result in zip(class_sensitive_attribution_results, class_total_attribution_results, zip(*other_class_sensitive_attribution_results), zip(*other_class_total_attribution_results))]
                    reliance_scores_dict = {"raw": raw_reliance_scores, "normalized": normalized_reliance_scores, "raw_by_class_comparison": raw_reliance_scores_by_class_comparison, "normalized_by_class_comparison": normalized_reliance_scores_by_class_comparison}
                    correlation_output = compute_all_correlations(class_group_counterfactual_fairness_results, reliance_scores_dict, keys=reliance_keys, fairness_abs=FAIRNESS_ABS, reliance_abs=RELIANCE_ABS)
                    correlation_results[aggregation][group][f"{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_confidence_change"][f"class_{target_class}"] = correlation_output["confidence_diff"]
                    correlation_results[aggregation][group][f"{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_abs_confidence_change"][f"class_{target_class}"] = correlation_output["abs_confidence_diff"]
                    correlation_results[aggregation][group]["reliance_score"][f"class_{target_class}"] = {}
                    for key in reliance_keys:
                        correlation_results[aggregation][group]["reliance_score"][f"class_{target_class}"][f"avg_{key}"] = np.mean(reliance_scores_dict[key])
                        if RELIANCE_ABS:
                            correlation_results[aggregation][group]["reliance_score"][f"class_{target_class}"][f"avg_{key}_abs"] = np.mean([abs(score) for score in reliance_scores_dict[key]])
                    correlation_results[aggregation][group]["reliance_score"][f"class_{target_class}"]["positive_attribution_ratio"] = np.mean([1 if score > 0 else 0 for score in raw_reliance_scores])
                    correlation_results[aggregation][group]["reliance_score"][f"class_{target_class}"]["negative_attribution_ratio"] = np.mean([1 if score < 0 else 0 for score in raw_reliance_scores])

                    correlation_results[aggregation][group]["fairness_score"][f"class_{target_class}"] = {}
                    correlation_results[aggregation][group]["fairness_score"][f"class_{target_class}"]["avg_confidence_diff"] = np.mean(class_group_counterfactual_fairness_results)
                    if FAIRNESS_ABS:
                        correlation_results[aggregation][group]["fairness_score"][f"class_{target_class}"]["avg_confidence_diff_abs"] = np.mean([abs(score) for score in class_group_counterfactual_fairness_results])
                    correlation_results[aggregation][group]["fairness_score"][f"class_{target_class}"]["positive_confidence_diff_ratio"] = np.mean([1 if score > 0 else 0 for score in class_group_counterfactual_fairness_results])
                    correlation_results[aggregation][group]["fairness_score"][f"class_{target_class}"]["negative_confidence_diff_ratio"] = np.mean([1 if score < 0 else 0 for score in class_group_counterfactual_fairness_results])
                    
                    # update overall fairness and reliance scores
                    overall_fairness[aggregation][f"class_{target_class}"] += class_group_counterfactual_fairness_results
                    overall_reliance_scores[aggregation][f"class_{target_class}"]["raw"] += raw_reliance_scores
                    overall_reliance_scores[aggregation][f"class_{target_class}"]["normalized"] += normalized_reliance_scores
                    overall_reliance_scores[aggregation][f"class_{target_class}"]["raw_by_class_comparison"] += raw_reliance_scores_by_class_comparison
                    overall_reliance_scores[aggregation][f"class_{target_class}"]["normalized_by_class_comparison"] += normalized_reliance_scores_by_class_comparison

                # compute correlation for all examples
                predicted_class_group_counterfactual_fairness_results = list(fairness_results[f"{group}_counterfactual_{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_predicted_class_confidence_diff_list"].values())
                
                predicted_class_sensitive_attribution_results = [attribution["predicted_class"]["sensitive_attribution"] for attribution in list(attribution_results.values())]
                predicted_class_total_attribution_results = [attribution["predicted_class"]["total_attribution"] for attribution in list(attribution_results.values())]

                raw_reliance_scores_predicted_class = [compute_reliance_score(predicted_class_sensitive_attribution_result, predicted_class_total_attribution_result, method="raw") for predicted_class_sensitive_attribution_result, predicted_class_total_attribution_result in zip(predicted_class_sensitive_attribution_results, predicted_class_total_attribution_results)]
                normalized_reliance_scores_predicted_class = [compute_reliance_score(predicted_class_sensitive_attribution_result, predicted_class_total_attribution_result, method="normalize") for predicted_class_sensitive_attribution_result, predicted_class_total_attribution_result in zip(predicted_class_sensitive_attribution_results, predicted_class_total_attribution_results)]
                predicted_classes = list(attribution_data["predicted_classes"].values())
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
                
                reliance_scores_dict = {"raw": raw_reliance_scores_predicted_class, "normalized": normalized_reliance_scores_predicted_class, "raw_by_class_comparison": raw_reliance_scores_by_class_comparison_predicted_class, "normalized_by_class_comparison": normalized_reliance_scores_by_class_comparison_predicted_class}
                correlation_output = compute_all_correlations(predicted_class_group_counterfactual_fairness_results, reliance_scores_dict, keys=reliance_keys, fairness_abs=FAIRNESS_ABS, reliance_abs=RELIANCE_ABS)
                correlation_results[aggregation][group][f"{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_confidence_change"]["predicted_class"] = correlation_output["confidence_diff"]
                correlation_results[aggregation][group][f"{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_abs_confidence_change"]["predicted_class"] = correlation_output["abs_confidence_diff"]
                
                correlation_results[aggregation][group]["reliance_score"]["predicted_class"] = {}
                for key in reliance_keys:
                    correlation_results[aggregation][group]["reliance_score"]["predicted_class"][f"avg_{key}"] = np.mean(reliance_scores_dict[key])
                    if RELIANCE_ABS:
                        correlation_results[aggregation][group]["reliance_score"]["predicted_class"][f"avg_{key}_abs"] = np.mean([abs(score) for score in reliance_scores_dict[key]])
                correlation_results[aggregation][group]["reliance_score"]["predicted_class"]["positive_attribution_ratio"] = np.mean([1 if score > 0 else 0 for score in raw_reliance_scores_predicted_class])
                correlation_results[aggregation][group]["reliance_score"]["predicted_class"]["negative_attribution_ratio"] = np.mean([1 if score < 0 else 0 for score in raw_reliance_scores_predicted_class])

                correlation_results[aggregation][group]["fairness_score"]["predicted_class"] = {}
                correlation_results[aggregation][group]["fairness_score"]["predicted_class"]["avg_confidence_diff"] = np.mean(predicted_class_group_counterfactual_fairness_results)
                if FAIRNESS_ABS:    
                    correlation_results[aggregation][group]["fairness_score"]["predicted_class"]["avg_confidence_diff_abs"] = np.mean([abs(score) for score in predicted_class_group_counterfactual_fairness_results])
                correlation_results[aggregation][group]["fairness_score"]["predicted_class"]["positive_confidence_diff_ratio"] = np.mean([1 if score > 0 else 0 for score in predicted_class_group_counterfactual_fairness_results])
                correlation_results[aggregation][group]["fairness_score"]["predicted_class"]["negative_confidence_diff_ratio"] = np.mean([1 if score < 0 else 0 for score in predicted_class_group_counterfactual_fairness_results])

                overall_fairness[aggregation]["predicted_class"] += predicted_class_group_counterfactual_fairness_results
                overall_reliance_scores[aggregation]["predicted_class"]["raw"] += raw_reliance_scores_predicted_class
                overall_reliance_scores[aggregation]["predicted_class"]["normalized"] += normalized_reliance_scores_predicted_class
                overall_reliance_scores[aggregation]["predicted_class"]["raw_by_class_comparison"] += raw_reliance_scores_by_class_comparison_predicted_class
                overall_reliance_scores[aggregation]["predicted_class"]["normalized_by_class_comparison"] += normalized_reliance_scores_by_class_comparison_predicted_class
        
        
        # compute overall correlation results
        for aggregation in correlation_results.keys():
            correlation_results[aggregation]["overall"] = {f"{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_confidence_change": {}, f"{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_abs_confidence_change": {}}
            correlation_results[aggregation]["overall"][f"{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_confidence_change"] = {f"class_{target_class}": {"raw": [], "normalized": [], "raw_by_class_comparison": [], "normalized_by_class_comparison": []} for target_class in range(args.num_labels)}
            correlation_results[aggregation]["overall"][f"{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_abs_confidence_change"] = {f"class_{target_class}": {"raw": [], "normalized": [], "raw_by_class_comparison": [], "normalized_by_class_comparison": []} for target_class in range(args.num_labels)}
            correlation_results[aggregation]["overall"][f"{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_confidence_change"]["predicted_class"] = {"raw": [], "normalized": [], "raw_by_class_comparison": [], "normalized_by_class_comparison": []}
            correlation_results[aggregation]["overall"][f"{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_abs_confidence_change"]["predicted_class"] = {"raw": [], "normalized": [], "raw_by_class_comparison": [], "normalized_by_class_comparison": []}
            for target_class in range(args.num_labels):
                overall_fairness_list = overall_fairness[aggregation][f"class_{target_class}"]
                overall_reliance_scores_dict = overall_reliance_scores[aggregation][f"class_{target_class}"]
                correlation_output = compute_all_correlations(overall_fairness_list, overall_reliance_scores_dict, keys=reliance_keys, fairness_abs=FAIRNESS_ABS, reliance_abs=RELIANCE_ABS)
                correlation_results[aggregation]["overall"][f"{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_confidence_change"][f"class_{target_class}"] = correlation_output["confidence_diff"]
                correlation_results[aggregation]["overall"][f"{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_abs_confidence_change"][f"class_{target_class}"] = correlation_output["abs_confidence_diff"]
            overall_fairness_list = overall_fairness[aggregation]["predicted_class"]
            overall_reliance_scores_dict = overall_reliance_scores[aggregation]["predicted_class"]
            correlation_output = compute_all_correlations(overall_fairness_list, overall_reliance_scores_dict, keys=reliance_keys, fairness_abs=FAIRNESS_ABS, reliance_abs=RELIANCE_ABS)
            correlation_results[aggregation]["overall"][f"{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_confidence_change"]["predicted_class"] = correlation_output["confidence_diff"]
            correlation_results[aggregation]["overall"][f"{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_abs_confidence_change"]["predicted_class"] = correlation_output["abs_confidence_diff"]

        
        correlation_file = os.path.join(args.explanation_dir, f"correlation_{method}_{args.bias_type}_{args.split}_results.json")
        with open(correlation_file, "w") as f:
            json.dump(correlation_results, f, indent=4)
        
        print(f"Correlation results for {method} saved to {correlation_file}")
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BERT Attribution with Captum')

    parser.add_argument('--explanation_dir', type=str, default='baseline_saliency_results/all_methods_1000_examples_512', help='Path to the saliency data')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use (e.g., train, test)')
    parser.add_argument('--num_labels', type=int, default=2, help='Number of labels in the classification')
    parser.add_argument('--methods', type=str, default=None, help='List of attribution methods to use separated by commas')
    parser.add_argument('--bias_type', type=str, default='race', help='Bias type to explain')
    #parser.add_argument('--counterfactual', action='store_true', help='Apply counterfactual perturbation')

    args = parser.parse_args()
    main(args)