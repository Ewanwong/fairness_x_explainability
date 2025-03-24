import argparse
import numpy as np
import json
import os
from tqdm import tqdm
from utils.utils import filter_text, compute_reliance_score, compute_reliance_score_by_class_comparison
from utils.utils import EXPLANATION_METHODS, BIAS_TYPES

RELIANCE_KEYS = ["normalized", "normalized_by_class_comparison"]

def main(args):

    if len(BIAS_TYPES[args.bias_type]) != 2:
        raise ValueError("Only binary bias types are supported")
    
    if args.methods is not None:
        methods = args.methods.replace(' ', '').split(",")
    else:
        methods = EXPLANATION_METHODS

    for method in methods:
        sensitive_reliance_results = {}
        attribution_file = os.path.join(args.explanation_dir, f"{method}_{BIAS_TYPES[args.bias_type][0]}_{args.split}_sensitive_attribution.json")
        if not os.path.exists(attribution_file):
            #print(f"File {sensitive_attribution_file} does not exist. Skipping...")
            continue
        else:
            with open(attribution_file) as f:
                orig_data = json.load(f)
            aggregations = list(orig_data.keys())
            aggregations = [aggr for aggr in aggregations if "predicted_class" not in aggr and "true_label" not in aggr]

        # not use class comparison for attention methods
        reliance_keys = RELIANCE_KEYS
        if "Attention" in method:
            reliance_keys = [reliance for reliance in reliance_keys if "by_class_comparison" not in reliance]

        for group in BIAS_TYPES[args.bias_type]:   
            attribution_file = os.path.join(args.explanation_dir, f"{method}_{group}_{args.split}_sensitive_attribution.json")
            if not os.path.exists(attribution_file):
                #print(f"File {group_explanation_file} does not exist. Skipping...")
                continue
            print(f"Extracting sensitive attribution from {attribution_file}")
            
            with open(attribution_file) as f:
                orig_data = json.load(f)
            
            predictions = list(orig_data["predicted_classes"].values())
            labels = list(orig_data["true_labels"].values())
            for aggregation in aggregations:  

                attribution_results = orig_data[aggregation]             
                if aggregation not in sensitive_reliance_results:
                    sensitive_reliance_results[aggregation] = {g: {} for g in BIAS_TYPES[args.bias_type]}
                    sensitive_reliance_results[aggregation]["overall"] = {}
                
                for i, (prediction, label) in tqdm(enumerate(zip(predictions, labels))):
                    attribution = list(attribution_results.values())[i]
                    sensitive_attribution = attribution["predicted_class"]["sensitive_attribution"]
                    total_attribution = attribution["predicted_class"]["total_attribution"]
                    other_class_sensitive_attributions = [attribution[f"class_{other_class}"]["sensitive_attribution"] for other_class in range(args.num_labels) if other_class != prediction]
                    other_class_total_attributions = [attribution[f"class_{other_class}"]["total_attribution"] for other_class in range(args.num_labels) if other_class != prediction]
                    raw_reliance_score = compute_reliance_score(sensitive_attribution, total_attribution, method="raw", normalization_factor=args.normalization_factor)
                    normalized_reliance_score = compute_reliance_score(sensitive_attribution, total_attribution, method="normalize", normalization_factor=args.normalization_factor)
                    raw_class_comparison_reliance_score = compute_reliance_score_by_class_comparison(sensitive_attribution, total_attribution, other_class_sensitive_attributions, other_class_total_attributions, method="raw", normalization_factor=args.normalization_factor)
                    normalized_class_comparison_reliance_score = compute_reliance_score_by_class_comparison(sensitive_attribution, total_attribution, other_class_sensitive_attributions, other_class_total_attributions, method="normalize", normalization_factor=args.normalization_factor)
                    reliance_dict = {"raw": raw_reliance_score, "normalized": normalized_reliance_score, "raw_by_class_comparison": raw_class_comparison_reliance_score, "normalized_by_class_comparison": normalized_class_comparison_reliance_score}
                    for target_class in range(args.num_labels):
                        if f"class_{target_class}" not in sensitive_reliance_results[aggregation][group]:
                            sensitive_reliance_results[aggregation][group][f"class_{target_class}"] = {"TP": {}, "FP": {}, "TN": {}, "FN": {}}
                            for type in sensitive_reliance_results[aggregation][group][f"class_{target_class}"]:
                                sensitive_reliance_results[aggregation][group][f"class_{target_class}"][type] = {reliance: [] for reliance in reliance_keys}
                        if f"class_{target_class}" not in sensitive_reliance_results[aggregation]["overall"]:
                            sensitive_reliance_results[aggregation]["overall"][f"class_{target_class}"] = {"TP": {}, "FP": {}, "TN": {}, "FN": {}}
                            for type in sensitive_reliance_results[aggregation]["overall"][f"class_{target_class}"]:
                                sensitive_reliance_results[aggregation]["overall"][f"class_{target_class}"][type] = {reliance: [] for reliance in reliance_keys}
                        
                        if target_class == prediction and target_class == label:
                            for reliance in reliance_keys:
                                sensitive_reliance_results[aggregation][group][f"class_{target_class}"]["TP"][reliance].append(reliance_dict[reliance])
                                sensitive_reliance_results[aggregation]["overall"][f"class_{target_class}"]["TP"][reliance].append(reliance_dict[reliance])
                                
                        elif target_class == prediction and target_class != label:
                            for reliance in reliance_keys:
                                sensitive_reliance_results[aggregation][group][f"class_{target_class}"]["FP"][reliance].append(reliance_dict[reliance])
                                sensitive_reliance_results[aggregation]["overall"][f"class_{target_class}"]["FP"][reliance].append(reliance_dict[reliance])
                                
                        elif target_class != prediction and target_class == label:
                            for reliance in reliance_keys:
                                sensitive_reliance_results[aggregation][group][f"class_{target_class}"]["FN"][reliance].append(reliance_dict[reliance])
                                sensitive_reliance_results[aggregation]["overall"][f"class_{target_class}"]["FN"][reliance].append(reliance_dict[reliance])
                                
                        elif target_class != prediction and target_class != label:
                            for reliance in reliance_keys:
                                sensitive_reliance_results[aggregation][group][f"class_{target_class}"]["TN"][reliance].append(reliance_dict[reliance])
                                sensitive_reliance_results[aggregation]["overall"][f"class_{target_class}"]["TN"][reliance].append(reliance_dict[reliance])
                                
                
                for target_class in range(args.num_labels):
                    for type in ["TP", "FP", "TN", "FN"]:
                        sensitive_reliance_results[aggregation][group][f"class_{target_class}"][type]["num_exsamples"] = len(sensitive_reliance_results[aggregation][group][f"class_{target_class}"][type][reliance_keys[0]])
                        for reliance in reliance_keys:
                            if len(sensitive_reliance_results[aggregation][group][f"class_{target_class}"][type][reliance]) > 0:
                                sensitive_reliance_results[aggregation][group][f"class_{target_class}"][type][f"avg_{reliance}"] = np.mean(sensitive_reliance_results[aggregation][group][f"class_{target_class}"][type][reliance])
                                sensitive_reliance_results[aggregation][group][f"class_{target_class}"][type][f"avg_abs_{reliance}"] = np.mean(np.abs(sensitive_reliance_results[aggregation][group][f"class_{target_class}"][type][reliance]))
                            else:
                                sensitive_reliance_results[aggregation][group][f"class_{target_class}"][type][f"avg_{reliance}"] = 0.0
                    sensitive_reliance_results[aggregation][group][f"class_{target_class}"]["TP"]["TPR"] = sensitive_reliance_results[aggregation][group][f"class_{target_class}"]["TP"]["num_exsamples"] / (sensitive_reliance_results[aggregation][group][f"class_{target_class}"]["TP"]["num_exsamples"] + sensitive_reliance_results[aggregation][group][f"class_{target_class}"]["FN"]["num_exsamples"])
                    sensitive_reliance_results[aggregation][group][f"class_{target_class}"]["FP"]["FPR"] = sensitive_reliance_results[aggregation][group][f"class_{target_class}"]["FP"]["num_exsamples"] / (sensitive_reliance_results[aggregation][group][f"class_{target_class}"]["FP"]["num_exsamples"] + sensitive_reliance_results[aggregation][group][f"class_{target_class}"]["TN"]["num_exsamples"])
                    sensitive_reliance_results[aggregation][group][f"class_{target_class}"]["FN"]["FNR"] = sensitive_reliance_results[aggregation][group][f"class_{target_class}"]["FN"]["num_exsamples"] / (sensitive_reliance_results[aggregation][group][f"class_{target_class}"]["TP"]["num_exsamples"] + sensitive_reliance_results[aggregation][group][f"class_{target_class}"]["FN"]["num_exsamples"])
                    sensitive_reliance_results[aggregation][group][f"class_{target_class}"]["TN"]["TNR"] = sensitive_reliance_results[aggregation][group][f"class_{target_class}"]["TN"]["num_exsamples"] / (sensitive_reliance_results[aggregation][group][f"class_{target_class}"]["FP"]["num_exsamples"] + sensitive_reliance_results[aggregation][group][f"class_{target_class}"]["TN"]["num_exsamples"])

        for aggregation in aggregations:
            for target_class in range(args.num_labels):
                for type in ["TP", "FP", "TN", "FN"]:
                    sensitive_reliance_results[aggregation]["overall"][f"class_{target_class}"][type]["num_exsamples"] = len(sensitive_reliance_results[aggregation]["overall"][f"class_{target_class}"][type][reliance_keys[0]])
                    for reliance in reliance_keys:
                        if len(sensitive_reliance_results[aggregation]["overall"][f"class_{target_class}"][type][reliance]) > 0:
                            sensitive_reliance_results[aggregation]["overall"][f"class_{target_class}"][type][f"avg_{reliance}"] = np.mean(sensitive_reliance_results[aggregation]["overall"][f"class_{target_class}"][type][reliance])
                            sensitive_reliance_results[aggregation]["overall"][f"class_{target_class}"][type][f"avg_abs_{reliance}"] = np.mean(np.abs(sensitive_reliance_results[aggregation]["overall"][f"class_{target_class}"][type][reliance]))
                        else:
                            sensitive_reliance_results[aggregation]["overall"][f"class_{target_class}"][type][f"avg_{reliance}"] = 0.0

                sensitive_reliance_results[aggregation]["overall"][f"class_{target_class}"]["TP"]["TPR"] = sensitive_reliance_results[aggregation]["overall"][f"class_{target_class}"]["TP"]["num_exsamples"] / (sensitive_reliance_results[aggregation]["overall"][f"class_{target_class}"]["TP"]["num_exsamples"] + sensitive_reliance_results[aggregation]["overall"][f"class_{target_class}"]["FN"]["num_exsamples"])
                sensitive_reliance_results[aggregation]["overall"][f"class_{target_class}"]["FP"]["FPR"] = sensitive_reliance_results[aggregation]["overall"][f"class_{target_class}"]["FP"]["num_exsamples"] / (sensitive_reliance_results[aggregation]["overall"][f"class_{target_class}"]["FP"]["num_exsamples"] + sensitive_reliance_results[aggregation]["overall"][f"class_{target_class}"]["TN"]["num_exsamples"])
                sensitive_reliance_results[aggregation]["overall"][f"class_{target_class}"]["FN"]["FNR"] = sensitive_reliance_results[aggregation]["overall"][f"class_{target_class}"]["FN"]["num_exsamples"] / (sensitive_reliance_results[aggregation]["overall"][f"class_{target_class}"]["TP"]["num_exsamples"] + sensitive_reliance_results[aggregation]["overall"][f"class_{target_class}"]["FN"]["num_exsamples"])
                sensitive_reliance_results[aggregation]["overall"][f"class_{target_class}"]["TN"]["TNR"] = sensitive_reliance_results[aggregation]["overall"][f"class_{target_class}"]["TN"]["num_exsamples"] / (sensitive_reliance_results[aggregation]["overall"][f"class_{target_class}"]["FP"]["num_exsamples"] + sensitive_reliance_results[aggregation]["overall"][f"class_{target_class}"]["TN"]["num_exsamples"])
        # put the lists into a separate dictionary
        for aggregation in aggregations:
            sensitive_reliance_results[f"{aggregation}_list"] = {}
            sensitive_reliance_results[f"{aggregation}_list"]["overall"] = {}
            for group in BIAS_TYPES[args.bias_type]:
                sensitive_reliance_results[f"{aggregation}_list"][group] = {}
                for target_class in range(args.num_labels):
                    sensitive_reliance_results[f"{aggregation}_list"][group][f"class_{target_class}"] = {}                    
                    for type in ["TP", "FP", "TN", "FN"]:
                        sensitive_reliance_results[f"{aggregation}_list"][group][f"class_{target_class}"][type] = {}                        
                        for reliance in reliance_keys:
                            sensitive_reliance_results[f"{aggregation}_list"][group][f"class_{target_class}"][type][reliance] = list(sensitive_reliance_results[aggregation][group][f"class_{target_class}"][type][reliance])                            
                            # remove the original lists
                            del sensitive_reliance_results[aggregation][group][f"class_{target_class}"][type][reliance]


            for target_class in range(args.num_labels):                
                sensitive_reliance_results[f"{aggregation}_list"]["overall"][f"class_{target_class}"] = {}
                for type in ["TP", "FP", "TN", "FN"]:                    
                    sensitive_reliance_results[f"{aggregation}_list"]["overall"][f"class_{target_class}"][type] = {}
                    for reliance in reliance_keys:                        
                        sensitive_reliance_results[f"{aggregation}_list"]["overall"][f"class_{target_class}"][type][reliance] = list(sensitive_reliance_results[aggregation]["overall"][f"class_{target_class}"][type][reliance])
                        # remove the original lists                        
                        del sensitive_reliance_results[aggregation]["overall"][f"class_{target_class}"][type][reliance]
            
        output_file = os.path.join(args.explanation_dir, f"{method}_{args.bias_type}_{args.split}_sensitive_reliance_by_error_type_division_by_{args.normalization_factor}.json")
        with open(output_file, "w") as f:   
            json.dump(sensitive_reliance_results, f, indent=4)
        print(f"Sensitive reliance results saved to {output_file}")
            
        
                                                                        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BERT Attribution with Captum')

    parser.add_argument('--explanation_dir', type=str, default='baseline_saliency_results/all_methods_1000_examples_512', help='Path to the saliency data')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use (e.g., train, test)')
    parser.add_argument('--num_labels', type=int, default=2, help='Number of labels in the classification')
    parser.add_argument('--methods', type=str, default=None, help='List of attribution methods to use separated by commas')
    parser.add_argument('--bias_type', type=str, default='race', help='Bias type to explain')
    parser.add_argument('--counterfactual', action='store_true', help='Apply counterfactual perturbation')
    parser.add_argument('--normalization_factor', type=str, default='max', choices=["max", "std", "norm"], help='Normalization method for the attribution scores')
    args = parser.parse_args()
    main(args)