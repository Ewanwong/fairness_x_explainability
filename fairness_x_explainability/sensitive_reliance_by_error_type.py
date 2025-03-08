import argparse
import numpy as np
import json
import os
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

RELIANCE_KEYS = ["normalized", "normalized_by_class_comparison"]

BIAS_TYPES = {
    "gender": ["female", "male"],
    "race": ["black", "white"],
}


SHOULD_CONTAIN = {
    "white": ["white", "caucasian", "europe"], 
    "black": ["black", "africa"],
    "male": [],
    "female": [],
}

SHOULD_NOT_CONTAIN = {
    "white": [],
    "black": ["nigg", "negro", "niger"],
    "male": [],
    "female": [],
}

SENSITIVE_TOKENS = {
    "white": ["white", "europe", "caucasia"],
    "black": ["black", "africa"],
    "male": [],
    "female": [],    
}



# TODO: a better filter function
def filter_text(text, should_contain, should_not_contain):
    contain_flag = False
    for word in should_contain:
        if word in text:
            contain_flag = True
            break
    if not contain_flag:
        return False
    for word in should_not_contain:
        if word in text:
            return False
    return True


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


def main(args):

    if len(BIAS_TYPES[args.bias_type]) != 2:
        raise ValueError("Only binary bias types are supported")
    
    if args.methods is not None:
        methods = args.methods.strip().split(",")
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
                    raw_reliance_score = compute_reliance_score(sensitive_attribution, total_attribution, method="raw")
                    normalized_reliance_score = compute_reliance_score(sensitive_attribution, total_attribution, method="normalize")
                    raw_class_comparison_reliance_score = compute_reliance_score_by_class_comparison(sensitive_attribution, total_attribution, other_class_sensitive_attributions, other_class_total_attributions, method="raw")
                    normalized_class_comparison_reliance_score = compute_reliance_score_by_class_comparison(sensitive_attribution, total_attribution, other_class_sensitive_attributions, other_class_total_attributions, method="normalize")
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
            
        output_file = os.path.join(args.explanation_dir, f"{method}_{args.bias_type}_{args.split}_sensitive_reliance_by_error_type.json")
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

    args = parser.parse_args()
    main(args)