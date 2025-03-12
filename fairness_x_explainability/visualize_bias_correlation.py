import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import argparse
import scipy.stats
import os
from utils.utils import compute_reliance_score, compute_reliance_score_by_class_comparison
from utils.utils import BIAS_TYPES, EXPLANATION_METHODS

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='BERT Attribution with Captum')

    parser.add_argument('--explanation_dir', type=str, default='baseline_saliency_results/all_methods_1000_examples_512', help='Path to the saliency data')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use (e.g., train, test)')
    parser.add_argument('--num_labels', type=int, default=2, help='Number of labels in the classification')
    parser.add_argument('--methods', type=str, default=None, help='List of attribution methods to use separated by commas')
    parser.add_argument('--bias_type', type=str, default='race', help='Bias type to explain')
    parser.add_argument('--counterfactual', action='store_true', help='Apply counterfactual perturbation')

    args = parser.parse_args()

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
        fairness_data = json.load(f)  

    if args.methods is not None:
        explanation_methods = args.methods.split(',')
    else:
        explanation_methods = EXPLANATION_METHODS

    for method in explanation_methods:
        correlation_results = {}
        attribution_file = os.path.join(args.explanation_dir, f"{method}_{BIAS_TYPES[args.bias_type][0]}_{args.split}_sensitive_attribution.json")
        if not os.path.exists(attribution_file):
            #print(f"File {sensitive_attribution_file} does not exist. Skipping...")
            continue
        print("visualizing", method)
        with open(attribution_file) as f:
            attribution_data = json.load(f)
        aggregations = list(attribution_data.keys())
        aggregations = [aggr for aggr in aggregations if "predicted_class" not in aggr and "true_label" not in aggr]

        for aggregation in aggregations:

            visualization_data = {}
            for group in BIAS_TYPES[args.bias_type]:
                for i in range(args.num_labels):
                    visualization_data[f"class_{i}_group_{group}_attribution"] = []
                    visualization_data[f"class_{i}_group_{group}_fairness"] = []   


            all_data = {}
            for group in BIAS_TYPES[args.bias_type]:
                counterfactual_group = BIAS_TYPES[args.bias_type][1 - BIAS_TYPES[args.bias_type].index(group)]
                
                sensitive_reliances = []
                with open(os.path.join(args.explanation_dir, f"{method}_{group}_{args.split}_sensitive_attribution.json")) as f:
                    attribution_data = json.load(f)[aggregation]
                for attribution in attribution_data.values():
                    sensitive_attribution = attribution["predicted_class"]
                    sensitive_reliance_score = compute_reliance_score(sensitive_attribution['sensitive_attribution'], sensitive_attribution['total_attribution'])
                    sensitive_reliances.append(sensitive_reliance_score)
            
                for prediction, counterfactual_predicted_class_confidence_diff, sensitive_reliance in zip(fairness_data[f"{group}_predictions"], fairness_data[f"{group}_counterfactual_{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_predicted_class_confidence_diff_list"].values(), sensitive_reliances):
                    
                    visualization_data[f"class_{prediction}_group_{group}_attribution"].append(sensitive_reliance)
                    visualization_data[f"class_{prediction}_group_{group}_fairness"].append(counterfactual_predicted_class_confidence_diff)


            for RELIANCE_ABS, FAIRNESS_ABS in [(False, False), (True, True)]:
                all_attribution = []
                all_fairness = []
                all_groups = []
                for group in BIAS_TYPES[args.bias_type]:
                    for i in range(args.num_labels):
                        all_attribution += visualization_data[f"class_{i}_group_{group}_attribution"]
                        all_fairness += visualization_data[f"class_{i}_group_{group}_fairness"]
                        all_groups += [f'class_{i}_group_{group}']*len(visualization_data[f"class_{i}_group_{group}_attribution"])
                #print(len(all_attribution), len(all_fairness), len(all_groups))
                
                if RELIANCE_ABS:
                    all_attribution = np.abs(all_attribution)
                if FAIRNESS_ABS:
                    all_fairness = np.abs(all_fairness)

                # Create a DataFrame
                df = pd.DataFrame({
                'sensitive_reliance': all_attribution,
                f'{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_predicted_confidence_diff': all_fairness,
                'group': all_groups
                })

                # Plot using seaborn
                plt.figure(figsize=(16, 10))
                sns.scatterplot(data=df, x='sensitive_reliance', y=f'{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_predicted_confidence_diff', hue='group', palette='tab10')

                # Show the plot
                plt.legend(title='Groups')
                plt.show()

                if not os.path.exists("visualization"):
                    os.makedirs("visualization")
                # save the plot
                plt.savefig(f"visualization/{model_type}_{method}_{aggregation}_fairness_vs_reliance_fairness_abs_{FAIRNESS_ABS}_reliance_abs_{RELIANCE_ABS}.png")