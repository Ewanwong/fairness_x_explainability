import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import argparse
import scipy.stats
import os

FAIRNESS_ABS = False
RELIANCE_ABS = False

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
        with open(attribution_file) as f:
            attribution_data = json.load(f)
        aggregations = list(attribution_data.keys())
        aggregations = [aggr for aggr in aggregations if "predicted_class" not in aggr and "true_label" not in aggr]

        for aggregation in aggregations:


            class_positive_race_black_attribution = []
            class_positive_race_black_fairness = []

            class_positive_race_white_attribution = []
            class_positive_race_white_fairness = []

            class_negative_race_black_attribution = []
            class_negative_race_black_fairness = []

            class_negative_race_white_attribution = []
            class_negative_race_white_fairness = []

    

            with open(os.path.join(args.explanation_dir, f"{method}_{BIAS_TYPES[args.bias_type][0]}_{args.split}_sensitive_attribution.json")) as f:
                black_attribution_data = json.load(f)[aggregation]
            with open(os.path.join(args.explanation_dir, f"{method}_{BIAS_TYPES[args.bias_type][1]}_{args.split}_sensitive_attribution.json")) as f:
                white_attribution_data = json.load(f)[aggregation]

            black_predictions = fairness_data[f"{BIAS_TYPES[args.bias_type][0]}_predictions"]
            white_predictions = fairness_data[f"{BIAS_TYPES[args.bias_type][1]}_predictions"]
            black_counterfactual_predicted_class_confidence_diff = fairness_data[f"{BIAS_TYPES[args.bias_type][0]}_counterfactual_{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_predicted_class_confidence_diff_list"].values()
            white_counterfactual_predicted_class_confidence_diff = fairness_data[f"{BIAS_TYPES[args.bias_type][1]}_counterfactual_{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_predicted_class_confidence_diff_list"].values()

            black_sensitive_attributions = []
            white_sensitive_attributions = []
            for black_attribution in black_attribution_data.values():
                sensitive_attribution = black_attribution["predicted_class"]
                sensitive_reliance_score = compute_reliance_score(sensitive_attribution['sensitive_attribution'], sensitive_attribution['total_attribution'])
                black_sensitive_attributions.append(sensitive_reliance_score)
            
            for white_attribution in white_attribution_data.values():
                sensitive_attribution = white_attribution["predicted_class"]
                sensitive_reliance_score = compute_reliance_score(sensitive_attribution['sensitive_attribution'], sensitive_attribution['total_attribution'])
                white_sensitive_attributions.append(sensitive_reliance_score)

            print(len(black_predictions), len(black_counterfactual_predicted_class_confidence_diff), len(black_sensitive_attributions))
            for black_prediction, black_counterfactual_predicted_class_confidence_diff, black_sensitive_attribution in zip(black_predictions, black_counterfactual_predicted_class_confidence_diff, black_sensitive_attributions):
                if black_prediction == 1:
                    class_positive_race_black_fairness.append(black_counterfactual_predicted_class_confidence_diff)
                    class_positive_race_black_attribution.append(black_sensitive_attribution)
                else:
                    class_negative_race_black_fairness.append(black_counterfactual_predicted_class_confidence_diff)
                    class_negative_race_black_attribution.append(black_sensitive_attribution)

            print(len(white_predictions), len(white_counterfactual_predicted_class_confidence_diff), len(white_sensitive_attributions))
            for white_prediction, white_counterfactual_predicted_class_confidence_diff, white_sensitive_attribution in zip(white_predictions, white_counterfactual_predicted_class_confidence_diff, white_sensitive_attributions):
                if white_prediction == 1:
                    class_positive_race_white_fairness.append(white_counterfactual_predicted_class_confidence_diff)
                    class_positive_race_white_attribution.append(white_sensitive_attribution)
                else:
                    class_negative_race_white_fairness.append(white_counterfactual_predicted_class_confidence_diff)
                    class_negative_race_white_attribution.append(white_sensitive_attribution)
            """
            print(f"class positive, race {BIAS_TYPES[args.bias_type][0]}: correlation")
            print(scipy.stats.pearsonr(class_positive_race_black_attribution, class_positive_race_black_fairness))
            print(f"class positive, race {BIAS_TYPES[args.bias_type][1]}: correlation")
            print(scipy.stats.pearsonr(class_positive_race_white_attribution, class_positive_race_white_fairness))
            print(f"class negative, race {BIAS_TYPES[args.bias_type][0]}: correlation")
            print(scipy.stats.pearsonr(class_negative_race_black_attribution, class_negative_race_black_fairness))
            print(f"class negative, race {BIAS_TYPES[args.bias_type][1]}: correlation")
            print(scipy.stats.pearsonr(class_negative_race_white_attribution, class_negative_race_white_fairness))    
            print(f"class positive: correlation")
            print(scipy.stats.pearsonr(class_positive_race_black_attribution+class_positive_race_white_attribution, class_positive_race_black_fairness+class_positive_race_white_fairness))
            print(f"class negative: correlation")
            print(scipy.stats.pearsonr(class_negative_race_black_attribution+class_negative_race_white_attribution, class_negative_race_black_fairness+class_negative_race_white_fairness))
            print(f"race {BIAS_TYPES[args.bias_type][0]}: correlation")
            print(scipy.stats.pearsonr(class_positive_race_black_attribution+class_negative_race_black_attribution, class_positive_race_black_fairness+class_negative_race_black_fairness))
            print(f"race {BIAS_TYPES[args.bias_type][1]}: correlation")"
            print(scipy.stats.pearsonr(class_positive_race_white_attribution+class_negative_race_white_attribution, class_positive_race_white_fairness+class_negative_race_white_fairness))
            """
            for RELIANCE_ABS, FAIRNESS_ABS in [(False, False), (True, True)]:
                all_attribution = class_positive_race_black_attribution + class_positive_race_white_attribution + class_negative_race_black_attribution + class_negative_race_white_attribution
                all_fairness = class_positive_race_black_fairness + class_positive_race_white_fairness + class_negative_race_black_fairness + class_negative_race_white_fairness
                """
                print("all: correlation")
                print(scipy.stats.pearsonr(all_attribution, all_fairness))
                """
                if RELIANCE_ABS:
                    all_attribution = np.abs(all_attribution)
                if FAIRNESS_ABS:
                    all_fairness = np.abs(all_fairness)

                # Create a DataFrame
                df = pd.DataFrame({
                'sensitive_reliance': all_attribution,
                f'{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_predicted_confidence_diff': all_fairness,
                'group': [f'class positive, race {BIAS_TYPES[args.bias_type][0]}']*len(class_positive_race_black_attribution) + [f'class positive, race {BIAS_TYPES[args.bias_type][1]}']*len(class_positive_race_white_attribution) + [f'class negative, race {BIAS_TYPES[args.bias_type][0]}']*len(class_negative_race_black_attribution) + [f'class negative, race {BIAS_TYPES[args.bias_type][1]}']*len(class_negative_race_white_attribution)  # Group labels for color differentiation
                })

                # Plot using seaborn
                plt.figure(figsize=(16, 10))
                sns.scatterplot(data=df, x='sensitive_reliance', y=f'{BIAS_TYPES[args.bias_type][0]}_to_{BIAS_TYPES[args.bias_type][1]}_predicted_confidence_diff', hue='group', palette='tab10')

                # Show the plot
                plt.legend(title='Groups')
                plt.show()

                # save the plot
                plt.savefig(f"visualization/{model_type}_{method}_{aggregation}_fairness_vs_reliance_fairness_abs_{FAIRNESS_ABS}_reliance_abs_{RELIANCE_ABS}.png")