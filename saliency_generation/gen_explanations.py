from saliency_utils.Explainer import BcosExplainer, AttentionExplainer, GradientNPropabationExplainer, OcclusionExplainer, ShapleyValueExplainer, LimeExplainer
from saliency_utils.utils import set_random_seed, split_dataset, apply_dataset_perturbation
import argparse
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoConfig
from bcos_lm.models.modeling_bert import BertForSequenceClassification
from bcos_lm.models.modeling_roberta import RobertaForSequenceClassification
from bcos_lm.models.modeling_distilbert import DistilBertForSequenceClassification
from datasets import load_dataset
import numpy as np
import json
import os
import random
from tqdm import tqdm

EXPLANATION_METHODS = {
    "Bcos": BcosExplainer,
    "Attention": AttentionExplainer,
    "Saliency": GradientNPropabationExplainer,
    "DeepLift": GradientNPropabationExplainer,
    "GuidedBackprop": GradientNPropabationExplainer,
    "InputXGradient": GradientNPropabationExplainer,
    "IntegratedGradients": GradientNPropabationExplainer,
    "SIG": GradientNPropabationExplainer,
    "Occlusion": OcclusionExplainer,
    "ShapleyValue": ShapleyValueExplainer,
    "KernelShap": ShapleyValueExplainer,
    "Lime": LimeExplainer,
}

BIAS_TYPES = {
    "gender": ["female", "male"],
    "race": ["black", "white"],
}

# TODO: Implement perturbation list
PERTURBATION_LIST = {
    "gender": [],
    "race": [],
}


def main(args):

    # convert strings to numbers
    args.num_labels = int(args.num_labels) if args.num_labels else None
    args.batch_size = int(args.batch_size) if args.batch_size else None
    args.max_length = int(args.max_length) if args.max_length else None
    args.num_examples = int(args.num_examples) if args.num_examples else None
    args.seed = int(args.seed) if args.seed else None
    args.shap_n_samples = int(args.shap_n_samples) if args.shap_n_samples else None


    # Set random seed for reproducibility
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load pre-trained BERT model and tokenizer
    if "distilbert" in args.model_dir.lower():
        Model = DistilBertForSequenceClassification
    elif "roberta" in args.model_dir.lower():
        Model = RobertaForSequenceClassification
    elif "bert" in args.model_dir.lower():
        Model = BertForSequenceClassification
    config = AutoConfig.from_pretrained(args.model_dir, num_labels=args.num_labels)
    #config.bcos = args.bcos
    #config.b = args.b

    config.output_attentions = True
    config.num_labels = args.num_labels
    #print(config)
    model = Model.load_from_pretrained(args.model_dir, config=config, output_attentions=True)
    model.eval()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    # Load a dataset from HuggingFace datasets library
    print("Loading dataset...")
    test_datasets = {}
    if "civil_comments" in args.dataset_name:
        for group in BIAS_TYPES[args.bias_type]:
            test_datasets[group] = load_dataset(args.dataset_name, group)
            test_datasets[group] = test_datasets[group].map(lambda example: {"label": 1 if example['sub_split'] == 'toxic' else 0},
                      keep_in_memory=True)
            test_datasets[group] = test_datasets[group].remove_columns(['sub_split', 'gold'])
            test_datasets[group] = test_datasets[group].map(lambda example: {"text": example["text"].lower()})
    elif "bias_in_bios" in args.dataset_name:
        assert args.bias_type == "gender", "bias_in_bios dataset only supports study of gender bias"
        dataset = load_dataset(args.dataset_name)
        for group in BIAS_TYPES[args.bias_type]:
            if group == "male":
                test_datasets[group] = dataset.filter(lambda example: example["gender"]==0)
            elif group == "female":
                test_datasets[group] = dataset.filter(lambda example: example["gender"]==1)
            else:
                raise ValueError("Invalid gender group for bias_in_bios dataset")
            test_datasets[group] = test_datasets[group].rename_column('profession', 'label')
            test_datasets[group] = test_datasets[group].rename_column("hard_text", "text")
            test_datasets[group] = test_datasets[group].map(lambda example: {"text": example["text"].lower()})
        
    elif "sst2" in args.dataset_name:
        assert args.bias_type == "gender", "sst-2 dataset only supports study of gender bias"
        dataset = load_dataset(args.dataset_name)
        # TODO: filter the dataset
        for group in BIAS_TYPES[args.bias_type]:
            if group == "male":
                test_datasets[group] = dataset.filter(lambda example: example["gender"]==0)
            elif group == "female":
                test_datasets[group] = dataset.filter(lambda example: example["gender"]==1)
            else:
                raise ValueError("Invalid gender group for bias_in_bios dataset")
            test_datasets[group] = test_datasets[group].rename_column('sentence', 'text')
            test_datasets[group] = test_datasets[group].rename_column(['idx'])
            test_datasets[group] = test_datasets[group].map(lambda example: {"text": example["text"].lower()})
    
    else:
        raise ValueError("Invalid dataset name")

    split_ratio = [float(r) for r in args.split_ratio.strip().split(",")]
    for group in test_datasets.keys():
        if "sst2" not in args.dataset_name:
            train_dataset = test_datasets[group]['train']
            test_dataset = test_datasets[group]['test']
            if 'val' in test_datasets[group]:
                val_dataset = test_datasets[group]['val']
            elif "validation" in test_datasets[group]:
                val_dataset = test_datasets[group]['validation']
            elif "dev" in test_datasets[group]:
                val_dataset = test_datasets[group]['dev']
            else:
                # TODO: problem when loading val set for a group data, now could use the train set during training for validation, consider filtering out val set from train-time val set?
                # Split the train dataset into train and validation sets
                train_dataset_size = len(train_dataset)
                indices = list(range(train_dataset_size))
                if len(split_ratio) == 1:
                    split = int(np.floor(split_ratio[0] * train_dataset_size))
                elif len(split_ratio) == 2:
                    ratio = split_ratio[0] / (split_ratio[0] + split_ratio[1])
                    split = int(np.floor(ratio * train_dataset_size))
                else:
                    raise ValueError("Invalid split ratio")
                np.random.shuffle(indices)

                train_indices, val_indices = indices[:split], indices[split:]

                val_dataset = Subset(train_dataset, val_indices)
                train_dataset = Subset(train_dataset, train_indices)

        else:
            if len(split_ratio) != 3:
                print("Split ratio should be in the format of train, val, test; use default split ratio of 4:3:3 instead")
                split_ratio = [4, 3, 3]
            train_dataset = test_datasets[group]['train']
            train_dataset_size = len(train_dataset)
            indices = list(range(train_dataset_size))
            ratio1 = split_ratio[0] / sum(split_ratio)
            ratio2 = split_ratio[1] / sum(split_ratio)
            split1 = int(np.floor(ratio1 * train_dataset_size))
            split2 = int(np.floor((ratio1 + ratio2) * train_dataset_size))
            np.random.shuffle(indices)

            train_indices, val_indices, test_indices = indices[:split1], indices[split1:split2], indices[split2:]

            test_dataset = Subset(train_dataset, test_indices)
            val_dataset = Subset(train_dataset, val_indices)
            train_dataset = Subset(train_dataset, train_indices)
        
        test_datasets[group] = {}
        test_datasets[group]['train'] = train_dataset
        test_datasets[group]['val'] = val_dataset
        test_datasets[group]['test'] = test_dataset


    # Initialize the explainer
    print("Running attribution methods...")
    all_methods = EXPLANATION_METHODS.keys()
    if args.methods:
        attribution_methods = args.methods.replace(' ', '').split(',')   
    else:
        attribution_methods = all_methods  # Use all methods if none specified


    # Create output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for method in attribution_methods:
        print(f"\nRunning {method} explainer...")
        if EXPLANATION_METHODS[method] == BcosExplainer:
            explainer = BcosExplainer(model, tokenizer, args.relative)
        elif EXPLANATION_METHODS[method] == ShapleyValueExplainer:
            explainer = ShapleyValueExplainer(model, tokenizer, method, args.baseline, args.shap_n_samples)
        # for GradientNPropabationExplainer, we need to specify the method
        elif EXPLANATION_METHODS[method] == GradientNPropabationExplainer:
            explainer = EXPLANATION_METHODS[method](model, tokenizer, method, args.baseline)
        else:
            explainer = EXPLANATION_METHODS[method](model, tokenizer) 

        # can only explain the label class to reduce the computation time
        #class_labels = [dataset['label']]
        #explanation_results = explainer.explain_dataset(dataset, num_classes=args.num_labels, class_labels=class_labels, batch_size=args.batch_size, max_length=args.max_length)

        for group in test_datasets.keys():
            test_dataset = test_datasets[group][args.split]
            test_dataset = test_dataset[:] if args.num_examples == -1 else test_dataset[:args.num_examples]
            test_dataset['index'] = list(range(len(test_dataset['text'])))

            explanation_results = explainer.explain_dataset(test_dataset, num_classes=args.num_labels, batch_size=args.batch_size, max_length=args.max_length, only_predicted_classes=args.only_predicted_classes)
            result = explanation_results

            # Save the results to a JSON file
            output_file = os.path.join(args.output_dir, f'{method}_{group}_{args.split}_explanations.json')
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=4)
            print(f"\nAttribution results saved to {output_file}")

            # TODO: apply counterfactual perturbation and explain
            if args.counterfactual:
                if group == "black":
                    perturbation_list = [("black", "white"), ["african", "european"], ["africa", "europe"]]
                elif group == "white":
                    perturbation_list = [("white", "black"), ["european", "african"], ["europe", "africa"], ["caucasian", "african"]]
                else:
                    raise ValueError("Invalid group for counterfactual perturbation")
                perturbed_test_dataset = apply_dataset_perturbation(test_dataset, perturbation_list)
                perturbed_explanation_results = explainer.explain_dataset(perturbed_test_dataset, num_classes=args.num_labels, batch_size=args.batch_size, max_length=args.max_length, only_predicted_classes=args.only_predicted_classes)
                perturbed_result = perturbed_explanation_results

                counterfactual_output_file = os.path.join(args.output_dir, f'{method}_{group}_counterfactual_{args.split}_explanations.json')
                with open(counterfactual_output_file, 'w') as f:
                    json.dump(perturbed_result, f, indent=4)
                print(f"\nCounterfactual attribution results saved to {counterfactual_output_file}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BERT Attribution with Captum')

    parser.add_argument('--dataset_name', type=str, default='stanfordnlp/imdb', help='Name of the HuggingFace dataset to use') #fancyzhx/ag_news, stanfordnlp/imdb
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use (e.g., train, test)')
    parser.add_argument('--split_ratio', type=str, default="0.8, 0.2",
                    help='Ratio to split the train set into train and validation sets')
    parser.add_argument('--model_dir', type=str, default='models/bert_base_imdb_512', help='Name of the pre-trained model')
    parser.add_argument('--num_labels', type=int, default=2, help='Number of labels in the classification')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length for tokenization')
    parser.add_argument('--baseline', type=str, default='pad', help='Baseline for the attribution methods, select from zero, mask, pad')    
    parser.add_argument('--num_examples', type=int, default=1000, help='Number of examples to process (-1 for all)')
    parser.add_argument('--methods', type=str, default=None, help='List of attribution methods to use separated by commas')
    parser.add_argument('--output_dir', type=str, default='baseline_saliency_results/all_methods_1000_examples_512', help='Directory to save the output files')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--shap_n_samples', type=int, default=25, help='Number of samples for Shapley Value Sampling')
    parser.add_argument('--only_predicted_classes', action='store_true', help='Only explain the predicted class')
    parser.add_argument('--relative', action='store_true', help='explain relative logits')
    parser.add_argument('--bias_type', type=str, default='gender', help='Bias type to explain')
    parser.add_argument('--counterfactual', action='store_true', help='Apply counterfactual perturbation')

    args = parser.parse_args()
    main(args)
