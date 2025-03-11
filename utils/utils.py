import random
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
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
    metrics_dict = {}
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')

    metrics_dict["accuracy"] = accuracy
    metrics_dict["f1"] = f1

    cm = confusion_matrix(labels, predictions, labels=range(num_classes))

    total_samples = np.sum(cm)

    for i in range(num_classes):
        # True Positives
        TP = cm[i, i]
        # False Negatives
        FN = np.sum(cm[i, :]) - TP
        # False Positives
        FP = np.sum(cm[:, i]) - TP
        # True Negatives
        TN = total_samples - (TP + FP + FN)

        # TPR (Sensitivity, Recall) = TP / (TP + FN)
        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        # FPR = FP / (FP + TN)
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        # TNR = TN / (TN + FP)
        tnr = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        # FNR = FN / (FN + TP)
        fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0

        metrics_dict[f"class_{i}"] = {"tpr": tpr, "fpr": fpr, "tnr": tnr, "fnr": fnr}
    
    return metrics_dict

# TODO: Implement perturbation functions
def perturb_example(text, perturbation_list):
    perturbed_text = text
    for orig, perturb in perturbation_list:
        perturbed_text = perturbed_text.replace(orig, perturb)
    return perturbed_text

def apply_dataset_perturbation(dataset, perturbation_list):
    perturbed_dataset = dataset.copy()
    perturbed_texts = []
    for text in perturbed_dataset['text']:
        perturbed_texts.append(perturb_example(text, perturbation_list))    
    perturbed_dataset['text'] = perturbed_texts
    return perturbed_dataset

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


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensuring deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split_dataset(dataset, split_ratio):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(split_ratio * dataset_size))
    np.random.shuffle(indices)

    val_indices, test_indices = indices[:split], indices[split:]
    dataset = Subset(dataset, test_indices)
    return dataset

def batch_loader(dataset, batch_size, shuffle=False):
    if type(dataset) == dict:
        # get the length of the dataset
        length = len(dataset[list(dataset.keys())[0]])
        indices = list(range(length))
        if shuffle:
            random.shuffle(indices)
        for i in range(0, length, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch = {key: [dataset[key][j] for j in batch_indices] for key in dataset.keys()}
            yield batch
    elif type(dataset) == list:
        length = len(dataset)
        indices = list(range(length))
        if shuffle:
            random.shuffle(indices)
        for i in range(0, length, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch = [dataset[j] for j in batch_indices]
            yield batch


