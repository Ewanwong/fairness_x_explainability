import argparse
import json
import logging
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from transformers import (AutoTokenizer, AutoConfig, AdamW,
                          get_linear_schedule_with_warmup)
from bcos_lm.models.modeling_bert import BertForSequenceClassification
from bcos_lm.models.modeling_roberta import RobertaForSequenceClassification
from bcos_lm.models.modeling_distilbert import DistilBertForSequenceClassification
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import random
import os
from tqdm import tqdm

def main():
    # Argument parser for hyperparameters
    parser = argparse.ArgumentParser(description="Fine-tune BERT for sequence classification")

    # Hyperparameters
    parser.add_argument('--model_name_or_path', type=str, default='bert-base-uncased',
                        help='Pre-trained model name or path')
    parser.add_argument('--dataset_name', type=str, default='fancyzhx/ag_news',
                        help='Dataset name (default: ag_news)')
    parser.add_argument('--num_labels', type=int, default=4,
                        help='Number of labels in the dataset')
    parser.add_argument('--output_dir', type=str, default='/local/yifwang/bcos_bert_base_agnews_512',
                        help='Directory to save the model')
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='Maximum input sequence length after tokenization')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                        help='Learning rate for the optimizer')
    parser.add_argument('--warmup_steps_or_ratio', type=float, default=0.1,
                        help='Number or ratio of warmup steps for the learning rate scheduler')
    parser.add_argument('--num_train_epochs', type=int, default=10,
                        help='Total number of training epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for initialization')
    parser.add_argument('--early_stopping_patience', type=int, default=-1,
                        help='Number of epochs with no improvement after which training will be stopped')
    parser.add_argument('--log_file', type=str, default='training.log',
                        help='Path to the log file')
    parser.add_argument('--eval_steps', type=int, default=1000,
                        help='Evaluate the model every X training steps')
    parser.add_argument('--save_steps', type=int, default=1000,
                        help='Save the model every X training steps')
    parser.add_argument('--split_ratio', type=str, default="0.8, 0.2",
                    help='Ratio to split the train set into train and validation sets')
    parser.add_argument('--b', type=float, default=2.0,)
    parser.add_argument('--bcos', action='store_true', help='Use BCOS')
    parser.add_argument('--bce', action='store_true', help='Use bce loss instead of cross entropy loss')
    parser.add_argument('--different_b_per_layer', action='store_true', help='Use different b per layer')
    parser.add_argument('--b_list', type=str, default="1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.0", help='List of b values for different layers')


    args = parser.parse_args()

    # create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    log_file = os.path.join(args.output_dir, args.log_file)

    # Set up logging
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    # Log the hyperparameters
    logging.info("Hyperparameters:")
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")

    # Set up the device for GPU usage if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Set seeds for reproducibility
    seed_val = args.seed

    def set_random_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_random_seed(seed_val)


    # Load the dataset
    logging.info(f"Loading {args.dataset_name} dataset...")
    if "civil_comments" in args.dataset_name:
        dataset = load_dataset(args.dataset_name, "all")
    else:
        dataset = load_dataset(args.dataset_name)
    if "civil_comments" in args.dataset_name:
        dataset = dataset.map(lambda example: {"label": 1 if example['sub_split'] == 'toxic' else 0}, 
                      keep_in_memory=True)
        dataset = dataset.remove_columns(['sub_split', 'gold'])
    elif "sst2" in args.dataset_name:
        dataset = dataset.rename_column('sentence', 'text')
        dataset = dataset.remove_columns(['idx'])
    elif "bias_in_bios" in args.dataset_name:
        dataset = dataset.rename_column('profession', 'label')
        dataset = dataset.rename_column("hard_text", "text")
    else:
        raise ValueError("Dataset not supported")

    # lowercase all examples
    dataset = dataset.map(lambda example: {"text": example["text"].lower()})

    # Initialize the tokenizer and model
    if "distilbert" in args.model_name_or_path.lower():
        Model = DistilBertForSequenceClassification
    elif "roberta" in args.model_name_or_path.lower():
        Model = RobertaForSequenceClassification
    elif "bert" in args.model_name_or_path.lower():
        Model = BertForSequenceClassification
    else:
        raise ValueError("Model not supported")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=args.num_labels)
    config.num_labels = args.num_labels
    config.bcos = args.bcos
    config.b = args.b
    config.bce = args.bce
    config.different_b_per_layer = args.different_b_per_layer
    b_list = [float(b) for b in args.b_list.strip().split(",")] if args.different_b_per_layer else [args.b] * config.num_hidden_layers
    assert len(b_list) == config.num_hidden_layers if hasattr(config, "num_hidden_layers") else config.n_layers, "Length of b_list should be equal to the number of hidden layers"
    config.b_list = b_list

    model = Model.load_from_pretrained(args.model_name_or_path, config=config)
    model.to(device)

    # Tokenization function
    def tokenize_function(examples):

        return tokenizer(examples['text'],
                        padding='max_length',
                        truncation=True,
                        max_length=args.max_seq_length)

    # Apply tokenization to the datasets
    logging.info("Tokenizing datasets...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Set the format of the datasets to PyTorch tensors
    if "bias_in_bios" in args.dataset_name:
        tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label', 'gender'])
    else:
        tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    # change the column "label" to "labels" to match the model's forward function
    tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')


    # Prepare data loaders
    split_ratio = [float(r) for r in args.split_ratio.strip().split(",")]
    if "sst2" not in args.dataset_name:
        train_dataset = tokenized_datasets['train']
        test_dataset = tokenized_datasets['test']
        if 'val' in tokenized_datasets:
            val_dataset = tokenized_datasets['val']
        elif "validation" in tokenized_datasets:
            val_dataset = tokenized_datasets['validation']
        elif "dev" in tokenized_datasets:
            val_dataset = tokenized_datasets['dev']
        else:
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
        train_dataset = tokenized_datasets['train']
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

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.batch_size)
    validation_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args.batch_size)

    # Initialize the optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    total_steps = len(train_dataloader) * args.num_train_epochs
    if args.warmup_steps_or_ratio > 1.0:
        warmup_steps = args.warmup_steps_or_ratio
    else:
        warmup_steps = int(total_steps * args.warmup_steps_or_ratio)

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    # Accuracy evaluation function
    def evaluate(model, dataloader, average='macro'):
        model.eval()
        predictions, true_labels = [], []

        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = model(input_ids=batch['input_ids'],
                                attention_mask=batch['attention_mask'],
                                labels=batch['labels'])

            logits = outputs.logits.detach().cpu().numpy()
            label_ids = batch['labels'].to('cpu').numpy()

            predictions.extend(np.argmax(logits, axis=1))
            true_labels.extend(label_ids)

        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average=average)
        model.train()
        return accuracy, f1

    # Early stopping parameters
    early_stopping_patience = args.early_stopping_patience if args.early_stopping_patience != -1 else np.inf
    best_accuracy = 0.0
    best_f1 = 0.0
    evaluations_no_improve = 0
    global_step = 0

    # Training loop
    for epoch_i in range(args.num_train_epochs):
        logging.info(f"\n======== Epoch {epoch_i + 1} / {args.num_train_epochs} ========")
        logging.info("Training...")
        total_loss = 0
        model.train()

        for step, batch in tqdm(enumerate(train_dataloader)):
            
            global_step += 1
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            outputs = model(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['labels'])
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()

            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Evaluate the model at specified steps
            if args.eval_steps and global_step % args.eval_steps == 0:
                logging.info(f"\nStep {global_step}: running evaluation...")
                val_accuracy, val_f1 = evaluate(model, validation_dataloader)
                logging.info(f"Validation Accuracy at step {global_step}: {val_accuracy:.4f}")
                logging.info(f"Validation F1 Score at step {global_step}: {val_f1:.4f}")

                # Check for early stopping
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                #if val_f1 > best_f1:
                #    best_f1 = val_f1
                    evaluations_no_improve = 0

                    # Save the best model using Hugging Face's save_pretrained
                    model.save_pretrained(args.output_dir)
                    tokenizer.save_pretrained(args.output_dir)
                    logging.info(f"Best model saved to {args.output_dir}")
                else:
                    evaluations_no_improve += 1
                    logging.info(f"No improvement in validation accuracy for {evaluations_no_improve} evaluation(s).")
                    if evaluations_no_improve >= early_stopping_patience:
                        logging.info("Early stopping triggered.")
                        break

            # Save the model at specified steps
            if args.save_steps and global_step % args.save_steps == 0:
                checkpoint_dir = os.path.join(args.output_dir, f'checkpoint-{global_step}')
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                logging.info(f"Model checkpoint saved at step {global_step} to {checkpoint_dir}")

        avg_train_loss = total_loss / len(train_dataloader)
        logging.info(f"Average training loss for epoch {epoch_i + 1}: {avg_train_loss:.4f}")

        # Evaluate at the end of each epoch if eval_steps is not set
        if not args.eval_steps:
            logging.info("Running Validation at the end of the epoch...")
            val_accuracy, val_f1 = evaluate(model, validation_dataloader)
            logging.info(f"Validation Accuracy: {val_accuracy:.4f}")
            logging.info(f"Validation F1 Score: {val_f1:.4f}")

            # Check for early stopping
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
            #if val_f1 > best_f1:
            #    best_f1 = val_f1
                evaluations_no_improve = 0

                # Save the best model using Hugging Face's save_pretrained
                model.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                logging.info(f"Best model saved to {args.output_dir}")
            else:
                evaluations_no_improve += 1
                logging.info(f"No improvement in validation accuracy for {evaluations_no_improve} epoch(s).")
                if evaluations_no_improve >= early_stopping_patience:
                    logging.info("Early stopping triggered.")
                    break

        # Save the model at the end of each epoch if save_steps is not set
        if not args.save_steps:
            checkpoint_dir = os.path.join(args.output_dir, f'checkpoint-epoch-{epoch_i + 1}')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            logging.info(f"Model checkpoint saved at the end of epoch {epoch_i + 1} to {checkpoint_dir}")

        # Break the outer loop if early stopping is triggered during evaluation steps
        if evaluations_no_improve >= early_stopping_patience:
            break

    # Load the best model
    model = Model.load_from_pretrained(args.output_dir, config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    model.to(device)

    # Test evaluation
    logging.info("\nRunning Test Evaluation...")
    test_accuracy, test_f1 = evaluate(model, test_dataloader)
    logging.info(f"Test Accuracy: {test_accuracy:.4f}")
    logging.info(f"Test F1 Score: {test_f1:.4f}")

if __name__ == '__main__':
    main()