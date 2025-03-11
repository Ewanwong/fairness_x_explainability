from utils.pointing_game_utils import GridPointingGame
from utils.Explainer import BcosExplainer, AttentionExplainer, GradientNPropabationExplainer, OcclusionExplainer, ShapleyValueExplainer, LimeExplainer
from argparse import ArgumentParser
import os

def main(args):


    # convert strings to numbers
    args.num_labels = int(args.num_labels) if args.num_labels else None
    args.batch_size = int(args.batch_size) if args.batch_size else None
    args.max_length = int(args.max_length) if args.max_length else None
    args.num_examples = int(args.num_examples) if args.num_examples else None
    args.seed = int(args.seed) if args.seed else None
    args.split_ratio = float(args.split_ratio) if args.split_ratio else None
    
    pointing_game = GridPointingGame(
        model_name_or_path=args.model_dir,
        dataset=args.dataset_name,
        num_labels=args.num_labels,
        split=args.split,
        split_ratio=args.split_ratio,
        load_pointing_game_examples_path=args.load_pointing_game_examples_path,
        save_pointing_game_examples_path=args.save_pointing_game_examples_path,
        num_segments=2,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_instances=args.num_examples,
        min_confidence=0.75,
        random_seed=args.seed,
    )


if __name__ == '__main__':
    parser = ArgumentParser(description='Evaluate the faithfulness for rationales using pointing game methods.')

    parser.add_argument('--dataset_name', type=str, default='stanfordnlp/imdb', help='Name of the HuggingFace dataset to use') #fancyzhx/ag_news, stanfordnlp/imdb
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use (e.g., train, test)')
    parser.add_argument('--split_ratio', type=float, default=0.5, help='Split ratio for test dataset')
    parser.add_argument('--model_dir', type=str, default='models/bert_base_imdb_512', help='Name of the pre-trained model')
    parser.add_argument('--load_pointing_game_examples_path', type=str, default=None, help='Path to load pointing game examples')
    parser.add_argument('--save_pointing_game_examples_path', type=str, default=None, help='Path to save pointing game examples')
    parser.add_argument('--num_labels', type=int, default=2, help='Number of labels in the classification')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length for tokenization')
    parser.add_argument('--num_examples', type=int, default=-1, help='Number of examples to process (-1 for all)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')


    args = parser.parse_args()
    main(args)