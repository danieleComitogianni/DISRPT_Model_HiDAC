# main.py
import argparse
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from train import run_training
from evaluate import run_evaluation

def main():
    """
    Main function to handle command-line arguments for training and evaluation.
    """
    parser = argparse.ArgumentParser(description="HiDAC Model Training and Evaluation CLI")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- Train command ---
    parser_train = subparsers.add_parser("train", help="Run the model training process.")
    
    # --- Evaluate command ---
    parser_evaluate = subparsers.add_parser("evaluate", help="Run evaluation on a trained model.")
    parser_evaluate.add_argument(
        "--model_dir",
        type=str,
        default="./hidac-final-run",
        help="Directory where the trained model (adapters and info.json) is saved."
    )
    parser_evaluate.add_argument(
        "--test_file",
        type=str,
        default="./data/dev.csv",
        help="Path to the test CSV file for evaluation."
    )
    # --- NEW ARGUMENT ---
    parser_evaluate.add_argument(
        "--predictions_dir",
        type=str,
        default="./predictions",
        help="Directory to save prediction files in DISRPT format. Set to 'None' to disable."
    )

    args = parser.parse_args()

    if args.command == "train":
        print("Starting training process...")
        run_training()
    elif args.command == "evaluate":
        
        # --- Evaluation Model Selection ---
        # By default, this script evaluates the model specified by the --model_dir argument.
        # To evaluate using the provided pre-trained model instead, uncomment the following line.
        # args.model_dir = './pretrained_model'
        
        # Handle case where user wants to disable file generation
        predictions_dir = None if args.predictions_dir.lower() == 'none' else args.predictions_dir
        
        print(f"Starting evaluation from model in '{args.model_dir}'...")
        run_evaluation(
            output_dir=args.model_dir, 
            test_data_path=args.test_file,
            predictions_dir=predictions_dir # Pass the new argument
        )

if __name__ == "__main__":
    main()
