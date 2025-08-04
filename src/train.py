# src/train.py
import os
import json
import torch
import transformers
from dataclasses import asdict
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback
)

# Local module imports
from src.config import Config
from src.model import HiDAC
from src.utils import set_seed, DualInputDataCollator, HiDACTrainer, compute_metrics
from src.data_loader import prepare_datasets
import warnings


def run_training():
    """
    Initializes and runs the full training pipeline for the HiDAC model.

    This function handles:
    - Configuration setup and seeding for reproducibility.
    - Data loading, tokenization, and preparation.
    - Initialization of the Trainer with the HiDAC model.
    - Execution of the training and final evaluation loops.
    - Saving the trained model adapters and final configuration.
    """
    # Suppress unnecessary warnings for cleaner output
    warnings.filterwarnings('ignore')
    transformers.logging.set_verbosity_error()

    # Initialize configuration and set random seed
    config = Config()
    set_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Data Preparation ---
    print("--- Loading and preparing datasets ---")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_dataset, val_dataset, label2id, id2label, formalism2id = prepare_datasets(tokenizer, config)
    print(f"\nTraining with dataset size: {len(train_dataset)}")

    # --- Training Arguments ---
    # Configure the training process using Hugging Face's TrainingArguments
    final_args = TrainingArguments(
        output_dir="./hidac-final-run",
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=50,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        save_total_limit=1,
        bf16=False,
        torch_compile=False,
        report_to="none",
        seed=config.seed,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_strategy="steps",
        logging_steps=50,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine"
    )

    # --- Model Initialization ---
    # Use a model_init function to ensure a fresh model is created for training
    def final_model_init():
        """Initializes a fresh HiDAC model for the Trainer."""
        return HiDAC(config=config, num_labels=len(label2id)).to(device)

    # --- Trainer Setup ---
    final_trainer = HiDACTrainer(
        model_init=final_model_init,
        args=final_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DualInputDataCollator(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # --- Execute Training ---
    print("\n--- Starting Final Training of HiDAC Model ---")
    final_trainer.train()
    final_metrics = final_trainer.evaluate()
    print(f"Final metrics: {final_metrics}")

    # --- Cleanup and Save ---
    # Remove forward hooks to prepare the model for saving and inference
    final_trainer.model.encoder.encoder.remove_hooks()

    print("\n--- Saving Model and Configuration ---")
    output_dir = final_trainer.args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Save only the trainable adapter weights for efficient storage
    trainable_params = {k: v.cpu() for k, v in final_trainer.model.named_parameters() if v.requires_grad}
    adapter_file_path = os.path.join(output_dir, "hidac_adapters.pth")
    torch.save(trainable_params, adapter_file_path)
    print(f"Adapter weights saved to: {adapter_file_path}")

    # Save all configuration and results to a JSON file for reproducibility
    config_dict = asdict(config)
    model_info = {
        "label2id": label2id, 
        "id2label": id2label, 
        "config": config_dict,
        "num_labels": len(label2id), 
        "unique_labels": list(label2id.keys()),
        "final_metrics": final_metrics
    }
    if formalism2id:
        model_info["formalism2id"] = formalism2id
        model_info["id2formalism"] = {v: k for k, v in formalism2id.items()}

    with open(os.path.join(output_dir, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)
    print(f"Model info saved to: {os.path.join(output_dir, 'model_info.json')}")
    
    print(f"\nTraining Complete. Final F1 Macro: {final_metrics.get('eval_f1_macro', 'N/A'):.4f}")
