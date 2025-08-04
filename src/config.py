# src/config.py
from dataclasses import dataclass

@dataclass
class Config:
    """
    Configuration class for the HiDAC model, training, and data paths.
    """
    # --- Core Model & Data Configuration ---
    model_name: str = 'xlm-roberta-large'
    max_length: int = 256
    train_data_path: str = './data/train.csv'
    val_data_path: str = './data/dev.csv'
    seed: int = 42

    # --- LoRA Configuration ---
    # Parameters for the Low-Rank Adaptation modules.
    lora_rank: int = 128          # The rank of the LoRA matrices.
    lora_alpha: int = 256         # The scaling factor for LoRA.
    lora_dropout: float = 0.1     # Dropout probability for LoRA layers.

    # --- Mixture-of-Experts (MoE) Configuration ---
    num_formalisms: int = 6       # Number of specialist experts.
    expert_start_layer: int = 12  # The first layer to use MoE-LoRA adapters.

    # --- Contrastive Learning Configuration ---
    # Hyperparameters for the Label-Centered Contrastive Loss.
    temperature: float = 0.1      # The temperature scaling factor for the softmax in SCL.
    projection_dim: int = 256     # The dimensionality of the projection head output.
    scl_loss_weight: float = 0.3  # The weight for the SCL loss component (lambda_cl).
    scl_warmup_epochs: int = 2    # Number of epochs to train with only CE loss before activating SCL.

    # --- Cross-Entropy Loss Configuration ---
    ce_loss_weight: float = 1.0   # The weight for the CE loss component (lambda_ce).