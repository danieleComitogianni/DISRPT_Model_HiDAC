# src/utils.py
import torch
import random
import numpy as np
from transformers import Trainer
from sklearn.metrics import accuracy_score, f1_score

def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across all relevant libraries.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class DualInputDataCollator:
    """
    A custom data collator for dual-encoder models.
    It receives a list of dataset examples and correctly pads the two separate
    inputs (input_ids_1, input_ids_2) independently before batching.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        """
        Collates and pads features for a dual-input batch.

        Args:
            features (list[dict]): A list of feature dictionaries from the dataset.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the batched and padded tensors.
        """
        # Separate features for each of the two text inputs
        batch_1_features = [{'input_ids': f['input_ids_1'], 'attention_mask': f['attention_mask_1']} for f in features]
        batch_2_features = [{'input_ids': f['input_ids_2'], 'attention_mask': f['attention_mask_2']} for f in features]

        # Pad each input stream independently to the longest sequence in its respective stream
        batch_1 = self.tokenizer.pad(batch_1_features, padding=True, return_tensors='pt')
        batch_2 = self.tokenizer.pad(batch_2_features, padding=True, return_tensors='pt')

        # Combine the padded inputs and other features (labels, formalism_ids) into a single batch dictionary
        batch = {
            'input_ids_1': batch_1['input_ids'],
            'attention_mask_1': batch_1['attention_mask'],
            'input_ids_2': batch_2['input_ids'],
            'attention_mask_2': batch_2['attention_mask'],
            'labels': torch.tensor([f['labels'] for f in features])
        }
        if 'formalism_ids' in features[0]:
            batch['formalism_ids'] = torch.tensor([f['formalism_ids'] for f in features])
            
        return batch

class HiDACTrainer(Trainer):
    """
    A custom Trainer class that handles the dual-loss objective and warmup schedule for HiDAC.
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Overrides the default compute_loss method to implement the SCL warmup.
        """
        # Dynamically set 'scl_active' based on the current training epoch
        # This ensures the contrastive loss is only computed after the warmup period.
        inputs['scl_active'] = self.state.epoch >= self.model.config.scl_warmup_epochs
        
        # Call the model's forward pass with the potentially modified inputs
        outputs = model(**inputs)
        loss = outputs['loss']

        # Log the individual loss components (CE and SCL) for monitoring purposes
        if self.model.training and self.state.is_local_process_zero and self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
            if outputs.get('ce_loss') is not None: self.log({"ce_loss": outputs['ce_loss'].item()})
            if outputs.get('scl_loss') is not None: self.log({"scl_loss": outputs['scl_loss'].item()})
            
        return (loss, outputs) if return_outputs else loss

def compute_metrics(p):
    """
    Computes and returns accuracy and macro F1-score for evaluation.

    Args:
        p (EvalPrediction): An object containing the model's predictions and the true labels.

    Returns:
        dict[str, float]: A dictionary with the calculated metrics.
    """
    # The predictions are logits, so we take the argmax to get the predicted class index
    preds = np.argmax(p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions, axis=1)
    
    # Calculate metrics using sklearn
    accuracy = accuracy_score(p.label_ids, preds)
    f1_macro = f1_score(p.label_ids, preds, average='macro')
    
    return {"accuracy": accuracy, "f1_macro": f1_macro}