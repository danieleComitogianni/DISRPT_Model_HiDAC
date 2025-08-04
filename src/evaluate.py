# src/evaluate.py
import os
import json
import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

from src.config import Config
from src.model import HiDAC
from src.data_loader import prepare_inference_data

def run_evaluation(output_dir, test_data_path):
    """Loads a trained model and runs evaluation on the test set."""
    model_info_path = os.path.join(output_dir, 'model_info.json')
    adapter_weights_path = os.path.join(output_dir, 'hidac_adapters.pth')

    if not os.path.exists(model_info_path) or not os.path.exists(adapter_weights_path):
        print(f"Error: Could not find model files in '{output_dir}'.")
        print("Please run training first or provide the correct path.")
        return

    with open(model_info_path, 'r') as f:
        model_info = json.load(f)

    # Recreate config from the saved file
    config = Config(**model_info['config'])
    label2id = model_info['label2id']
    id2label = {int(k): v for k, v in model_info['id2label'].items()}
    num_labels = model_info['num_labels']
    formalism2id = model_info.get("formalism2id", {})
    print("Configuration and mappings loaded successfully.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    final_model = HiDAC(config=config, num_labels=num_labels).to(device)
    adapter_weights = torch.load(adapter_weights_path, map_location=device)
    final_model.load_state_dict(adapter_weights, strict=False)
    final_model.eval()
    print("Final HiDAC model rebuilt and adapter weights loaded.")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    test_dataset, test_df = prepare_inference_data(tokenizer, config, test_data_path, label2id, formalism2id)
    print("Test data prepared successfully.")

    all_preds = []
    batch_size = 32
    with torch.no_grad():
        for i in tqdm(range(0, len(test_dataset), batch_size), desc="Running Inference"):
            batch = test_dataset[i : i + batch_size]
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = final_model(**batch, scl_active=False)
            preds = outputs['logits'].argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    test_df['predictions'] = all_preds
    print("Inference complete.")

    print("\n" + "="*50); print("          Final HiDAC Model Performance Results"); print("="*50 + "\n")
    true_labels = test_df['label_id'].tolist()
    accuracy = accuracy_score(true_labels, all_preds)
    f1_macro = f1_score(true_labels, all_preds, average='macro', zero_division=0)

    print(f"**Overall Performance:**\n  - Accuracy: {accuracy:.2%}\n  - F1 Macro: {f1_macro:.4f}\n")
    print("**Performance by Language:**")
    for lang in sorted(test_df['lang'].dropna().unique()):
        lang_df = test_df[test_df['lang'] == lang]
        acc = accuracy_score(lang_df['label_id'], lang_df['predictions'])
        f1 = f1_score(lang_df['label_id'], lang_df['predictions'], average='macro', zero_division=0)
        print(f"  - {lang.upper()}: Accuracy: {acc:.2%}, F1 Macro: {f1:.4f}")

    print("\n**Performance by Framework:**")
    for fw in sorted(test_df['framework'].dropna().unique()):
        fw_df = test_df[test_df['framework'] == fw]
        acc = accuracy_score(fw_df['label_id'], fw_df['predictions'])
        f1 = f1_score(fw_df['label_id'], fw_df['predictions'], average='macro', zero_division=0)
        print(f"  - {fw.upper()}: Accuracy: {acc:.2%}, F1 Macro: {f1:.4f}")
    
    print("\n" + "="*50)
    final_model.encoder.encoder.remove_hooks()
    print("Forward hooks removed.")
