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

def generate_disrpt_output_files(df, id2label, test_data_path, output_dir):
    """Generates prediction files compatible with the official DISRPT script."""
    print(f"\nGenerating prediction files in '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)

    # Map predicted IDs to their string labels
    df['predicted_label'] = df['predictions'].map(id2label)

    # Determine the split name ('dev' or 'test') from the input file name
    split_name = 'dev' if 'dev' in os.path.basename(test_data_path).lower() else 'test'

    output_columns = [
        'doc', 'unit1_toks', 'unit2_toks', 'unit1_txt', 'unit2_txt',
        'unit1_raw_text', 'unit2_raw_text', 's1_toks', 's2_toks',
        'unit1_sent', 'unit2_sent', 'dir', 'type',
        'orig_label',  
        'label_text'   
    ]

    for dataset_name in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset_name].copy()
        
        # Sort by the original row order to match the gold file
        dataset_df.sort_values(by='row_in_file', inplace=True)

        # Replace original labels with predictions
        dataset_df['orig_label'] = dataset_df['predicted_label']
        dataset_df['label_text'] = dataset_df['predicted_label']

        # Create dataset-specific directory
        dataset_output_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)

        # Construct file name and path
        output_filename = f"{dataset_name}_{split_name}.rels"
        output_filepath = os.path.join(dataset_output_dir, output_filename)

        # Write to file
        dataset_df.to_csv(
            output_filepath,
            sep='\t',
            header=True, 
            index=False,
            columns=output_columns
        )
        print(f"  ✅ Saved predictions for {dataset_name} to {output_filepath}")

def run_evaluation(output_dir, test_data_path, predictions_dir=None):
    """Loads a trained model and runs evaluation on the test set."""
    model_info_path = os.path.join(output_dir, 'model_info.json')
    adapter_weights_path = os.path.join(output_dir, 'hidac_adapters.pth')

    if not os.path.exists(model_info_path) or not os.path.exists(adapter_weights_path):
        print(f"Error: Could not find model files in '{output_dir}'.")
        print("Please run training first or provide the correct path.")
        return

    with open(model_info_path, 'r') as f:
        model_info = json.load(f)

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

    print("\n" + "="*50); print("           Final HiDAC Model Performance Results"); print("="*50 + "\n")
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

    print("\n**Performance by Dataset:**")
    for dataset in sorted(test_df['dataset'].dropna().unique()):
        dataset_df = test_df[test_df['dataset'] == dataset]
        if not dataset_df.empty:
            acc = accuracy_score(dataset_df['label_id'], dataset_df['predictions'])
            f1 = f1_score(dataset_df['label_id'], dataset_df['predictions'], average='macro', zero_division=0)
            print(f"  - {dataset}: Accuracy: {acc:.2%}, F1 Macro: {f1:.4f}")

    print("\n" + "="*50)

    if predictions_dir:
        generate_disrpt_output_files(test_df, id2label, test_data_path, predictions_dir)

    final_model.encoder.encoder.remove_hooks()
    print("\nForward hooks removed.")
