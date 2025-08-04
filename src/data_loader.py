# src/data_loader.py
import pandas as pd
from datasets import Dataset

def prepare_datasets(tokenizer, config):
    """
    Loads, processes, and tokenizes the training and validation datasets.

    This function reads the raw CSV files, maps the string labels and frameworks
    to integer IDs, and tokenizes the text pairs for the dual-encoder model.

    Args:
        tokenizer: The Hugging Face tokenizer to use.
        config (Config): The configuration object containing data paths and settings.

    Returns:
        tuple: A tuple containing:
            - train_dataset (Dataset): The tokenized training dataset.
            - val_dataset (Dataset): The tokenized validation dataset.
            - label2id (dict): A mapping from string labels to integer IDs.
            - id2label (dict): A mapping from integer IDs to string labels.
            - formalism2id (dict or None): A mapping from framework strings to IDs.
    """
    # =========================================================================
    # --- FOR FINAL SUBMISSION (Loads the full dataset) ---
    # =========================================================================
    # # Load raw data from CSV files
    # train_df = pd.read_csv(config.train_data_path)
    # val_df = pd.read_csv(config.val_data_path)
    #
    # # Create label mappings based on the training data
    # unique_labels = sorted(train_df['label'].unique())
    # label2id = {label: i for i, label in enumerate(unique_labels)}
    # id2label = {i: label for i, label in enumerate(unique_labels)}
    # train_df['label_id'] = train_df['label'].map(label2id)
    # val_df['label_id'] = val_df['label'].map(label2id)
    #
    # formalism2id = None
    # # Create formalism mappings if the 'framework' column exists
    # if 'framework' in train_df.columns:
    #     unique_formalisms = sorted(train_df['framework'].unique())
    #     formalism2id = {form: i for i, form in enumerate(unique_formalisms)}
    #     train_df['formalism_id'] = train_df['framework'].map(formalism2id)
    #     val_df['formalism_id'] = val_df['framework'].map(formalism2id)
    #     config.num_formalisms = len(unique_formalisms) # Dynamically update config
    #     print(f"Found {config.num_formalisms} formalisms: {unique_formalisms}")
    # else:
    #     # If no framework is specified, assign a default ID of 0
    #     train_df['formalism_id'] = 0
    #     val_df['formalism_id'] = 0
    # =========================================================================


    # =========================================================================
    # --- FOR LOCAL TESTING (Loads only 1k samples) ---
    # =========================================================================
    # Load a small subset of the raw data from CSV files for quick local testing
    train_df = pd.read_csv(config.train_data_path).head(1000)
    val_df = pd.read_csv(config.val_data_path).head(1000)

    # Create label mappings based on the full training data to ensure consistency
    full_train_df = pd.read_csv(config.train_data_path)
    unique_labels = sorted(full_train_df['label'].unique())
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}
    train_df['label_id'] = train_df['label'].map(label2id)
    val_df['label_id'] = val_df['label'].map(label2id)

    formalism2id = None
    # Create formalism mappings if the 'framework' column exists
    if 'framework' in full_train_df.columns:
        unique_formalisms = sorted(full_train_df['framework'].unique())
        formalism2id = {form: i for i, form in enumerate(unique_formalisms)}
        train_df['formalism_id'] = train_df['framework'].map(formalism2id)
        val_df['formalism_id'] = val_df['framework'].map(formalism2id)
        config.num_formalisms = len(unique_formalisms) # Dynamically update config
        print(f"Found {config.num_formalisms} formalisms: {unique_formalisms}")
    else:
        # If no framework is specified, assign a default ID of 0
        train_df['formalism_id'] = 0
        val_df['formalism_id'] = 0
    # =========================================================================


    def tokenize_function_dual(examples):
        """Tokenizes a batch of examples for the dual-encoder."""
        tok1 = tokenizer(examples['text1'], truncation=True, max_length=config.max_length)
        tok2 = tokenizer(examples['text2'], truncation=True, max_length=config.max_length)
        return {
            'input_ids_1': tok1['input_ids'], 'attention_mask_1': tok1['attention_mask'],
            'input_ids_2': tok2['input_ids'], 'attention_mask_2': tok2['attention_mask']
        }

    # Convert pandas DataFrames to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    # Define columns to remove after tokenization to keep the dataset clean
    train_cols_to_remove = [c for c in ['text1', 'text2', 'label', 'framework', 'original_label'] if c in train_dataset.column_names]
    val_cols_to_remove = [c for c in ['text1', 'text2', 'label', 'framework'] if c in val_dataset.column_names]

    # Apply tokenization and clean up columns
    train_dataset = train_dataset.map(tokenize_function_dual, batched=True, remove_columns=train_cols_to_remove).rename_column("label_id", "labels")
    val_dataset = val_dataset.map(tokenize_function_dual, batched=True, remove_columns=val_cols_to_remove).rename_column("label_id", "labels")

    # Rename the 'formalism_id' column to match the model's expected input name
    if 'formalism_id' in train_dataset.column_names:
        train_dataset = train_dataset.rename_column("formalism_id", "formalism_ids")
        val_dataset = val_dataset.rename_column("formalism_id", "formalism_ids")

    return train_dataset, val_dataset, label2id, id2label, formalism2id

def prepare_inference_data(tokenizer, config, test_data_path, label2id, formalism2id):
    """
    Loads, processes, and tokenizes a test dataset for inference.

    Args:
        tokenizer: The Hugging Face tokenizer to use.
        config (Config): The configuration object.
        test_data_path (str): The path to the test CSV file.
        label2id (dict): The pre-computed label-to-ID mapping.
        formalism2id (dict): The pre-computed formalism-to-ID mapping.

    Returns:
        tuple: A tuple containing:
            - test_dataset (Dataset): The tokenized test dataset, ready for inference.
            - test_df (pd.DataFrame): The original test DataFrame with added ID columns.
    """
    test_df = pd.read_csv(test_data_path)
    # Apply the existing mappings to the test data
    test_df['formalism_id'] = test_df['framework'].map(formalism2id).fillna(0).astype(int)
    test_df['label_id'] = test_df['label'].map(label2id)

    def tokenize_for_inference(examples):
        """Tokenizes a batch for inference, padding to max_length."""
        tok1 = tokenizer(examples['text1'], truncation=True, max_length=config.max_length, padding="max_length")
        tok2 = tokenizer(examples['text2'], truncation=True, max_length=config.max_length, padding="max_length")
        return {
            'input_ids_1': tok1['input_ids'], 'attention_mask_1': tok1['attention_mask'],
            'input_ids_2': tok2['input_ids'], 'attention_mask_2': tok2['attention_mask'],
            'formalism_ids': examples['formalism_id']
        }

    test_dataset = Dataset.from_pandas(test_df)
    test_dataset = test_dataset.map(tokenize_for_inference, batched=True, batch_size=1000)
    
    # Set the dataset format to PyTorch tensors for model compatibility
    test_dataset.set_format(
        type='torch', 
        columns=['input_ids_1', 'attention_mask_1', 'input_ids_2', 'attention_mask_2', 'formalism_ids']
    )
    
    return test_dataset, test_df