import json
import torch
from torch.utils.data import Dataset, DataLoader

def load_nli_data(data_path):
    """
    Load NLI dataset
    Args:
        data_path: str, path to JSON file
    Returns:
        list: data samples
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def create_training_dataset(data_samples, tokenizer, max_length=512):
    """
    Create dataset for training
    Args:
        data_samples: list of data samples
        tokenizer: tokenizer
        max_length: int, max sequence length
    Returns:
        Dataset
    """
    processed_data = []
    
    for sample in data_samples:
        premise = sample['premise']
        proposition = sample['proposition']
        label = sample['label']
        
        # Format input
        input_text = format_input_prompt(premise, proposition)
        
        # Tokenize
        inputs = tokenizer(
            input_text,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        processed_data.append({
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'premise': premise,
            'proposition': proposition,
            'label': label
        })
    
    return processed_data

def save_candidates_to_json(candidates_dict, output_path):
    """
    Save generated candidates to JSON file
    Args:
        candidates_dict: dict with input as key, candidates as value
        output_path: str, output file path
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(candidates_dict, f, ensure_ascii=False, indent=2)

def load_candidates_from_json(json_path):
    """
    Load candidates from JSON file
    Args:
        json_path: str, path to JSON file
    Returns:
        dict: candidates dictionary
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)
