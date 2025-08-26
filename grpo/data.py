# ./data.py

import json
import torch
from torch.utils.data import Dataset, DataLoader

def load_data(dataset):
    with open(dataset, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def create_training_dataset(self, data_samples, tokenizer, max_length=512):
    processed_data = []

    for sample in self.data_samples:
        premise = sample["premise"]
        proposition == sample["proposition"]
        label = sample["label"]

        input_text = format_input_prompt(premise, proposition)

        # Tokenize
        inputs = tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding=False, # we will pad in the DataLoader with DynamicPadding
            return_tensors='pt'
        )

        processed_data.append({
            "input_ids": inputs["input_ids"].squeeze(),  # Remove batch dimension
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
            "premise": premise,
            "proposition": proposition
        })
    
    return processed_data

def save_candidate_to_json(self, candidates, output_dir):
    with open(output_dir, 'w', encoding='utf-8') as f:
        json.dump(candidates, f, ensure_ascii=False, indent=4)

def load_candidates_from_json(self, candidate_file):
    with open(candidate_file, 'r', encoding='utf-8') as f:
        candidates = json.load(f)
    return candidates
