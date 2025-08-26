# ./data.py

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any

def load_data(dataset: str) -> List[Dict[str, Any]]:
    with open(dataset, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_candidate_to_json(candidates: Dict[str, List[str]], output_dir: str):
    with open(output_dir, 'w', encoding='utf-8') as f:
        json.dump(candidates, f, ensure_ascii=False, indent=4)

def load_candidates_from_json(candidate_file: str) -> Dict[str, List[str]]:
    with open(candidate_file, 'r', encoding='utf-8') as f:
        candidates = json.load(f)
    return candidates

class GRPODataset(Dataset):
    def __init__(self, data_samples: List[Dict], tokenizer, max_length=512):
        self.data_samples = data_samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, index):
        sample = self.data_samples[index]
        premise = sample["premise"]
        proposition = sample["proposition"]
        label = sample["label"]
        reference = sample.get("output", "")

        input_text = format_input_prompt(premise, proposition, label)

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(),  # Remove batch dimension
            "attention_mask": inputs["attention_mask"].squeeze(),
            "premise": premise,
            "proposition": proposition,
            "labels": label,
            "reference": reference,
            "query": input_text
        }
