# ./data.py

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any
from model import format_input_prompt

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
    def __init__(self, data_samples: List[Dict], tokenizer, max_length=512, use_chat_template=True):
        self.data_samples = data_samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_chat_template = use_chat_template
        
    def __len__(self):
        return len(self.data_samples)

    def create_prompt(self, premise: str, proposition: str, label: str) -> str:
        messages = [
            {"role": "system", "content": "다음 전제와 가설의 관계를 바탕으로 함의 분석 설명문을 생성하세요."},
            {"role": "user", "content": f"[전제]: {premise}\n[가설]: {proposition}\n[관계]: {label}\n[함의 분석 설명문]:"}
        ]      
        return messages

    def clean_text(self, text: str) -> str: 
        return text

    def __getitem__(self, index):
        sample = self.data_samples[index]
        premise = sample['input']["premise"]
        proposition = sample['input']["proposition"]
        label = sample['input']["label"]
        reference = sample.get("output", "")

        if self.use_chat_template:
            messages = self.create_prompt(premise, proposition, label)

            query_text = self.tokenizer.apply_chat_template(
                messages,
                tokenizer=False,
                add_generation_prompt=True
            )

            query_text = self.clean_text(query_text)
        
        else:
            from model import format_input_prompt
            query_text = format_input_prompt(premise, proposition, label)

        return {
            "prompt": query_text,
            "premise": premise,
            "proposition": proposition,
            "label": label,
            "reference": reference,
        }