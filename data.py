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

def save_candidate_to_format(candidates: Dict[str, List[str]], original_data: List[Dict], output_dir: str, adapter_name: str = "adapter"):

    result_data = []

    for sample in original_data:
        premise = sample['input']['premise']
        proposition = sample['input']['proposition']
        label = sample['input']['label']
        key = f"{premise}|||{proposition}"

        new_sample = {
            "id": sample["id"],
            "input": sample["input"],
            "output": {}
        }

        candidate_list = candidates.get(key, [])
        for idx, candidate in enumerate(candidate_list, 1):
            new_sample["output"][f"{adapter_name}_candidate_{idx}"] = candidate

        if not candidate_list:
            for i in range(5):
                new_sample["output"][f"{adapter_name}_candidate_{i+1}"] = ""

        result_data.append(new_sample)

    with open(output_dir, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=4)

    print_log(f"Saved candidates to {output_dir}")

def save_combined_cadidates(adapter_a_candidates: Dict[str, List[str]], adapter_b_candidates: Dict[str, List[str]], original_data: List[Dict], output_dir: str):
    result_data = []

    for sample in original_data:
        premise = sample['input']['premise']
        proposition = sample['input']['proposition']
        key = f"{premise}|||{proposition}"

        new_sample = { 
            "id": sample["id"],
            "input": sample["input"],
            "output": {}
        }

        # Adapter A Candidates -> Fill empty A candidates with ""(exception handling)
        candidates_a = adapter_a_candidates.get(key, [])
        for i, candidate in enumerate(candidates_a, 1):
            new_sample["output"][f"adapter_a_candidate_{i}"] = candidate
        for i in range(len(candidates_a)+1, 6):
            new_sample["output"][f"adapter_a_candidate_{i+1}"] = ""

        # Adapter B Candidates -> Fill empty B candidates with ""
        candidates_b = adapter_b_candidates.get(key, [])
        for i, candidate in enumerate(candidates_b, 1):
            new_sample["output"][f"adapter_b_candidate_{i}"] = candidate
        for i in range(len(candidates_b)+1, 6):
            new_sample["output"][f"adapter_b_candidate_{i+1}"] = ""

        result_data.append(new_sample)
    with open(output_dir, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=4)

    print_log(f"Saved combined candidates to {output_dir}")


def save_candidate_to_json(candidates: Dict[str, List[str]], output_dir: str):
    with open(output_dir, 'w', encoding='utf-8') as f:
        json.dump(candidates, f, ensure_ascii=False, indent=4)
        
def load_candidates_from_json(candidate_file: str) -> Dict[str, List[str]]:
    with open(candidate_file, 'r', encoding='utf-8') as f:
        candidates = json.load(f)
    return candidates

class GRPODataset(Dataset):
    def __init__(self, data_samples: List[Dict], tokenizer, max_length=230, use_chat_template=True):
        self.data_samples = data_samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_chat_template = use_chat_template
        
    def __len__(self):
        return len(self.data_samples)

    def create_prompt(self, premise: str, proposition: str, label: str) -> str:
        if self.use_chat_template:
            messages = [
                {"role": "system", "content": "다음 전제와 가설의 관계를 바탕으로 함의 분석 설명문을 생성하세요."},
                {"role": "user", "content": f"[전제]: {premise}\n[가설]: {proposition}\n[관계]: {label}\n[함의 분석 설명문]:"}
            ]      
        
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return self.clean_text(prompt)
        else:
            return format_input_prompt(premise, proposition, label)

    def clean_text(self, text: str) -> str: 
        text = text.replace('<|end_of_text|>', '')

        empty_system_content = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\n<|eot_id|><|start_header_id|>system<|end_header_id|>"
        if text.startswith(empty_system_content):
            text = text.replace("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\n<|eot_id|>", "<|begin_of_text|>", 1)
        
        return text.strip()

    def __getitem__(self, index):
        sample = self.data_samples[index]
        premise = sample['input']["premise"]
        proposition = sample['input']["proposition"]
        label = sample['input']["label"]
        reference = sample.get("output", "")
        
        query_text = self.create_prompt(premise, proposition, label)

        return {
            "prompt": query_text,
            "reference": reference,
            "premise": premise,
            "proposition": proposition,
            "label": label
        }
