# ./data.py

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any
from model import format_input_prompt
from datetime import datetime


# 수정사항 create_consistent_key 함수 추가함
def create_consistent_key(premise: str, proposition: str) -> str:
    return f"{premise} ||| {proposition}"
# 여기까지 수정함

def print_log(message: str, prefix: str = "LOG") -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

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

def save_candidate_to_format(candidates: Dict[str, List[str]], original_data: List[Dict], output_dir: str, adapter_name: str = "adapter"):
    result_data = []

    for sample in original_data:
        premise = sample['input']['premise']
        proposition = sample['input']['proposition']
        # Use consistent key format
        key = create_consistent_key(premise, proposition)

        new_sample = {
            "id": sample["id"],
            "input": sample["input"],
            "output": {}
        }

        candidate_list = candidates.get(key, [])
        for idx, candidate in enumerate(candidate_list, 1):
            new_sample["output"][f"{adapter_name}_candidate_{idx}"] = candidate

        # Fill empty slots with empty strings
        for i in range(len(candidate_list), 5):
            new_sample["output"][f"{adapter_name}_candidate_{i+1}"] = ""

        result_data.append(new_sample)

    with open(output_dir, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)  # Reduced indent for smaller files

    print_log(f"Saved candidates to {output_dir}")


def save_combined_cadidates(adapter_a_candidates: Dict[str, List[str]], adapter_b_candidates: Dict[str, List[str]], original_data: List[Dict], output_dir: str):
    result_data = []

    for sample in original_data:
        premise = sample['input']['premise']
        proposition = sample['input']['proposition']
        key = create_consistent_key(premise, proposition)

        new_sample = { 
            "id": sample["id"],
            "input": sample["input"],
            "output": {}
        }

        # Adapter A Candidates
        candidates_a = adapter_a_candidates.get(key, [])
        for i, candidate in enumerate(candidates_a, 1):
            new_sample["output"][f"adapter_a_candidate_{i}"] = candidate
        for i in range(len(candidates_a), 5):
            new_sample["output"][f"adapter_a_candidate_{i+1}"] = ""

        # Adapter B Candidates
        candidates_b = adapter_b_candidates.get(key, [])
        for i, candidate in enumerate(candidates_b, 1):
            new_sample["output"][f"adapter_b_candidate_{i}"] = candidate
        for i in range(len(candidates_b), 5):
            new_sample["output"][f"adapter_b_candidate_{i+1}"] = ""

        result_data.append(new_sample)
    
    with open(output_dir, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    print_log(f"Saved combined candidates to {output_dir}")

class GRPODataset(Dataset):
    # def __init__(self, data_samples: List[Dict], tokenizer, max_length=512):
    #     self.data_samples = data_samples
    #     self.tokenizer = tokenizer
    #     self.max_length = max_length
        
    # def __len__(self):
    #     return len(self.data_samples)

    # def __getitem__(self, index):
    #     sample = self.data_samples[index]
    #     premise = sample['input']["premise"]
    #     proposition = sample['input']["proposition"]
    #     label = sample['input']["label"]
    #     reference = sample.get("output", "")

    #     # query = format_input_prompt(premise, proposition, label)

    #     prompt = format_input_prompt(
    #         sample['input']["premise"],
    #         sample['input']["proposition"],
    #         sample['input']["label"]
    #     )

    #     return {
    #         "prompt": prompt,
    #         # "query": query,
    #         "premise": premise,
    #         "proposition": proposition,
    #         "labels": label,
    #         "reference": reference
    #     }

    def __init__(self, data, tokenizer, use_chat_template: bool = True, max_input_length: int=512):
        self.data = data
        self.tokenizer = tokenizer
        self.use_chat_template = use_chat_template
        self.max_input_length = max_input_length

    def __len__(self):
        return len(self.data)

    def clean_text_tokens(self, text):
        text = text.replace('<|end_of_text|>', '')
        
        # 중복된 시스템 헤더 제거
        empty_system_content = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\n<|eot_id|><|start_header_id|>system<|end_header_id|>"
        if text.startswith(empty_system_content):
            text = text.replace("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\n<|eot_id|>", "<|begin_of_text|>", 1)
        
        return text

    def __getitem__(self, idx):
        sample = self.data[idx]

        base_prompt = format_input_prompt(
            sample['input']["premise"],
            sample['input']["proposition"],
            sample['input']["label"]
        )

        if self.use_chat_template:
            system_content = """당신은 한국어 자연어 추론(NLI) 전문가입니다. 주어진 전제와 가설을 분석하여 함의 관계를 설명해주세요.

    **중요한 규칙:**
    1. 반드시 '[설명] '으로 시작해서 설명문 생성을 시작하세요.
    2. 설명은 한 문장 이상, 세 문장 이하로 작성하고, 마지막에 전제와 가설의 관계가 함의 또는 모순임을 명확히 드러내야 합니다.
    - 예: '함의이다.', '함의에 해당된다.', '모순이다.', '모순에 속한다.' 등
    3. 전제와 가설의 관계는 무조건 '함의', '모순' 중 하나입니다. '중립'이나, '특정 관계에 해당되지 않는다.' 등의 표현은 허용되지 않습니다.
    4. 설명문은 최대 길이 75토큰을 넘지 않도록 최대한 간결하고 명확하게 작성하세요.
    5. 설명문은 한국어로 작성되어야 합니다.

위의 규칙을 엄격히 준수하여 답변해 주세요."""
            messages = [
                {
                    "role": "system", 
                    "content": system_content
                },
                {
                    "role": "user", 
                    "content": base_prompt
                }
            ]
            
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            formatted_prompt = self.clean_text_tokens(formatted_prompt)

        else:
            formatted_prompt = base_prompt
        
        return {
            'prompt': formatted_prompt,
            'premise': sample['input']["premise"],
            'proposition': sample['input']["proposition"], 
            'labels': sample['input']["label"],
            'reference': sample.get("output", "")
        }

