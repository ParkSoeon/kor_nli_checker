# ./data.py - 개선된 버전

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any
from model import format_input_prompt
import gc
from datetime import datetime

def print_log(message: str, prefix: str = "LOG") -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def create_consistent_key(premise: str, proposition: str) -> str:
    """Create consistent key format across all modules"""
    return f"{premise}|||{proposition}"

def load_data(dataset: str) -> List[Dict[str, Any]]:
    with open(dataset, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

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

def save_candidate_to_json(candidates: Dict[str, List[str]], output_dir: str):
    with open(output_dir, 'w', encoding='utf-8') as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)
        
def load_candidates_from_json(candidate_file: str) -> Dict[str, List[str]]:
    with open(candidate_file, 'r', encoding='utf-8') as f:
        candidates = json.load(f)
    return candidates

class OptimizedGRPODataset(Dataset):
    """Memory-optimized dataset for GRPO training"""
    
    def __init__(self, data_samples: List[Dict], tokenizer, max_length=230, use_chat_template=True):
        self.data_samples = data_samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_chat_template = use_chat_template
        
        # Pre-process and cache prompts to reduce computation during training
        print_log(f"Pre-processing {len(data_samples)} samples...")
        self.processed_samples = []
        for i, sample in enumerate(data_samples):
            try:
                processed = self._process_sample(sample)
                self.processed_samples.append(processed)
            except Exception as e:
                print_log(f"Error processing sample {i}: {e}")
                # Add dummy sample to maintain indexing
                self.processed_samples.append({
                    "prompt": "",
                    "reference": "",
                    "premise": "",
                    "proposition": "",
                    "label": ""
                })
        
        print_log(f"Successfully processed {len(self.processed_samples)} samples")
        
    def _process_sample(self, sample):
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
        
    def __len__(self):
        return len(self.processed_samples)

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
        # Remove unwanted tokens
        text = text.replace('<|end_of_text|>', '')
        
        # Handle empty system content
        empty_system_content = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\n<|eot_id|><|start_header_id|>system<|end_header_id|>"
        if text.startswith(empty_system_content):
            text = text.replace("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\n<|eot_id|>", "<|begin_of_text|>", 1)
        
        return text.strip()

    def __getitem__(self, index):
        if index >= len(self.processed_samples):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.processed_samples)}")
        
        sample = self.processed_samples[index]
        
        # Return only necessary data
        return {
            "prompt": sample["prompt"],
            "reference": sample["reference"],
            "premise": sample["premise"],
            "proposition": sample["proposition"],
            "label": sample["label"]
        }

# Keep backward compatibility
GRPODataset = OptimizedGRPODataset

class MemoryEfficientDataLoader:
    """Custom data loader with memory optimization"""
    
    def __init__(self, dataset, batch_size=4, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        
    def __iter__(self):
        if self.shuffle:
            import random
            random.shuffle(self.indices)
        
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            yield self._collate_batch(batch)
            
            # Memory cleanup after each batch
            if i % (self.batch_size * 10) == 0:
                gc.collect()
    
    def _collate_batch(self, batch):
        """Collate batch items into tensors"""
        keys = batch[0].keys()
        collated = {}
        
        for key in keys:
            collated[key] = [item[key] for item in batch]
        
        return collated
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
