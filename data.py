import json
import random
from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
import torch
from typing import Dict, List, Any, Optional

def print_log(message, prefix="LOG"):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{prefix}] {message}")

class EntailmentDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_input_length=512, max_output_length=256, 
                 model_type="causal", mode="train", use_chat_template=True, 
                 fewshot_examples=None, num_fewshot=1):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.model_type = model_type
        self.mode = mode
        self.use_chat_template = use_chat_template
        self.fewshot_examples = fewshot_examples if fewshot_examples else []
        self.num_fewshot = min(num_fewshot, len(self.fewshot_examples))

        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        print_log(f"Loaded {len(self.data)} examples from {data_path}")
        print_log(f"Using {self.num_fewshot} few-shot examples")
        print_log(f"Chat template: {self.use_chat_template}")

    def create_fewshot_prompt(self, premise, proposition, label=None):
        examples = []

        # Add few-shot examples
        for example in self.fewshot_examples[:self.num_fewshot]:
            ex_premise = example['input']['premise']
            ex_proposition = example['input']['proposition']
            ex_label = example['input']['label']  
            ex_output = example['output']
            examples.append({
                "premise": ex_premise,
                "proposition": ex_proposition,
                "label": ex_label,
                "explanation": ex_output
            })

        if self.use_chat_template:
            messages = [{"role": "system", "content": "당신은 한국어 자연어 추론(NLI) 전문가입니다. 주어진 전제와 가설을 분석하여 함의 관계를 설명해주세요."}]

            if examples:
                example_content = "다음은 자연어 추론 과제의 예시입니다:\n\n"
                for i, ex in enumerate(examples, 1):
                    example_content += f"[예시 {i}]\n"
                    example_content += f"[전제] {ex['premise']}\n" 
                    example_content += f"[가설] {ex['proposition']}\n"
                    example_content += f"[관계] {ex['label']}\n"
                    example_content += f"[설명] {ex['explanation']}\n\n"

                example_content += "이제 새로운 전제와 가설에 대해 관계를 분석하여 설명문을 설명하세요.\n\n"
                messages.append({"role": "user", "content": example_content})

            current_message = f"[전제] {premise}\n[가설] {proposition}\n[관계] {label}"
            messages.append({"role": "user", "content": current_message})

            return messages
        else:
            prompt_parts = ["다음은 자연어 추론 작업입니다. 전제와 가설 사이의 관계를 분석하고 설명해주세요.\n"]
            
            for ex in examples:
                prompt_parts.append(f"[전제] {ex['premise']}")
                prompt_parts.append(f"[가설] {ex['proposition']}")
                prompt_parts.append(f"[관계] {ex['label']}")
                prompt_parts.append(f"[설명] {ex['explanation']}\n")
            
            prompt_parts.append(f"[전제] {premise}")
            prompt_parts.append(f"[가설] {proposition}")
            prompt_parts.append(f"[관계] {label}")
            if self.mode == "train":
                prompt_parts.append("[설명]")
            
            return "\n".join(prompt_parts)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        premise = item['input']['premise']
        proposition = item['input']['proposition']
        label = item['input']['label']
        output = item['output']

        if self.use_chat_template:
            messages = self.create_fewshot_prompt(premise, proposition, label)
            
            if self.mode == "train":
                # Add target as assistant response
                full_message = messages + [{"role": "assistant", "content": output}]
                
                # apply_chat_template 적용
                full_text_list = self.tokenizer.apply_chat_template(
                    full_message,
                    tokenize=False,
                    add_generation_prompt=False
                )

                # Fix potential duplicate system headers
                if full_text_list.startswith("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\n<|eot_id|><|start_header_id|>system<|end_header_id|>"):
                    full_text_list = full_text_list.replace("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\n<|eot_id|>", "<|begin_of_text|>", 1)
            
                full_text = self.tokenizer(
                    full_text_list,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_input_length + self.max_output_length,
                    add_special_tokens=False
                )

                prefix_list = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )

                prefix = self.tokenizer(
                    prefix_list,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_input_length,
                    add_special_tokens=False
                )

                input_ids = full_text["input_ids"].squeeze(0)  # (1, L) -> (L,)
                attention_mask = full_text["attention_mask"].squeeze(0)
                prefix_input_ids = prefix["input_ids"].squeeze(0)  # (1, L) -> (L,)

                # Create labels for training
                labels = input_ids.clone()
                prefix_len = len(prefix_input_ids)
                
                # Mask input tokens (set to -100 to ignore in loss)
                labels[:prefix_len] = -100 

                # Debug logging for first few examples
                if idx < 3:
                    decoded = self.tokenizer.decode(input_ids, skip_special_tokens=False)
                    print_log(f"==== Training Example {idx} ====")
                    print_log(f"Premise: {premise}")
                    print_log(f"Proposition: {proposition}")
                    print_log(f"Label: {label}")
                    print_log(f"Target Output: {output}")
                    print_log(f"Input tokens length: {prefix_len}")
                    print_log(f"Total tokens length: {len(input_ids)}")
                    print_log(f"Target tokens length: {len(input_ids) - prefix_len}")
                    print_log("="*80)

                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }

            else:  # inference/test mode
                inf_ft_list = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True  # Add generation prompt for inference
                )

                # Fix potential duplicate system headers
                if inf_ft_list.startswith("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\n<|eot_id|><|start_header_id|>system<|end_header_id|>"):
                    inf_ft_list = inf_ft_list.replace("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\n<|eot_id|>", "<|begin_of_text|>", 1)
                
                input_encoding = self.tokenizer(
                    inf_ft_list,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_input_length,
                    add_special_tokens=False
                )

                # Log first few examples for debugging
                if idx < 3:
                    print_log(f"==== Inference Example {idx} ====")
                    print_log(f"Premise: {premise}")
                    print_log(f"Proposition: {proposition}")
                    print_log(f"Label: {label}")
                    print_log(f"Input prompt length: {len(input_encoding['input_ids'].squeeze())}")

                return {
                    "input_ids": input_encoding["input_ids"].squeeze(),
                    "attention_mask": input_encoding["attention_mask"].squeeze(),
                    "target": output,
                    "id": item.get('id', idx),
                    "prompt": inf_ft_list,
                    "premise": premise,
                    "proposition": proposition,
                    "label": label
                }
        else:
            print("[ERROR] Not Set as Chat Template Mode. Please check the configuration.")
            assert 0

class DynamicDataCollator:
    """Dynamic padding data collator that handles both causal and seq2seq models"""
    
    def __init__(self, tokenizer, model_type="causal", pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.pad_to_multiple_of = pad_to_multiple_of
        
        if model_type == "causal":
            self.base_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                pad_to_multiple_of=pad_to_multiple_of
            )
        else:
            self.base_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                pad_to_multiple_of=pad_to_multiple_of
            )
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Handle inference mode (features without labels)
        if not any("labels" in feature for feature in features):
            return self._collate_inference(features)
        
        # Handle training mode (features with labels)
        return self._collate_training(features)
    
    def _collate_training(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate training examples with dynamic padding"""
        
        # Extract sequences
        input_ids = [feature["input_ids"] for feature in features]
        attention_masks = [feature["attention_mask"] for feature in features] 
        labels = [feature["labels"] for feature in features]
        
        # Find max length in batch
        max_length = max(len(seq) for seq in input_ids)
        
        # Round up to multiple of pad_to_multiple_of if specified
        if self.pad_to_multiple_of:
            max_length = ((max_length + self.pad_to_multiple_of - 1) 
                         // self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        # Pad sequences
        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []
        
        for i in range(len(features)):
            seq_len = len(input_ids[i])
            pad_len = max_length - seq_len
            
            # Pad input_ids
            padded_seq = torch.cat([
                input_ids[i], 
                torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=input_ids[i].dtype)
            ])
            padded_input_ids.append(padded_seq)
            
            # Pad attention_mask
            padded_mask = torch.cat([
                attention_masks[i],
                torch.zeros(pad_len, dtype=attention_masks[i].dtype)
            ])
            padded_attention_masks.append(padded_mask)
            
            # Pad labels
            padded_label = torch.cat([
                labels[i],
                torch.full((pad_len,), -100, dtype=labels[i].dtype)  # -100 is ignored in loss
            ])
            padded_labels.append(padded_label)
        
        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_masks),
            "labels": torch.stack(padded_labels)
        }
    
    def _collate_inference(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate inference examples with dynamic padding"""
        
        input_ids = [feature["input_ids"] for feature in features]
        attention_masks = [feature["attention_mask"] for feature in features]
        
        # Find max length in batch
        max_length = max(len(seq) for seq in input_ids)
        
        # Round up to multiple of pad_to_multiple_of if specified
        if self.pad_to_multiple_of:
            max_length = ((max_length + self.pad_to_multiple_of - 1) 
                         // self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        # Pad sequences
        padded_input_ids = []
        padded_attention_masks = []
        
        for i in range(len(features)):
            seq_len = len(input_ids[i])
            pad_len = max_length - seq_len
            
            # Pad input_ids (pad to the left for generation)
            if self.model_type == "causal":
                # For causal models, pad to the right
                padded_seq = torch.cat([
                    input_ids[i],
                    torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=input_ids[i].dtype)
                ])
                padded_mask = torch.cat([
                    attention_masks[i],
                    torch.zeros(pad_len, dtype=attention_masks[i].dtype)
                ])
            else:
                # For seq2seq models, pad to the right
                padded_seq = torch.cat([
                    input_ids[i],
                    torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=input_ids[i].dtype)
                ])
                padded_mask = torch.cat([
                    attention_masks[i],
                    torch.zeros(pad_len, dtype=attention_masks[i].dtype)
                ])
                
            padded_input_ids.append(padded_seq)
            padded_attention_masks.append(padded_mask)
        
        batch = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_masks)
        }
        
        # Add other fields for inference
        other_keys = set(features[0].keys()) - {"input_ids", "attention_mask"}
        for key in other_keys:
            batch[key] = [feature[key] for feature in features]
        
        return batch

def prepare_fewshot_examples(train_path: str, seed: int, num_examples: int = 3) -> List[Dict]:
    """Prepare few-shot examples from training data"""
    print_log(f"Preparing few-shot examples from {train_path}")
    
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    print_log(f"Total training examples available: {len(train_data)}")
    
    # Set seed for reproducible few-shot selection
    random.seed(seed)
    selected_examples = random.sample(train_data, min(num_examples, len(train_data)))
    random.seed()  # Reset seed
    
    print_log(f"Selected {len(selected_examples)} few-shot examples with seed {seed}")
    
    # Log first few-shot example for debugging
    if selected_examples:
        first_ex = selected_examples[0]
        print_log("First few-shot example:")
        print_log(f"  Premise: {first_ex['input']['premise']}")
        print_log(f"  Proposition: {first_ex['input']['proposition']}")
        print_log(f"  Label: {first_ex['input']['label']}")
        print_log(f"  Output: {first_ex['output']}")
    
    return selected_examples

def create_datasets(args, tokenizer, fewshot_examples):
    """Create train, eval, and test datasets"""
    print_log("Creating datasets...")
    
    train_dataset = EntailmentDataset(
        args.train_path, tokenizer, args.max_input_length, args.max_output_length,
        args.model_type, "train", args.use_chat_template, fewshot_examples, args.num_fewshot
    )
    
    eval_dataset = EntailmentDataset(
        args.dev_path, tokenizer, args.max_input_length, args.max_output_length,
        args.model_type, "train", args.use_chat_template, fewshot_examples, args.num_fewshot
    )

    test_dataset = EntailmentDataset(
        args.test_path, tokenizer, args.max_input_length, args.max_output_length,
        args.model_type, "test", args.use_chat_template, fewshot_examples, args.num_fewshot
    )
    
    print_log(f"Created datasets - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, eval_dataset, test_dataset

def create_data_collator(tokenizer, model_type="causal", pad_to_multiple_of=8):
    """Create appropriate data collator with dynamic padding"""
    print_log(f"Creating dynamic data collator for {model_type} model")
    
    collator = DynamicDataCollator(
        tokenizer=tokenizer,
        model_type=model_type,
        pad_to_multiple_of=pad_to_multiple_of
    )
    
    print_log("Dynamic data collator created successfully")
    return collator
