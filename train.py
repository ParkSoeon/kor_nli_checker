# train.py ...... I don't like the name of this file....

from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from data import load_data, GRPODataset
from typing import Callable, Dict, List, Optional, Any
from model import format_input_prompt
from datetime import datetime
from metrics import compute_adapter_a_reward, compute_adapter_b_reward
import os
import gc

def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def print_log(message: str, prefix: str ="LOG") -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class MemoryOptimizeDataCollator(DataCollatorForLanguageModeling):

    def __call__(self, features):
        batch = super().__call__(features)
        
        keys_to_remove = []
        for key in batch.keys():
            if key not in ['input_ids', 'attention_mask', 'labels']:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            if key in batch:
                del batch[key]
        
        return batch

def create_grpo_trainer(
    model, tokenizer, dataset, reward_function: Callable,
    output_dir: str, learning_rate: float = 5e-5, batch_size: int = 3, epochs: int = 3, use_memory_optimization: bool = True, **kwargs
) -> GRPOTrainer:

    if use_memory_optimization:
        data_collator = MemoryOptimizeDataCollator(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        report_to="wandb",
        dataloader_drop_last=True,
        
        num_generations=3,
        # generation_batch_size=batch_size,
        max_prompt_length=230,
        max_completion_length=64,  
        temperature=0.7,
        top_p=0.95,

        gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps', 1),
        
        **{k: v for k, v, in kwargs.items() if k not in ['gradient_accumulation_steps']}
    )

    grpo_trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=[reward_function],
        processing_class=tokenizer
    )

    return grpo_trainer

class MemoryAwareRewardFunction:
    def __init__(self, reference_map: Dict[str, str], max_batch_size: int = 4):
        self.reference_map = reference_map
        self.max_batch_size = max_batch_size
        self.call_count = 0

    def clear_cache(self):
        self.call_count += 1
        if self.call_count % 10 == 0:
            clear_memory()

def train_adapter_a(adapter_a, tokenizer, train_data: List[Dict], val_data: List[Dict], output_dir: str, args) -> torch.nn.Module:
    
    train_dataset = GRPODataset(train_data, tokenizer, use_chat_template=True)

    print_log(f"Train Dataset Size: {len(train_dataset)}")
    print_log(f"=== Train Dataset Samples ===")

    for i in range(min(5, len(train_dataset))):
        sample = train_dataset[i]
        print_log(f"\n--- Sample {i+1} ---")
        print_log(f"Prompt     : \n{sample['prompt']}")
        print_log(f"Premise    : {sample['premise']}")
        print_log(f"Proposition: {sample['proposition']}")
        print_log(f"Label      : {sample['label']}")
        print_log(f"Reference  : {sample['reference']}")

    # Create a reference map for reward calculation
    reference_map = {}

    for sample in train_data:
        # sample = format_input_prompt(sample['input']["premise"], sample['input']["proposition"], sample['input']["label"])
        # reference_map[sample] = sample.get("output", "")
        key = f"{sample['input']['premise']} ||| {sample['input']['proposition']}"
        reference_map[key] = sample.get("output", "")

    print_log(f"Reference Map Size: {len(reference_map)}")

    clear_memory()

    # Define a Reward Function based on ROUGE(for Adapter A)
    def create_adapter_a_reward_function(reference_map, args):
        call_count = 0
        
        def adapter_a_reward_function(**kwargs) -> List[float]:
            nonlocal call_count
            call_count += 1
            
            if call_count % 10 == 0:
                clear_memory()
            
            completions = kwargs.get('completions', [])
            prompts = kwargs.get('prompts', [])
            premise = kwargs.get('premise', [])
            proposition = kwargs.get('proposition', [])
            reference = kwargs.get('reference', [])
            
            print_log(f"Reward function called with {len(completions) if completions else 0} completions")
            
            if not completions:
                print_log("No completions received")
                return [0.0] * 5
            
            rewards = []
            
            for i, completion_text in enumerate(completions):
                ref_text = ""
                
                if premise and proposition and i < len(premise) and i < len(proposition):
                    key = f"{premise[i]} ||| {proposition[i]}"
                    ref_text = reference_map.get(key, "")
                
                reward = compute_adapter_a_reward(
                    generated=completion_text, 
                    references=ref_text,
                    lambda1=args.lambda1,
                    lambda2=args.lambda2,
                    lambda3=args.lambda3
                )
                rewards.append(reward)
                    
                if i < 3:
                    print_log(f"Sample {i}: Generated='{completion_text[:50]}...', Reference='{ref_text[:50]}...', Reward={reward:.4f}")
            
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            print_log(f"Adapter A Average Reward: {avg_reward:.4f}")
            
            return rewards
        
        adapter_a_reward_function.__name__ = "adapter_a_reward_function"
        return adapter_a_reward_function

    reward_function = create_adapter_a_reward_function(reference_map, args)

    trainer = create_grpo_trainer(
        model=adapter_a,
        tokenizer=tokenizer,
        dataset=train_dataset,
        reward_function=reward_function,
        output_dir=f"{output_dir}/adapter_a",
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        use_memory_optimization=True,
        gradient_checkpointing=getattr(args, 'gradient_checkpointing', False),
        gradient_accumulation_steps=max(1, 8 // args.batch_size)
    )

    print_log(">> Starting Adapter A Training")
    trainer.train()
    print_log(">> Finished Adapter A Training")

    # adapter_a_save_dir = f"{output_dir}/adapter_a_checkpoint_{get_timestamp()}"
    # os.makedirs(adapter_a_save_dir, exist_ok=True)

    # adapter_a.save_pretrained(adapter_a_save_dir)

    clear_memory()

    return adapter_a

def train_adapter_b(adapter_b, tokenizer, train_data: List[Dict], val_data: List[Dict], adapter_a_candidates: Dict[str, List[str]], output_dir: str, args, ppl_model=None) -> torch.nn.Module:

    print_log(f"=== Adapter B Training Setup ===")
    print_log(f"Adapter A Candidates Samples: {len(adapter_a_candidates)}")

    for i, (key, cands) in enumerate(list(adapter_a_candidates.items())[:5]):
        print_log(f"\n--- Adapter A Candidates Sample {i+1} ---")
        print_log(f"Key: {key}")
        for j, cand in enumerate(cands):
            print_log(f"Candidate {j+1}: {cand}")

    train_dataset = GRPODataset(train_data, tokenizer, use_chat_template=True)

    reference_map = {}

    for sample in train_data:
        
        key = f"{sample['input']['premise']} ||| {sample['input']['proposition']}"
        reference_map[key] = sample.get("output", "")

    clear_memory()

    # Define a Reward Function based on Interactive BLEU, ROUGE-L, and PPL(for Adapter B)
    # def adapter_b_reward_function(**kwargs) -> float:
    #     """
    #     **kwargs: Contains additional info like prompts
    #     """
    #     print_log(f"Reward Function called with kwargs keys: {list(kwargs.keys())}")

    #     completions = kwargs.get('completions', [])
    #     prompts = kwargs.get('prompts', [])
    #     premise = kwargs.get('premise', [])
    #     proposition = kwargs.get('proposition', [])
    #     reference = kwargs.get('reference', [])

    #     print_log(f"    Number of samples: {len(samples) if samples else 0}")
    #     print_log(f"    Number of responses: {len(responses) if responses else 0}")

    #     rewards: List[float] = []
        
    #     num_completions = len(completions) if completions else 0\

    #     for i in range(num_completions):
    #         completion_text = completions[i] if i < len(completions) else ""

    #         # Create key to find Adapter A candidates
    #         if premise and proposition and i < len(premise) and i < len(proposition):
    #             key = f"{premise[i]} ||| {proposition[i]}"
    #         else:
    #             key = list(adapter_a_candidates.keys())[0] if adapter_a_candidates else ""

    #         a_candidates = adapter_a_candidates.get(key, [])

    #         # Find for Reference Text
    #         ref_text = ""
    #         if reference and i < len(reference):
    #             ref_text = reference[i]
    #         else:
    #             ref_text = reference_map.get(key, "")

    #         reward = compute_adapter_b_reward(
    #             generated=completion_text, references=ref_text, adapter_a_cands=a_candidates, 
    #             model = ppl_model, tokenizer=tokenizer,
    #             lambda1=args.lambda1,
    #             lambda2=args.lambda2,
    #             lambda3=args.lambda3
    #         )

    #         print_log(f"=== Adapter B Reward {i} ===")
    #         print_log(f"    Generated: {response}")
    #         print_log(f"    Reference: {reference}")
    #         print_log(f"    Adapter A Candidates: {a_candidates}")
    #         print_log(f"    Reward   : {reward:.4f}")
            
    #         rewards.append(reward)

    #     print_log(f"Average Reward: {sum(rewards)/len(rewards) if rewards else 0:.4f}")
    #     return rewards

    def create_adapter_b_reward_function(reference_map, adapter_a_candidates, args, ppl_model=None):
        call_count = 0
        
        def adapter_b_reward_function(**kwargs) -> List[float]:
            nonlocal call_count
            call_count += 1
            
            if call_count % 10 == 0:
                clear_memory()
            
            completions = kwargs.get('completions', [])
            prompts = kwargs.get('prompts', [])
            premise = kwargs.get('premise', [])
            proposition = kwargs.get('proposition', [])
            reference = kwargs.get('reference', [])
            
            print_log(f"Adapter B reward function called with {len(completions) if completions else 0} completions")
            
            if not completions:
                print_log("No completions received")
                return [0.0] * 5
            
            rewards = []
            
            for i, completion_text in enumerate(completions):
                a_candidates = []
                ref_text = ""
                
                if premise and proposition and i < len(premise) and i < len(proposition):
                    key = f"{premise[i]} ||| {proposition[i]}"
                    a_candidates = adapter_a_candidates.get(key, [])
                    ref_text = reference_map.get(key, "")

                reward = compute_adapter_b_reward(
                    generated=completion_text,
                    references=ref_text,
                    adapter_a_cands=a_candidates,
                    model=ppl_model,
                    tokenizer=tokenizer,
                    lambda1=args.lambda1,
                    lambda2=args.lambda2,
                    lambda3=args.lambda3
                )
                rewards.append(reward)
                
                if i < 3:  
                    print_log(f"Sample {i}: Generated='{completion_text[:50]}...', Reference='{ref_text[:50]}...', Reward={reward:.4f}")
            
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            print_log(f"Adapter B Average Reward: {avg_reward:.4f}")
            
            return rewards
        
        adapter_b_reward_function.__name__ = "adapter_b_reward_function"
        return adapter_b_reward_function

    reward_function = create_adapter_b_reward_function(reference_map, adapter_a_candidates, args, ppl_model)

    # Create Trainer
    trainer = create_grpo_trainer(
        model=adapter_b,
        tokenizer=tokenizer,
        dataset=train_dataset,
        reward_function=reward_function,
        output_dir=f"{output_dir}/adapter_b",
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    print_log(">> Starting Adapter B Training")
    trainer.train()
    print_log(">> Finished Adapter B Training")

    # adapter_b_save_dir = f"{output_dir}/adapter_b_checkpoint_{get_timestamp()}"
    # os.makedirs(adapter_b_save_dir, exist_ok=True)
    # adapter_b.save_pretrained(adapter_b_save_dir)
    
    clear_memory()

    return adapter_b
