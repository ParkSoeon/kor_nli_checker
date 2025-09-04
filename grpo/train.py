# train.py ...... I don't like the name of this file....

from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, TrainingArguments
import torch
from data import load_data, GRPODataset
from typing import Callable, Dict, List
from model import format_input_prompt
from datetime import datetime
from metrics import compute_adapter_a_reward, compute_adapter_b_reward

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def print_log(message, prefix="LOG"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def creat_grpo_trainer(
    model, tokenizer, dataset, reward_function: Callable,
    output_dir: str, learning_rate: float = 5e-5, batch_size: int = 8, epochs: int = 3, **kwargs
):
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
        
        num_generations=8,
        generation_batch_size=8,
        **kwargs
    )

    grpo_trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_function,
    )

    return grpo_trainer

def train_adapter_a(adapter_a, tokenizer, train_data: List[Dict], val_data: List[Dict], output_dir: str, args) -> torch.nn.Module:
    
    train_dataset = GRPODataset(train_data, tokenizer)

    print_log(f"Train Dataset Size: {len(train_dataset)}")
    print_log(f"=== Train Dataset Samples ===")

    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        print_log(f"\n--- Sample {i+1} ---")
        print_log(f"Prompt     : \n{sample['prompt']}")
        print_log(f"Premise    : {sample['premise']}")
        print_log(f"Proposition: {sample['proposition']}")
        print_log(f"Label      : {sample['labels']}")
        print_log(f"Reference  : {sample['reference']}")

    reference_map = {}

    for sample in train_data:
        query = format_input_prompt(sample['input']["premise"], sample['input']["proposition"], sample['input']["label"])
        reference_map[query] = sample.get("output", "")

    print_log(f"Reference Map Size: {len(reference_map)}")

    # Define a Reward Function based on ROUGE(for Adapter A)
    def adapter_a_reward_function(completions, **kwargs):
        rewards = []
        
        # Get prompts from kwargs
        prompts = kwargs.get('prompts', [])
        
        # Handle different completion formats
        if isinstance(completions[0], dict):
            completion_texts = [comp.get("content", comp.get("text", str(comp))) for comp in completions]
        else:
            completion_texts = completions
        
        for i, completion_text in enumerate(completion_texts):
            # Get corresponding prompt
            prompt = prompts[i] if i < len(prompts) else ""
            reference = reference_map.get(prompt, "")
            
            reward = compute_adapter_a_reward(
                completion_text, reference, 
                lambda1=args.lambda1, 
                lambda2=args.lambda2, 
                lambda3=args.lambda3
            )
            
            print_log(f"=== Adapter A Reward {i} ===")
            print_log(f"    Query    : {prompt}")
            print_log(f"    Generated: {completion_text}")
            print_log(f"    Reference: {reference}")
            print_log(f"    Reward   : {reward:.4f}")
            
            rewards.append(reward)

        return rewards

    trainer = creat_grpo_trainer(
        model=adapter_a,
        tokenizer=tokenizer,
        dataset=train_dataset,
        reward_function=adapter_a_reward_function,
        output_dir=f"{output_dir}/adapter_a",
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    print_log("Starting Adapter A Training")
    trainer.train()
    print_log("Finished Adapter A Training")
    return adapter_a

def train_adapter_b(adapter_b, tokenizer, train_data: List[Dict], val_data: List[Dict], adapter_a_candidates: Dict[str, List[str]], output_dir: str, args, ppl_model=None) -> torch.nn.Module:

    print_log(f"=== Adapter B Training Setup ===")
    print_log(f"Adapter A Candidates Samples: {len(adapter_a_candidates)}")

    for i, (key, cands) in enumerate(list(adapter_a_candidates.items())[:5]):
        print_log(f"\n--- Adapter A Candidates Sample {i+1} ---")
        print_log(f"Key: {key}")
        for j, cand in enumerate(cands):
            print_log(f"Candidate {j+1}: {cand}")

    train_dataset = GRPODataset(train_data, tokenizer)

    reference_map = {}
    for sample in train_data:
        query = format_input_prompt(sample['input']["premise"], sample['input']["proposition"], sample['input']["label"])
        reference_map[query] = sample.get("output", "")

    # Define a Reward Function based on Interactive BLEU, ROUGE-L, and PPL(for Adapter B)
    def adapter_b_reward_function(completions, **kwargs):
        """
        completions: List of completion dictionaries
        **kwargs: Contains additional info like prompts
        """
        rewards = []
        
        # Get prompts from kwargs
        prompts = kwargs.get('prompts', [])
        
        # Handle different completion formats
        if isinstance(completions[0], dict):
            completion_texts = [comp.get("content", comp.get("text", str(comp))) for comp in completions]
        else:
            completion_texts = completions

        for i, completion_text in enumerate(completion_texts):
            prompt = prompts[i] if i < len(prompts) else ""
            
            key = None
            for k in adapter_a_candidates.keys():
                premise, proposition = k.split(" ||| ")
                if premise in prompt and proposition in prompt:
                    key = k
                    break

            if key is None:
                rewards.append(0.0)
                continue

            a_candidates = adapter_a_candidates[key]
            reference = reference_map.get(prompt, "")

            reward = compute_adapter_b_reward(
                completion_text, reference, a_candidates, 
                tokenizer=tokenizer, model=ppl_model, 
                lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3
            )

            print_log(f"=== Adapter B Reward {i} ===")
            print_log(f"    Query    : {prompt}")
            print_log(f"    Generated: {completion_text}")
            print_log(f"    Reference: {reference}")
            print_log(f"    Adapter A Candidates: {a_candidates}")
            print_log(f"    Reward   : {reward:.4f}")
            
            rewards.append(reward)
        
        return rewards

    # Create Trainer
    trainer = creat_grpo_trainer(
        model=adapter_b,
        tokenizer=tokenizer,
        dataset=train_dataset,
        reward_function=adapter_b_reward_function,
        output_dir=f"{output_dir}/adapter_b",
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    print_log("Starting Adapter B Training")
    trainer.train()
    print_log("Finished Adapter B Training")
    return adapter_b
