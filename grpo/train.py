# train.py - 개선된 버전

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
import weakref
import threading
from contextlib import contextmanager

def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def print_log(message: str, prefix: str ="LOG") -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

@contextmanager
def memory_cleanup():
    """Context manager for automatic memory cleanup"""
    try:
        yield
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

class OptimizedDataCollator(DataCollatorForLanguageModeling):
    """Memory-optimized data collator"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.required_keys = {'input_ids', 'attention_mask', 'labels'}
    
    def __call__(self, features):
        # Process batch with minimal memory footprint
        processed_features = []
        for feature in features:
            # Only keep essential keys
            cleaned_feature = {k: v for k, v in feature.items() if k in self.required_keys or k.startswith(('premise', 'proposition', 'reference'))}
            processed_features.append(cleaned_feature)
        
        batch = super().__call__(processed_features)
        
        # Ensure proper device placement
        if torch.cuda.is_available():
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].cuda()
        
        return batch

class RewardFunctionManager:
    """Manages reward function lifecycle and memory"""
    
    def __init__(self, max_cache_size: int = 100):
        self.max_cache_size = max_cache_size
        self.call_count = 0
        self.cache = {}
        self.lock = threading.Lock()
    
    def cleanup_cache(self):
        with self.lock:
            if len(self.cache) > self.max_cache_size:
                # Remove oldest entries
                oldest_keys = list(self.cache.keys())[:len(self.cache) - self.max_cache_size // 2]
                for key in oldest_keys:
                    del self.cache[key]
            
            self.call_count += 1
            if self.call_count % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

def create_grpo_trainer(
    model, tokenizer, dataset, reward_function: Callable,
    output_dir: str, learning_rate: float = 5e-5, batch_size: int = 3, 
    epochs: int = 3, use_memory_optimization: bool = True, **kwargs
) -> GRPOTrainer:

    data_collator = OptimizedDataCollator(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )

    # Optimized GRPO configuration
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        
        # Logging and saving
        logging_steps=max(10, len(dataset) // (batch_size * 10)),  # Adaptive logging
        save_steps=max(50, len(dataset) // (batch_size * 5)),      # Adaptive saving
        save_total_limit=2,
        
        # Memory optimization
        remove_unused_columns=False,
        dataloader_drop_last=True,
        fp16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7,
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
        
        # Generation parameters
        num_generations=3,  # Reduced for memory efficiency
        max_prompt_length=230,
        max_completion_length=64,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        
        # Gradient settings
        gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps', max(1, 8 // batch_size)),
        max_grad_norm=1.0,  # Gradient clipping
        
        # Optimizer settings
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay=0.01,
        
        # Learning rate scheduler
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        
        # Reporting
        report_to="wandb" if kwargs.get('use_wandb', True) else None,
        
        # Additional optimizations
        dataloader_num_workers=0,  # Avoid multiprocessing overhead
        dataloader_pin_memory=True if torch.cuda.is_available() else False,
    )

    grpo_trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=[reward_function],
        processing_class=tokenizer,
        data_collator=data_collator
    )

    return grpo_trainer

def train_adapter_a(adapter_a, tokenizer, train_data: List[Dict], val_data: List[Dict], output_dir: str, args) -> torch.nn.Module:
    
    print_log("Starting Adapter A training setup...")
    
    with memory_cleanup():
        train_dataset = GRPODataset(train_data, tokenizer, use_chat_template=True)
        print_log(f"Train Dataset Size: {len(train_dataset)}")
        
        # Log sample data for debugging
        if len(train_dataset) > 0:
            for i in range(min(3, len(train_dataset))):
                sample = train_dataset[i]
                print_log(f"Sample {i+1} - Prompt length: {len(sample['prompt'])}, Reference: {sample['reference'][:50]}...")

    # Create reference map with consistent keys
    reference_map = {}
    for sample in train_data:
        # Use the same key format as in generator.py
        key = f"{sample['input']['premise']}|||{sample['input']['proposition']}"
        reference_map[key] = sample.get("output", "")
    
    print_log(f"Reference Map Size: {len(reference_map)}")

    # Create optimized reward function for Adapter A
    reward_manager = RewardFunctionManager()
    
    def create_adapter_a_reward_function(reference_map, args, reward_manager):
        def adapter_a_reward_function(**kwargs) -> List[float]:
            reward_manager.cleanup_cache()
            
            completions = kwargs.get('completions', [])
            premise = kwargs.get('premise', [])
            proposition = kwargs.get('proposition', [])
            
            if not completions:
                print_log("Warning: No completions received in reward function")
                return [0.0] * len(kwargs.get('prompts', []))
            
            rewards = []
            
            for i, completion_text in enumerate(completions):
                ref_text = ""
                
                # Use consistent key format
                if premise and proposition and i < len(premise) and i < len(proposition):
                    key = f"{premise[i]}|||{proposition[i]}"
                    ref_text = reference_map.get(key, "")
                
                # Skip empty references
                if not ref_text.strip():
                    print_log(f"Warning: Empty reference for sample {i}")
                    rewards.append(0.0)
                    continue
                
                try:
                    reward = compute_adapter_a_reward(
                        generated=completion_text.strip(),
                        references=ref_text.strip(),
                        lambda1=args.lambda1,
                        lambda2=args.lambda2,
                        lambda3=args.lambda3
                    )
                    rewards.append(float(reward))
                except Exception as e:
                    print_log(f"Error computing reward for sample {i}: {e}")
                    rewards.append(0.0)
            
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            print_log(f"Adapter A Batch - Samples: {len(rewards)}, Avg Reward: {avg_reward:.4f}")
            
            return rewards
        
        return adapter_a_reward_function

    reward_function = create_adapter_a_reward_function(reference_map, args, reward_manager)
    
    # Create trainer with optimized settings
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
        gradient_accumulation_steps=max(1, 8 // args.batch_size)
    )

    print_log(">> Starting Adapter A GRPO Training")
    
    try:
        trainer.train()
        print_log(">> Adapter A Training completed successfully")
    except Exception as e:
        print_log(f"Error during Adapter A training: {e}")
        raise
    finally:
        # Cleanup
        del reward_manager, reference_map
        gc.collect()

    return adapter_a

def train_adapter_b(adapter_b, tokenizer, train_data: List[Dict], val_data: List[Dict], 
                   adapter_a_candidates: Dict[str, List[str]], output_dir: str, args, ppl_model=None) -> torch.nn.Module:
    
    print_log("Starting Adapter B training setup...")
    print_log(f"Adapter A Candidates: {len(adapter_a_candidates)} samples")
    
    # Log some adapter A candidates for debugging
    for i, (key, cands) in enumerate(list(adapter_a_candidates.items())[:2]):
        print_log(f"Sample {i+1} Key: {key[:50]}...")
        print_log(f"  Candidates: {len(cands)}")
        for j, cand in enumerate(cands[:2]):
            print_log(f"    Cand {j+1}: {cand[:30]}...")

    with memory_cleanup():
        train_dataset = GRPODataset(train_data, tokenizer, use_chat_template=True)

    # Create reference map with consistent keys
    reference_map = {}
    for sample in train_data:
        key = f"{sample['input']['premise']}|||{sample['input']['proposition']}"
        reference_map[key] = sample.get("output", "")

    # Create optimized reward function for Adapter B
    reward_manager = RewardFunctionManager()
    
    def create_adapter_b_reward_function(reference_map, adapter_a_candidates, args, ppl_model, reward_manager):
        def adapter_b_reward_function(**kwargs) -> List[float]:
            reward_manager.cleanup_cache()
            
            completions = kwargs.get('completions', [])
            premise = kwargs.get('premise', [])
            proposition = kwargs.get('proposition', [])
            
            if not completions:
                print_log("Warning: No completions received in Adapter B reward function")
                return [0.0] * len(kwargs.get('prompts', []))
            
            rewards = []
            
            for i, completion_text in enumerate(completions):
                a_candidates = []
                ref_text = ""
                
                # Use consistent key format
                if premise and proposition and i < len(premise) and i < len(proposition):
                    key = f"{premise[i]}|||{proposition[i]}"
                    a_candidates = adapter_a_candidates.get(key, [])
                    ref_text = reference_map.get(key, "")
                
                # Skip if no Adapter A candidates
                if not a_candidates:
                    print_log(f"Warning: No Adapter A candidates for sample {i}")
                    rewards.append(0.0)
                    continue
                
                if not ref_text.strip():
                    print_log(f"Warning: Empty reference for sample {i}")
                    rewards.append(0.0)
                    continue
                
                try:
                    reward = compute_adapter_b_reward(
                        generated=completion_text.strip(),
                        references=ref_text.strip(),
                        adapter_a_cands=a_candidates,
                        model=ppl_model,
                        tokenizer=tokenizer,
                        lambda1=args.lambda1,  # For negative interactive BLEU
                        lambda2=args.lambda2,  # For ROUGE-L
                        lambda3=args.lambda3   # For negative PPL penalty
                    )
                    rewards.append(float(reward))
                except Exception as e:
                    print_log(f"Error computing Adapter B reward for sample {i}: {e}")
                    rewards.append(0.0)
            
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            print_log(f"Adapter B Batch - Samples: {len(rewards)}, Avg Reward: {avg_reward:.4f}")
            
            return rewards
        
        return adapter_b_reward_function

    reward_function = create_adapter_b_reward_function(reference_map, adapter_a_candidates, args, ppl_model, reward_manager)

    # Create trainer
    trainer = create_grpo_trainer(
        model=adapter_b,
        tokenizer=tokenizer,
        dataset=train_dataset,
        reward_function=reward_function,
        output_dir=f"{output_dir}/adapter_b",
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        use_memory_optimization=True,
        gradient_accumulation_steps=max(1, 8 // args.batch_size)
    )

    print_log(">> Starting Adapter B GRPO Training")
    
    try:
        trainer.train()
        print_log(">> Adapter B Training completed successfully")
    except Exception as e:
        print_log(f"Error during Adapter B training: {e}")
        raise
    finally:
        # Cleanup
        del reward_manager, reference_map
        if ppl_model:
            del ppl_model
        gc.collect()

    return adapter_b
