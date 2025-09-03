# ./generator.py - 개선된 버전

import torch
import gc
from tqdm import tqdm
from typing import List, Dict
from transformers import AutoTokenizer
from peft import PeftModel
from model import load_model_and_tokenizer
from data import GRPODataset, create_consistent_key
from torch.utils.data import DataLoader
import os
from datetime import datetime

def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def print_log(message: str, prefix: str = "LOG") -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def clear_memory():
    """Enhanced memory clearing"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()

def load_adapter_for_inference(model_path_or_model, base_model_name=None, tokenizer=None, device="cuda"):
    """Load adapter model for inference with better error handling"""
    
    if isinstance(model_path_or_model, str):
        print_log(f"Loading adapter from path: {model_path_or_model}")
        
        # Load base model if needed
        if base_model_name:
            base_model, tokenizer = load_model_and_tokenizer(base_model_name, device)
        else:
            # Extract base model info from adapter config
            config_path = os.path.join(model_path_or_model, "adapter_config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    adapter_config = json.load(f)
                base_model_name = adapter_config.get('base_model_name_or_path')
                if base_model_name:
                    base_model, tokenizer = load_model_and_tokenizer(base_model_name, device)
                else:
                    raise ValueError("Base model name not found in adapter config")
            else:
                raise ValueError(f"Adapter config not found at {config_path}")
        
        # Load PEFT model
        model = PeftModel.from_pretrained(base_model, model_path_or_model)
        model.eval()
        
        # Enable memory-efficient inference
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        return model, tokenizer
    else:
        # Already loaded model
        model_path_or_model.eval()
        return model_path_or_model, tokenizer

@torch.no_grad()
def generate_candidates_batch(model, tokenizer, prompts: List[str], num_candidates: int = 5, 
                             max_new_tokens: int = 64, temperature: float = 0.7, 
                             top_p: float = 0.95, device="cuda"):
    """Optimized batch candidate generation"""
    
    model.eval()
    
    # Pre-validate inputs
    if not prompts:
        print_log("Warning: No prompts provided")
        return []
    
    # Tokenize with better error handling
    try:
        inputs = tokenizer(
            prompts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=512,
            add_special_tokens=True
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
    except Exception as e:
        print_log(f"Error tokenizing prompts: {e}")
        return [[] for _ in prompts]
    
    all_candidates = []
    
    # Generate candidates one by one to save memory
    for candidate_idx in range(num_candidates):
        try:
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    num_return_sequences=1,
                    # Add repetition penalty to improve diversity
                    repetition_penalty=1.1,
                )
            
            # Decode generated texts
            generated_texts = []
            for i, output in enumerate(outputs):
                try:
                    # Extract only the new tokens
                    prompt_length = inputs['input_ids'][i].shape[0]
                    if len(output) > prompt_length:
                        generated_part = output[prompt_length:]
                        generated_text = tokenizer.decode(generated_part, skip_special_tokens=True)
                        generated_texts.append(generated_text.strip())
                    else:
                        generated_texts.append("")
                except Exception as e:
                    print_log(f"Error decoding output {i}: {e}")
                    generated_texts.append("")
            
            # Store this round of candidates
            if candidate_idx == 0:
                all_candidates = [[text] for text in generated_texts]
            else:
                for i, text in enumerate(generated_texts):
                    if i < len(all_candidates):
                        all_candidates[i].append(text)
            
            # Clean up intermediate results
            del outputs
            clear_memory()
            
        except Exception as e:
            print_log(f"Error generating candidate {candidate_idx}: {e}")
            # Add empty candidates for this round
            if candidate_idx == 0:
                all_candidates = [[""] for _ in prompts]
            else:
                for i in range(len(prompts)):
                    if i < len(all_candidates):
                        all_candidates[i].append("")
    
    # Clean up
    del inputs
    clear_memory()
    
    return all_candidates

def generate_adapter_candidates(model_path_or_model, tokenizer, data: List[Dict], 
                              batch_size: int = 4, num_candidates: int = 5, 
                              device="cuda", use_model_path=False, base_model_name=None,
                              adapter_type="A"):
    """Generic function for generating adapter candidates"""
    
    print_log(f"Generating Adapter {adapter_type} candidates with batch_size={batch_size}")
    
    # Load model
    if use_model_path or isinstance(model_path_or_model, str):
        model, tokenizer = load_adapter_for_inference(model_path_or_model, base_model_name, tokenizer, device)
    else:
        model = model_path_or_model
        model.eval()
    
    # Create dataset
    dataset = GRPODataset(data, tokenizer, use_chat_template=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    all_candidates = {}
    processed_samples = 0
    
    try:
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Generating Adapter {adapter_type} candidates")):
            try:
                prompts = batch['prompt']
                premises = batch['premise']
                propositions = batch['proposition']
                
                # Generate candidates for this batch
                batch_candidates = generate_candidates_batch(
                    model, tokenizer, prompts, num_candidates, device=device
                )
                
                # Store results with consistent keys
                for i, (premise, proposition, candidates) in enumerate(zip(premises, propositions, batch_candidates)):
                    key = create_consistent_key(premise, proposition)
                    all_candidates[key] = candidates
                    processed_samples += 1
                
                # Periodic memory cleanup
                if batch_idx % 5 == 0:
                    clear_memory()
                    
            except Exception as e:
                print_log(f"Error processing batch {batch_idx}: {e}")
                continue
    
    except Exception as e:
        print_log(f"Error during candidate generation: {e}")
        raise
    
    finally:
        # Clean up model if it was loaded from path
        if use_model_path or isinstance(model_path_or_model, str):
            del model
            clear_memory()
    
    print_log(f"Generated candidates for {len(all_candidates)} samples (processed {processed_samples} total)")
    return all_candidates

def generate_adapter_a_candidates(model_path_or_model, tokenizer, data: List[Dict], 
                                batch_size: int = 4, num_candidates: int = 5, 
                                device="cuda", use_model_path=False, base_model_name=None):
    """Generate Adapter A candidates"""
    return generate_adapter_candidates(
        model_path_or_model, tokenizer, data, batch_size, num_candidates,
        device, use_model_path, base_model_name, adapter_type="A"
    )

def generate_adapter_b_candidates(model_path_or_model, tokenizer, data: List[Dict], 
                                batch_size: int = 4, num_candidates: int = 5, 
                                device="cuda", use_model_path=False, base_model_name=None):
    """Generate Adapter B candidates"""
    return generate_adapter_candidates(
        model_path_or_model, tokenizer, data, batch_size, num_candidates,
        device, use_model_path, base_model_name, adapter_type="B"
    )

def generate_candidates_streaming(model_path_or_model, tokenizer, data: List[Dict], 
                                output_file: str, batch_size: int = 2, num_candidates: int = 5, 
                                device="cuda", adapter_type="A"):
    """Memory-efficient streaming generation for very large datasets"""
    
    import json
    
    print_log(f"Starting streaming generation for Adapter {adapter_type}")
    
    # Load model
    if isinstance(model_path_or_model, str):
        model, tokenizer = load_adapter_for_inference(model_path_or_model, device=device)
    else:
        model = model_path_or_model
        model.eval()
    
    # Create dataset
    dataset = GRPODataset(data, tokenizer, use_chat_template=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Streaming file writing
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('{\n')
        first_item = True
        
        try:
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Streaming Adapter {adapter_type}")):
                try:
                    prompts = batch['prompt']
                    premises = batch['premise']
                    propositions = batch['proposition']
                    
                    # Generate candidates
                    batch_candidates = generate_candidates_batch(
                        model, tokenizer, prompts, num_candidates, device=device
                    )
                    
                    # Write to file immediately
                    for i, (premise, proposition, candidates) in enumerate(zip(premises, propositions, batch_candidates)):
                        key = create_consistent_key(premise, proposition)
                        
                        if not first_item:
                            f.write(',\n')
                        else:
                            first_item = False
                        
                        f.write(f'  {json.dumps(key, ensure_ascii=False)}: {json.dumps(candidates, ensure_ascii=False)}')
                        f.flush()  # Ensure data is written
                    
                    # Memory cleanup
                    if batch_idx % 3 == 0:
                        clear_memory()
                        
                except Exception as e:
                    print_log(f"Error in streaming batch {batch_idx}: {e}")
                    continue
        
        finally:
            f.write('\n}')
            
            # Final cleanup
            if isinstance(model_path_or_model, str):
                del model
                clear_memory()
    
    print_log(f"Streaming generation completed. Results saved to {output_file}")

# Utility function for memory monitoring
def monitor_memory_usage():
    """Monitor and log current memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print_log(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB, Peak: {max_allocated:.2f}GB")
    
    import psutil
    ram_usage = psutil.Process().memory_info().rss / 1024**3
    print_log(f"RAM Usage: {ram_usage:.2f}GB")
