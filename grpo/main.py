# ./main.py

import argparse
import os
import wandb
import numpy as np
import torch
import gc
from model import load_model_and_tokenizer, create_lora_config, create_dual_adapters, format_input_prompt
from data import load_data, save_candidate_to_format, load_candidates_from_json, save_candidate_to_json, save_combined_cadidates
from generator import generate_adapter_a_candidates, generate_adapter_b_candidates
from train import train_adapter_a, train_adapter_b
from datetime import datetime
import copy
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers import AutoTokenizer

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def print_log(message, prefix="LOG"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
def parse_args():
    parser = argparse.ArgumentParser(description="Dual Adapter GRPO Trainer")

    parser.add_argument('--model_name', type=str, required=True, help='Pre-trained model name or path')
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data JSON file')
    parser.add_argument('--val_data', type=str, required=True, help='Path to validation data JSON file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save models and outputs')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    
    parser.add_argument('--batch_size', type=int, default=5, help='Training batch size')
    parser.add_argument('--batch_inf_size', type=int, default=1, help='Inference batch size for candidate generation')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for optimizer')
    parser.add_argument('--num_candidates', type=int, default=5, help='Number of candidates to generate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--do_sample', action='store_true', help='Enable do_sample for candidate generation')

    parser.add_argument('--lambda1', type=float, default=0.5, help='Weight for ROUGE-1 in Adapter A reward')
    parser.add_argument('--lambda2', type=float, default=0.3, help='Weight for ROUGE-2 in Adapter A reward')
    parser.add_argument('--lambda3', type=float, default=0.2, help='Weight for ROUGE-L in Adapter A reward')

    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout rate')

    parser.add_argument('--ppl_model', type=str, required=True, help='Model name for PPL calculation in Adapter B reward')

    parser.add_argument('--train', action='store_true', help='Enable training mode')
    parser.add_argument('--inf', action='store_true', help='Enable inference mode')
    parser.add_argument('--adapter_a_only', action='store_true', help='Enable experiment mode with reduced data and epochs for quick testing')
    parser.add_argument('--adapter_b_only', action='store_true', help='Enable experiment mode with reduced data and epochs for quick testing')
    parser.add_argument('--full_exp', action='store_true', help='Disable experiment mode for full training')
    
    parser.add_argument('--adapter_a_path', type=str, default=None, help='Path to pre-trained Adapter A model (for Adapter B training)')
    parser.add_argument('--adapter_b_path', type=str, default=None, help='Path to pre-trained Adapter B model (for inference)')
    parser.add_argument('--adapter_a_candidate_file', type=str, default=None, help='Path to pre-generated Adapter A candidates JSON file')

    parser.add_argument('--use_chat_template', action='store_true', help='Use chat template for input formatting')

    parser.add_argument('--adapter_b_data', type=str, default=None, help='Path to data file with Adapter A candidates for Adapter B training')

    return parser.parse_args()

def run_adapter_a_experiment(args, base_model, tokenizer, train_data, val_data):
    print_log("Running Adapter A in Experiment Mode")
    timestamp = get_timestamp()

    lora_config = create_lora_config(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout
    )
    adapter_a = get_peft_model(base_model, lora_config)

    adapter_a = train_adapter_a(
        adapter_a, tokenizer, train_data, val_data, args.output_dir, args
    )

    adapter_a_model_dir = os.path.join(args.output_dir, f"adapter_a_final_{timestamp}")
    os.makedirs(adapter_a_model_dir, exist_ok=True)
    adapter_a.save_pretrained(adapter_a_model_dir)

    print_log(f"Adapter A Model saved to {args.output_dir}")
    print_log("Adapter A Training Complete")

    if not args.train:
        del adapter_a
        clear_memory()
        return adapter_a_model_dir

    return adapter_a, adapter_a_model_dir

def run_adapter_a_inference(args, adapter_a_path, tokenizer, data):
    timestamp = get_timestamp()

    adapter_a_candidates = generate_adapter_a_candidates(
        adapter_a_path, tokenizer, data,
        batch_size=args.batch_inf_size,
        num_candidates=args.num_candidates,
        device=args.device,
        base_model_name=args.model_name,
        # 수정사항
        # use_model_path=True,
        use_chat_template=args.use_chat_template,
        
    )

    candidate_file = os.path.join(args.output_dir, f"adapter_a_candidates_inference_{timestamp}.json")
    save_candidate_to_json(adapter_a_candidates, candidate_file)
    print_log(f"Adapter A candidates saved to {candidate_file}")

    formatted_file = os.path.join(args.output_dir, f"adapter_a_candidates_formatted_{timestamp}.json")
    save_candidate_to_format(adapter_a_candidates, data, formatted_file, adapter_name="adapter_a")
    print_log(f"Adapter A candidates formatted and saved to {formatted_file}")

    return adapter_a_candidates, candidate_file

def run_adapter_b_experiment(args, base_model, tokenizer, adapter_b_data_file, val_data):
    print_log("Running Adapter B with pre-generated Adapter A candidates")
    timestamp = get_timestamp()

    train_data, adapter_a_candidates, reference_map = load_adapter_b_data(adapter_b_data_file)
    
    ppl_model = None
    if args.ppl_model:
        print_log(f"Loading PPL model {args.ppl_model} for Adapter B reward")
        ppl_model, _ = load_model_and_tokenizer(args.ppl_model, device=args.device)

    lora_config = create_lora_config(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout
    )

    adapter_b = get_peft_model(copy.deepcopy(base_model), lora_config)

    adapter_b = train_adapter_b(
        adapter_b, tokenizer, train_data, val_data, 
        adapter_a_candidates, args.output_dir, args, ppl_model
    )

    adapter_b_model_dir = os.path.join(args.output_dir, f"adapter_b_final_{timestamp}")
    os.makedirs(adapter_b_model_dir, exist_ok=True)
    adapter_b.save_pretrained(adapter_b_model_dir)

    print_log(f"Adapter B Model saved to {adapter_b_model_dir}")
    print_log("Adapter B Training Complete")

    if ppl_model:
        del ppl_model
        clear_memory()

    if not args.train:
        del adapter_b
        clear_memory()
        return adapter_b_model_dir

    return adapter_b, adapter_b_model_dir

def run_adapter_b_inference(args, adapter_b_path, tokenizer, data):
    timestamp = get_timestamp()

    adapter_b_candidates = generate_adapter_b_candidates(
        adapter_b_path, tokenizer, data,
        batch_size=args.batch_inf_size,
        num_candidates=args.num_candidates,
        device=args.device,
        use_model_path=True,
        base_model_name=args.model_name
    )

    candidate_file = os.path.join(args.output_dir, f"adapter_b_candidates_inference_{timestamp}.json")
    save_candidate_to_json(adapter_b_candidates, candidate_file)
    print_log(f"Adapter B candidates saved to {candidate_file}")

    formatted_file = os.path.join(args.output_dir, f"adapter_b_candidates_formatted_{timestamp}.json")
    save_candidate_to_format(adapter_b_candidates, data, formatted_file, adapter_name="adapter_b")
    print_log(f"Adapter B candidates formatted and saved to {formatted_file}")

    return adapter_b_candidates, candidate_file

def combine_candidates_for_reranking(adapter_a_candidates, adapter_b_candidates, original_data, output_dir):
    print_log("Combining Adapter A and Adapter B Candidates for Reranking")
    timestamp = get_timestamp()

    combined_candidates = {}

    all_keys = set(adapter_a_candidates.keys()) | set(adapter_b_candidates.keys())

    for key in all_keys:
        a_cands = adapter_a_candidates.get(key, [])
        b_cands = adapter_b_candidates.get(key, [])
        combined_candidates[key] = a_cands + b_cands

    combined_raw_file = os.path.join(output_dir, f"combined_candidates_raw_{timestamp}.json")
    save_candidate_to_json(combined_candidates, combined_raw_file)
    print_log(f"Combined raw candidates saved to {combined_raw_file}")
    
    combined_formatted_file = os.path.join(output_dir, f"combined_candidates_formatted_{timestamp}.json")
    save_combined_cadidates(adapter_a_candidates, adapter_b_candidates, original_data, combined_formatted_file)
    print_log(f"Combined candidates saved to {combined_formatted_file}")

    return combined_candidates, combined_formatted_file

def main():
    args = parse_args()

    run_timestamp = get_timestamp() 

    wandb.init(project="2025HCLT(dual_adapter_grpo)", name=f"grpo_run_{get_timestamp()}")

    os.makedirs(args.output_dir, exist_ok=True)

    print_log("Starting Dual Adapter GRPO Training")

    # Load datasets
    print_log("Loading Data")
    train_data = load_data(args.train_data)
    val_data = load_data(args.val_data)
    print_log(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

    print_log("=== Data Samples ===")
    if len(train_data) > 0:
        sample = train_data[0]
        print_log(f"Sample Keys: {list(sample.keys())}")
        print_log(f"Sample Premise: {sample['input']['premise']}")
        print_log(f"Sample Proposition: {sample['input']['proposition']}")
        print_log(f"Sample Label: {sample['input']['label']}")
        if 'output' in sample:
            print_log(f"Sample Reference Output: {sample['output']}")
    print_log("====================")

    base_model = None
    tokenizer = None

    if not args.inf:
        print_log("Loading Base Model for Training")
        base_model, tokenizer = load_model_and_tokenizer(args.model_name, args.device)
    else:
        print_log("Inference Mode: Skipping Base Model Loading")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.train:
        print_log("Training Mode Enabled")

        if args.adapter_a_only:
            adapter_a, adapter_a_dir = run_adapter_a_experiment(args, base_model, tokenizer, train_data, val_data)

        elif args.adapter_b_only:
            if not args.adapter_a_candidate_file or not os.path.exists(args.adapter_a_candidate_file):
                raise ValueError("Adapter A candidate file must be provided and exist for Adapter B only mode.")

            print_log(f"Loading Adapter A candidates from {args.adapter_a_candidate_file}")
            adapter_a_candidates = load_candidates_from_json(args.adapter_a_candidate_file)
            adapter_b, adapter_b_dir = run_adapter_b_experiment(args, base_model, tokenizer, adapter_b_data_file, val_data)

            combine_candidates, combined_files = combine_candidates_for_reranking(adapter_a_candidates, {}, train_data, args.output_dir)

        elif args.full_exp:
            adapter_a, adapter_a_dir = run_adapter_a_experiment(args, base_model, tokenizer, train_data, val_data)

            del adapter_a
            clear_memory()

            adapter_a_candidates, _ = run_adapter_a_inference(args, adapter_a_dir, tokenizer, train_data)
            adapter_b, adapter_b_dir = run_adapter_b_experiment(args, base_model, tokenizer, train_data, val_data)
            
            combine_candidates, combined_files = combine_candidates_for_reranking(adapter_a_candidates, {}, train_data, args.output_dir)
    
    elif args.inf:
        print_log("Inference Mode Enabled")

        if args.adapter_a_only:
            adapter_a_candidates, _ = run_adapter_a_inference(args, args.adapter_a_path, tokenizer, val_data)

        elif args.adapter_b_only:
            adapter_b_candidates, _ = run_adapter_b_inference(args, args.adapter_b_path, tokenizer, val_data)

        elif args.full_exp:
            adapter_a_candidates, _ = run_adapter_a_inference(args, args.adapter_a_path, tokenizer, val_data)
            adapter_b_candidates, _ = run_adapter_b_inference(args, args.adapter_b_path, tokenizer, val_data)
            combine_candidates, combined_files = combine_candidates_for_reranking(adapter_a_candidates, adapter_b_candidates, val_data, args.output_dir)
    
    else:
        print_log("Full Pipeline Mode Enabled")

        if args.adapter_a_only:
            adapter_a_dir = run_adapter_a_experiment(args, base_model, tokenizer, train_data, val_data)
            clear_memory()
            adapter_a_candidates, _ = run_adapter_a_inference(args, adapter_a_dir, tokenizer, val_data)

        elif args.full_exp:
            adapter_a_dir = run_adapter_a_experiment(args, base_model, tokenizer, train_data, val_data)
            clear_memory()
            adapter_a_candidates, _ = run_adapter_a_inference(args, adapter_a_dir, tokenizer, train_data)

            adapter_b_dir = run_adapter_b_experiment(args, base_model, tokenizer, adapter_b_data_file, val_data)
            clear_memory()
            adapter_b_candidates, _ = run_adapter_b_inference(args, adapter_b_dir, tokenizer, val_data)

            combine_candidates, combined_files = combine_candidates_for_reranking(adapter_a_candidates, adapter_b_candidates, val_data, args.output_dir)

    print_log("Pipeline Completed")
    print_log(f"Models and candidates saved to {args.output_dir}")

    if not args.inf:
        wandb.finish()

if __name__ == "__main__":
    main()
