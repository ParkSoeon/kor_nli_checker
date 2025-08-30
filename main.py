# ./main.py

import argparse
import os
import wandb
import numpy as np
import torch
from model import load_model_and_tokenizer, create_lora_config, create_dual_adapters, format_input_prompt, save_adapter_safely
from data import load_data, save_candidate_to_json
from generator import generate_adapter_a_candidates, generate_adapter_b_candidates
from train import train_adapter_a, train_adapter_b
from datetime import datetime
import copy

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def print_log(message, prefix="LOG"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def parse_args():
    parser = argparse.ArgumentParser(description="Dual Adapter GRPO Trainer")

    parser.add_argument('--model_name', type=str, required=True, help='Pre-trained model name or path')
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data JSON file')
    parser.add_argument('--val_data', type=str, required=True, help='Path to validation data JSON file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save models and outputs')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for optimizer')
    parser.add_argument('--num_candidates', type=int, default=5, help='Number of candidates to generate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')

    parser.add_argument('--lambda1', type=float, default=0.5, help='Weight for ROUGE-1 in Adapter A reward')
    parser.add_argument('--lambda2', type=float, default=0.3, help='Weight for ROUGE-2 in Adapter A reward')
    parser.add_argument('--lambda3', type=float, default=0.2, help='Weight for ROUGE-L in Adapter A reward')

    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout rate')

    parser.add_argument('--ppl_model', type=str, required=True, help='Model name for PPL calculation in Adapter B reward')

    parser.add_argument('--adapter_a_only', action='store_true', help='Enable experiment mode with reduced data and epochs for quick testing')
    parser.add_argument('--adapter_b_only', action='store_true', help='Enable experiment mode with reduced data and epochs for quick testing')
    parser.add_argument('--full_exp', action='store_true', help='Disable experiment mode for full training')
    parser.add_argument('--adapter_a_candidate_file', type=str, default=None, help='Path to pre-generated Adapter A candidates JSON file')

    return parser.parse_args()

def run_adapter_a_experiment(args, base_model, tokenizer, train_data, val_data):
    print_log("Running Adapter A in Experiment Mode")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lora_config = create_lora_config(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout
    )
    adapter_a, _ = create_dual_adapters(base_model, lora_config)

    adapter_a = train_adapter_a(
        adapter_a, tokenizer, train_data, val_data, args.output_dir, args
    )
    adapter_a_val_candidates = generate_adapter_a_candidates(
        adapter_a, tokenizer, val_data, batch_size=args.batch_size, num_candidates=args.num_candidates, device=args.device
    )

    os.mkdir(args.output_dir, exist_ok=True)
    adapter_a_train_file = os.path.join(args.output_dir, "adapter_a_val_candidates_{timestamp}.json")
    save_candidate_to_json(adapter_a_train_candidates, adapter_a_train_file)

    adapter_a_model_dir = os.path.join(args.output_dir, f"adapter_a_final_{timestamp}")
    os.makedirs(adapter_a_model_dir, exist_ok=True)
    adapter_a.save_pretrained(adapter_a_model_dir)

    # # adapter_a.save_pretrained(os.path.join(args.output_dir, f"adapter_a_{timestamp}"))
    # adapter_a_model_dir = os.path.join(args.output_dir, f"adapter_a_{timestamp}")
    # save_success = save_adapter_safely(adapter_a, adapter_a_model_dir, model_name="adapter_a")

    print_log(f"Adapter A Model and Candidates saved to {args.output_dir}")
    print_log("Adapter A Training Complete")

    return adapter_a_candidates

def run_adapter_b_experiment(args, base_model, tokenizer, train_data, val_data, adapter_a_candidates):
    print_log("Running Adapter B in Experiment Mode")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Load PPL model for Adapter B reward
    ppl_model = None
    if args.ppl_model:
        print_log(f"Loading PPL model {args.ppl_model} for Adapter B reward")
        ppl_model, _ = load_model_and_tokenizer(args.ppl_model, device=args.device)

    lora_config = create_lora_config(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout
    )

    fresh_base_model = copy.deepcopy(base_model)
    _, adapter_b = create_dual_adapters(fresh_base_model, lora_config)

    adapter_b = train_adapter_b(
        adapter_b, tokenizer, train_data, val_data, 
        adapter_a_candidates, args.output_dir, args, ppl_model
    )

    print_log("Generating Adapter B Candidates for ALL data")
    adapter_b_candidates = generate_adapter_b_candidates(
        adapter_b, tokenizer, val_data, 
        batch_size=args.batch_size, 
        num_candidates=args.num_candidates, 
        device=args.device
    )

    adapter_b_val_file = os.path.join(args.output_dir, f"adapter_b_val_candidates_{timestamp}.json")
    save_candidate_to_json(adapter_b_candidates, adapter_b_val_file)

    adapter_b_model_dir = os.path.join(args.output_dir, f"adapter_b_final_{timestamp}")
    os.makedirs(adapter_b_model_dir, exist_ok=True)
    adapter_b.save_pretrained(adapter_b_model_dir)

    print_log(f"Adapter B Model and Candidates saved to {args.output_dir}")
    print_log("Adapter B Training Complete")

    return adapter_b_candidates

def combine_candidates_for_reranking(adapter_a_candidates, adapter_b_candidates, output_dir, timestamp):
    print_log("Combining Adapter A and Adapter B Candidates for Reranking")

    combined_candidates = {}

    all_keys = set(adapter_a_candidates.keys()) | set(adapter_b_candidates.keys())

    for key in adapter_a_candidates.keys():
        a_cands = adapter_a_candidates.get(key, [])
        b_cands = adapter_b_candidates.get(key, [])

        combined_candidates[key] = a_cands + b_cands

    combined_files = os.path.join(output_dir, f"candidates_for_reranking_{timestamp}.json")
    save_candidate_to_json(combined_candidates, combined_files)

    print_log(f"Combined candidates saved to {combined_files}")
    return combined_candidates


def main():
    args = parse_args()

    run_timestamp = get_timestamp() 

    wandb.init(project="2025HCLT(dual_adapter_grpo)", name=f"grpo_run_{get_timestamp()}")

    os.makedirs(args.output_dir, exist_ok=True)

    print_log("Starting Dual Adapter GRPO Training")
    base_model, tokenizer = load_model_and_tokenizer(args.model_name, args.device)

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

    if args.adapter_a_only:
        adapter_a_candidates, adapter_a_model_dir = run_adapter_a_experiment(args, base_model, tokenizer, train_data, val_data)

    elif args.adapter_b_only:
        if not args.adapter_a_candidates_file or not os.path.exists(args.adapter_a_candidate_file):
            raise ValueError("Adapter A candidate file must be provided and exist for Adapter B only mode.")

        print_log(f"Loading Adapter A candidates from {args.adapter_a_candidate_file}")
        adapter_a_candidates = load_candidates_from_json(args.adapter_a_candidate_file)
        adapter_b_candidates, adapter_b_model_dir = run_adapter_b_experiment(args, base_model, tokenizer, train_data, val_data, adapter_a_candidates)

        combine_candidates, combined_files = combine_candidates_for_reranking(adapter_a_candidates, adapter_b_candidates, args.output_dir)

    elif args.full_exp:
        adapter_a_candidates, adapter_a_model_dir = run_adapter_a_experiment(args, base_model, tokenizer, train_data, val_data)
        adapter_b_candidates, adapter_b_model_dir = run_adapter_b_experiment(args, base_model, tokenizer, train_data, val_data, adapter_a_candidates)

        combine_candidates, combined_files = combine_candidates_for_reranking(adapter_a_candidates, adapter_b_candidates, args.output_dir)
    
    print_log("Training Complete")
    print_log(f"Models and candidates saved to {args.output_dir}")

    wandb.finish()

if __name__ == "__main__":
    main()
