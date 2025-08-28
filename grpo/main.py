# ./main.py

import argparse
import os
import wandb
import numpy as np
import torch
from model import load_model_and_tokenizer, create_lora_config, create_dual_adapters, format_input_prompt
from data import load_data, save_candidate_to_json
from generator import generate_adapter_a_candidates, generate_adapter_b_candidates
from train import train_adapter_a, train_adapter_b
from datetime import datetime

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

    return parser.parse_args()

def main():
    args = parse_args()

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

    print_log("Creating LoRA Adapters")
    lora_config = create_lora_config(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    adapter_a, adapter_b = create_dual_adapters(base_model, lora_config)

    print_log("Training Adapter A(ROUGE Optimizer)")
    adapter_a = train_adapter_a(
        adapter_a, tokenizer, train_data, val_data, args.output_dir, args
    )

    print_log("Generating Adapter A Candidates")
    adapter_a_candidates = generate_adapter_a_candidates(
        adapter_a, tokenizer, train_data, batch_size=args.batch_size, num_candidates=args.num_candidates
    ) 
    for i in range(len(train_data)):
        print_log(f"=== Input {i+1} ===")
        sample = train_data[i]
        key = f"{sample['input']['premise']} ||| {sample['input']['proposition']}"
        print_log(f"Input Prompt: {format_input_prompt(sample['input']['premise'], sample['input']['proposition'], sample['input']['label'])}")
        print_log(f"Adapter A Candidates: {adapter_a_candidates.get(key, [])}")
        if 'output' in sample:
            print_log(f"Reference Output: {sample['output']}")
        print_log("====================")

    # Save Adapter A candidates as JSON
    adapter_a_candidate_file = os.path.join(args.output_dir, "adapter_a_candidates.json")
    save_candidate_to_json(adapter_a_candidates, adapter_a_candidate_file)

    ppl_model = None
    if args.ppl_model:
        print_log(f"Loading PPL model: {args.ppl_model}")
        ppl_model, _ = load_model_and_tokenizer(args.ppl_model)

    print_log("Training Adapter B(Interactive BLEU, ROUGE-L, PPL Optimizer)")
    adapter_b = train_adapter_b(
        adapter_b, tokenizer, train_data, val_data, 
        adapter_a_candidates, args.output_dir, args, ppl_model
    )

    print_log("Saving Adapter Models")
    adapter_a.save_pretrained(os.path.join(args.output_dir, "adapter_a"))
    adapter_b.save_pretrained(os.path.join(args.output_dir, "adapter_b"))

    print_log("Generating Adapter B Candidates")
    adapter_b_candidates = generate_adapter_b_candidates(
        adapter_b, tokenizer, train_data, batch_size=args.batch_size, num_candidates=args.num_candidates, device=args.device
    ) 

    adapter_b_candidate_file = os.path.join(args.output_dir, "adapter_b_candidates.json")
    save_candidate_to_json(adapter_b_candidates, adapter_b_candidate_file)
    print_log("Training Complete")
    print_log(f"Models and candidates saved to {args.output_dir}")

    wandb.finish()

if __name__ == "__main__":
    main()
