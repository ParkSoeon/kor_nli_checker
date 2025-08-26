# ./main.py

import argparse
import os
import wandb
import numpy as np
import torch

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

    return parser.parse_args()

def main():

    wandb.init(project="2025HCLT--dual_adapter_grpo)", name=f"grpo_run_{get_timestamp()}")

    os.makedirs(args.output_dir, exist_ok=True)
    args = parse_args()

    print_log("Starting Dual Adapter GRPO Training")
    base_model, tokenizer = load_model_and_tokenizer(args.model_name)

    print_log("Loading Data")
    train_data = load_data(args.train_data)
    val_data = load_data(args.val_data)
    print_log(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

    print_log("Training Adapter A(ROUGE Optimizer)")
    adapter_a = train_adapter_grpo(
        adapter_a, base_model, tokenizer, train_data, val_data, args ####
    )

    print_log("Generating Adapter A Candidates")
    adapter_a_candidates = generate_adapter_a_candidates(
        adapter_a, tokenizer, train_data, batch_size=args.batch_size, num_candidates=args.num_candidates
    ) ####

    # Save Adapter A candidates as JSON
    adapter_a_candidate_file = os.path.join(args.output_dir, "adapter_a_candidates.json")
    save_candidate_to_json(adapter_a_candidates, adapter_a_candidate_file)

    print_log("Training Adapter B(Interactive BLEU, ROUGE-L, PPL Optimizer)")
    score_adapter_b_reward = lambda generated, references: compute_adapter_b_reward(
        generated, references, adapter_a_candidates, tokenizer, alpha=0.5, beta=0.3, gamma=0.2
    ) ####
    adapter_b = train_adapter_grpo(
        adapter_b, base_model, tokenizer, train_data, val_data, args, reward_function=score_adapter_b_reward
    ) ####

    print_log("Saving Adapter Models")
    adapter_a.save_pretrained(os.path.join(args.output_dir, "adapter_a"))
    adapter_b.save_pretrained(os.path.join(args.output_dir, "adapter_b"))

    print_log("Generating Adapter B Candidates")
    adapter_b_candidates = generate_adapter_b_candidates(
        adapter_b, tokenizer, train_data, batch_size=args.batch_size, num_candidates=args.num_candidates
    ) ####
    adapter_b_candidate_file = os.path.join(args.output_dir, "adapter_b_candidates.json")
    save_candidate_to_json(adapter_b_candidates, adapter_b_candidate_file)
    print_log("Training Complete")
    print_log(f"Models and candidates saved to {args.output_dir}")
    wandb.finish()

if __name__ == "__main__":
    main()
