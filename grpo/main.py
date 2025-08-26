import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Dual Adapter GRPO Training')
    parser.add_argument('--model_name', type=str, required=True, help='Base model name')
    parser.add_argument('--data_path', type=str, required=True, help='Training data path')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--epochs', type=int, default=3, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading base model and tokenizer...")
    base_model, tokenizer = load_base_model(args.model_name)
    
    print("Creating dual adapters...")
    lora_config = create_lora_config()
    adapter_a, adapter_b = create_dual_adapters(base_model, lora_config)
    
    print("Loading training data...")
    data_samples = load_nli_data(args.data_path)
    
    print("Training Adapter A (Accuracy-focused)...")
    adapter_a = train_adapter_grpo(
        adapter_a, tokenizer, data_samples, 
        reward_function=compute_adapter_a_reward,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size
    )
    
    print("Generating candidates from Adapter A...")
    adapter_a_candidates = batch_generate_adapter_a_candidates(
        adapter_a, tokenizer, data_samples
    )
    
    # Save Adapter A candidates
    candidates_path = os.path.join(args.output_dir, 'adapter_a_candidates.json')
    save_candidates_to_json(adapter_a_candidates, candidates_path)
    
    print("Training Adapter B (Diversity-focused)...")
    adapter_b_reward_fn = lambda gen, ref, a_cands: compute_adapter_b_reward(
        gen, ref, a_cands, adapter_b, tokenizer
    )
    
    adapter_b = train_adapter_grpo(
        adapter_b, tokenizer, data_samples,
        reward_function=adapter_b_reward_fn,
        adapter_a_candidates=adapter_a_candidates,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size
    )
    
    print("Saving models...")
    adapter_a.save_pretrained(os.path.join(args.output_dir, 'adapter_a'))
    adapter_b.save_pretrained(os.path.join(args.output_dir, 'adapter_b'))
    
    print("Training completed!")

if __name__ == "__main__":
    main()
