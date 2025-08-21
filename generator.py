import os
import sys
import json
import argparse
import torch
import wandb
import numpy as np
import random
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForSeq2Seq, 
    DataCollatorForLanguageModeling
)
from evaluate import load

def get_timestamp():
    """Get current timestamp for file naming"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def print_log(message, prefix="LOG"):
    """Print message with timestamp and prefix"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{prefix}] {message}")

def get_parse():
    parser = argparse.ArgumentParser(description="Generate text using a pre-trained model.")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, required=True, help="Name of the pre-trained model.")
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Name of the tokenizer (default: same as model).")
    parser.add_argument("--model_type", type=str, choices=["causal", "seq2seq"], default="causal", help="Model type.")
    parser.add_argument("--cache_dir", type=str, default="/home/nlplab/hdd1/cache_dir", help="Cache directory.")
    
    # Data arguments
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training file.")
    parser.add_argument("--dev_path", type=str, required=True, help="Path to the validation file.")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test file.")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory.")
    parser.add_argument("--max_input_length", type=int, default=512, help="Maximum input length.")
    parser.add_argument("--max_output_length", type=int, default=256, help="Maximum target length.")
    
    # Training arguments
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Training batch size per device.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Evaluation batch size per device.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save steps.")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluation steps.")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Save total limit.")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    
    # Generation arguments
    parser.add_argument("--num_cands", type=int, default=3, help="Number of candidates to generate.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling.")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty.")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling for generation.")
    
    # Few-shot arguments
    parser.add_argument("--num_fewshot", type=int, default=3, help="Number of few-shot examples.")
    parser.add_argument("--fewshot_seed", type=int, default=42, help="Seed for few-shot example selection.")
    
    # General arguments
    parser.add_argument("--mode", type=str, choices=["train", "inf", "test"], default="train", help="Mode of operation.")
    parser.add_argument("--wandb_project", type=str, default="text_generation", help="WandB project name.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--use_chat_template", action="store_true", default=True, help="Use chat template for formatting.")
    
    return parser.parse_args()

class EntailmentDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_input_length=512, max_output_length=256, 
                 model_type="causal", mode="train", use_chat_template=True, 
                 fewshot_examples=None, num_fewshot=3):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.model_type = model_type
        self.mode = mode
        self.use_chat_template = use_chat_template
        self.fewshot_examples = fewshot_examples if fewshot_examples else []
        self.num_fewshot = min(num_fewshot, len(self.fewshot_examples))

        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line.strip()) for line in f if line.strip()]

        print_log(f"Loaded {len(self.data)} examples from {data_path}")
        print_log(f"Using {self.num_fewshot} few-shot examples")
        print_log(f"Chat template: {self.use_chat_template}")

    def create_fewshot_prompt(self, premise, proposition, label=None):
        """Create few-shot prompt with examples"""
        examples = []

        # Add few-shot examples
        for example in self.fewshot_examples[:self.num_fewshot]:
            ex_premise = example['input']['premise']
            ex_proposition = example['input']['proposition']
            ex_label = example['input']['label']  # Fixed: was looking at 'output']['label']
            ex_output = example['output']

            examples.append({
                "premise": ex_premise,
                "proposition": ex_proposition,
                "label": ex_label,
                "explanation": ex_output
            })

        if self.use_chat_template:
            messages = [
                {"role": "system", "content": "당신은 한국어 자연어 추론(NLI) 전문가입니다. 주어진 전제와 가설을 분석하여 함의 관계를 설명해주세요."}
            ]

            # Add few-shot examples
            for ex in examples:
                user_message = f"[전제] {ex['premise']}\n[가설] {ex['proposition']}\n[관계] {ex['label']}"
                assistant_message = f"{ex['explanation']}"
                messages.extend([
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_message}
                ])

            # Add current example
            current_message = f"[전제] {premise}\n[가설] {proposition}\n[관계] {label}"
            messages.append({"role": "user", "content": current_message})

            return messages
        else:
            # Traditional prompt format (simplified implementation)
            prompt_parts = ["다음은 자연어 추론 작업입니다. 전제와 가설 사이의 관계를 분석하고 설명해주세요.\n"]
            
            # Add examples
            for ex in examples:
                prompt_parts.append(f"[전제] {ex['premise']}")
                prompt_parts.append(f"[가설] {ex['proposition']}")
                prompt_parts.append(f"[관계] {ex['label']}")
                prompt_parts.append(f"[설명] {ex['explanation']}\n")
            
            # Add current example
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
                messages.append({"role": "assistant", "content": output})
                full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
                
                # Log first few examples for debugging
                if idx < 3:
                    print_log(f"Training example {idx}:")
                    print_log(f"Full prompt:\n{full_text[:500]}...")
                    print_log(f"Target: {output}")

                encoding = self.tokenizer(
                    full_text,
                    truncation=True,
                    max_length=self.max_input_length + self.max_output_length,
                    padding=False,
                    return_tensors="pt"
                )

                labels = encoding["input_ids"].clone()

                # Calculate input length to mask properly
                input_only = self.tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
                input_encoding = self.tokenizer(input_only, truncation=True, max_length=self.max_input_length)
                input_len = len(input_encoding["input_ids"])

                # Mask input tokens
                labels[0, :input_len] = -100 

                return {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "labels": labels.squeeze(),
                }

            else:  # inference/test mode
                input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                # Log first few examples for debugging
                if idx < 3:
                    print_log(f"Inference example {idx}:")
                    print_log(f"Input prompt:\n{input_text[:500]}...")
                
                input_encoding = self.tokenizer(
                    input_text,
                    truncation=True,
                    max_length=self.max_input_length,
                    padding=False,
                    return_tensors="pt"
                )

                return {
                    "input_ids": input_encoding["input_ids"].squeeze(),
                    "attention_mask": input_encoding["attention_mask"].squeeze(),
                    "target": output,
                    "id": item.get('id', idx)
                }
        else:
            # Non-chat template mode
            input_prompt = self.create_fewshot_prompt(premise, proposition, label)
            
            if self.mode == "train":
                full_text = f"{input_prompt} {output}"
                
                # Log first few examples
                if idx < 3:
                    print_log(f"Training example {idx} (non-chat):")
                    print_log(f"Full text: {full_text[:500]}...")
                
                encoding = self.tokenizer(
                    full_text,
                    truncation=True,
                    max_length=self.max_input_length + self.max_output_length,
                    padding=False,
                    return_tensors="pt"
                )
                
                labels = encoding['input_ids'].clone()
                
                # Mask input part
                input_encoding = self.tokenizer(input_prompt, truncation=True, max_length=self.max_input_length)
                input_len = len(input_encoding['input_ids'])
                labels[0, :input_len] = -100
                
                return {
                    "input_ids": encoding['input_ids'].squeeze(),
                    "attention_mask": encoding['attention_mask'].squeeze(),
                    "labels": labels.squeeze()
                }
            else:
                # For inference
                if idx < 3:
                    print_log(f"Inference example {idx} (non-chat):")
                    print_log(f"Input prompt: {input_prompt[:500]}...")
                
                input_encoding = self.tokenizer(
                    input_prompt,
                    truncation=True,
                    max_length=self.max_input_length,
                    padding=False,
                    return_tensors="pt"
                )
                
                return {
                    "input_ids": input_encoding['input_ids'].squeeze(),
                    "attention_mask": input_encoding['attention_mask'].squeeze(),
                    "target": output,
                    "id": item.get('id', idx)
                }

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print_log(f"Seed set to {seed}")

def load_model_and_tokenizer(args):
    tokenizer_name = args.tokenizer_name or args.model_name
    print_log(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print_log("Set pad_token to eos_token")

    print_log(f"Loading model: {args.model_name}")
    print_log(f"Model type: {args.model_type}")
    
    if args.model_type == "causal":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            torch_dtype=torch.float16 if args.fp16 else torch.float32,
            device_map="auto" 
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            torch_dtype=torch.float16 if args.fp16 else torch.float32,
            device_map="auto"
        )

    print_log(f"Model loaded. Parameters: {model.num_parameters():,}")
    return model, tokenizer

def prepare_fewshot_examples(train_path, seed, num_examples=10):
    print_log(f"Preparing few-shot examples from {train_path}")
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = [json.loads(line.strip()) for line in f if line.strip()]
    
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
        print_log(f"  Premise: {first_ex['input']['premise'][:100]}...")
        print_log(f"  Proposition: {first_ex['input']['proposition'][:100]}...")
        print_log(f"  Label: {first_ex['input']['label']}")
        print_log(f"  Output: {first_ex['output'][:100]}...")
    
    return selected_examples

def train_model(args):
    print_log("Starting training mode")
    
    # Prepare few-shot examples
    fewshot_examples = prepare_fewshot_examples(args.train_path, args.fewshot_seed, args.num_fewshot * 3)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)
    
    # Create datasets
    print_log("Creating training dataset...")
    train_dataset = EntailmentDataset(
        args.train_path, tokenizer, args.max_input_length, args.max_output_length,
        args.model_type, "train", args.use_chat_template, fewshot_examples, args.num_fewshot
    )
    
    print_log("Creating evaluation dataset...")
    eval_dataset = EntailmentDataset(
        args.dev_path, tokenizer, args.max_input_length, args.max_output_length,
        args.model_type, "train", args.use_chat_template, fewshot_examples, args.num_fewshot
    )
    
    # Data collator
    if args.model_type == "causal":
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        print_log("Using DataCollatorForLanguageModeling")
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            pad_to_multiple_of=8
        )
        print_log("Using DataCollatorForSeq2Seq")
    
    # Create output directory with timestamp
    timestamp = get_timestamp()
    output_dir = os.path.join(args.output_dir, f"model_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print_log(f"Output directory: {output_dir}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        fp16=args.fp16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        remove_unused_columns=False,
        report_to="wandb" if args.wandb_project else "none",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    print_log("Starting training...")
    trainer.train()
    
    print_log("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print_log(f"Training complete. Model saved to {output_dir}")
    return model, tokenizer

def generate_candidates(model, tokenizer, test_dataset, args):
    """Generate multiple candidates for each test example"""
    print_log("Starting candidate generation")
    model.eval()
    results = []
    
    dataloader = DataLoader(
        test_dataset, 
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=lambda batch: {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'targets': [item['target'] for item in batch],
            'ids': [item['id'] for item in batch]
        }
    )
    
    print_log(f"Generation settings:")
    print_log(f"  Max new tokens: {args.max_new_tokens}")
    print_log(f"  Num candidates: {args.num_cands}")
    print_log(f"  Temperature: {args.temperature}")
    print_log(f"  Top-k: {args.top_k}")
    print_log(f"  Top-p: {args.top_p}")
    print_log(f"  Do sample: {args.do_sample}")
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating")):
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        
        # Log first batch details
        if batch_idx == 0:
            print_log(f"First batch input shape: {input_ids.shape}")
            print_log(f"First input tokens: {input_ids[0][:20].tolist()}")
        
        # Generate multiple candidates
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=args.num_cands,
                temperature=args.temperature if args.do_sample else None,
                top_k=args.top_k if args.do_sample else None,
                top_p=args.top_p if args.do_sample else None,
                repetition_penalty=args.repetition_penalty,
                do_sample=args.do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode outputs
        batch_size = input_ids.size(0)
        for i in range(batch_size):
            candidates = []
            for j in range(args.num_cands):
                idx = i * args.num_cands + j
                output_ids = outputs[idx][input_ids.size(1):]  # Remove input part
                generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
                candidates.append(generated_text.strip())
            
            results.append({
                "id": batch['ids'][i],
                "target": batch['targets'][i],
                "candidates": candidates
            })
            
            # Log first few results
            if len(results) <= 3:
                print_log(f"Generated example {len(results)}:")
                print_log(f"  Target: {batch['targets'][i][:100]}...")
                print_log(f"  Candidate 1: {candidates[0][:100]}...")
    
    print_log(f"Generated {len(results)} results")
    return results

def evaluate_results(results, args):
    """Evaluate generated results using various metrics"""
    print_log("Starting evaluation")
    
    # Load metrics
    rouge_metric = load("rouge")
    bertscore_metric = load("bertscore")
    
    # Prepare data for evaluation
    targets = []
    best_candidates = []
    all_candidates = []
    
    for result in results:
        target = result['target']
        candidates = result['candidates']
        
        targets.append(target)
        best_candidates.append(candidates[0])  # Use first candidate as best
        all_candidates.extend(candidates)
    
    print_log(f"Evaluating {len(targets)} examples")
    
    # ROUGE evaluation
    print_log("Computing ROUGE scores...")
    rouge_results = rouge_metric.compute(
        predictions=best_candidates,
        references=targets,
        use_stemmer=True
    )
    
    # BERTScore evaluation
    print_log("Computing BERTScore...")
    bertscore_results = bertscore_metric.compute(
        predictions=best_candidates,
        references=targets,
        model_type="klue/bert-base",  # Korean BERT
        lang="ko"
    )
    
    # Calculate averages
    eval_results = {
        "rouge1": rouge_results['rouge1'],
        "rouge2": rouge_results['rouge2'],
        "rougeL": rouge_results['rougeL'],
        "bertscore_precision": np.mean(bertscore_results['precision']),
        "bertscore_recall": np.mean(bertscore_results['recall']),
        "bertscore_f1": np.mean(bertscore_results['f1']),
        "num_examples": len(results),
        "num_candidates_per_example": args.num_cands
    }
    
    print_log("Evaluation completed")
    return eval_results

def main():
    args = get_parse()
    set_seed(args.seed)
    
    print_log("Starting main execution")
    print_log(f"Mode: {args.mode}")
    print_log(f"Model: {args.model_name}")
    print_log(f"Use chat template: {args.use_chat_template}")
    
    # Create timestamped output directory
    timestamp = get_timestamp()
    
    # Initialize wandb
    if args.wandb_project and args.mode == "train":
        wandb.init(
            project=args.wandb_project,
            config=vars(args)
        )
        print_log("Wandb initialized")
    
    if args.mode == "train":
        model, tokenizer = train_model(args)
        
    elif args.mode in ["inf", "test"]:
        print_log("Starting inference mode")
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(args)
        
        # Prepare few-shot examples
        fewshot_examples = prepare_fewshot_examples(args.train_path, args.fewshot_seed, args.num_fewshot * 3)
        
        # Create test dataset
        print_log("Creating test dataset...")
        test_dataset = EntailmentDataset(
            args.test_path, tokenizer, args.max_input_length, args.max_output_length,
            args.model_type, "test", args.use_chat_template, fewshot_examples, args.num_fewshot
        )
        
        # Generate candidates
        results = generate_candidates(model, tokenizer, test_dataset, args)
        
        # Create output directory with timestamp
        output_dir = os.path.join(args.output_dir, f"results_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        output_file = os.path.join(output_dir, f"generated_results_{timestamp}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print_log(f"Results saved to {output_file}")
        
        # Evaluate results
        eval_results = evaluate_results(results, args)
        
        # Save evaluation results
        eval_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=2)
        
        print_log("Evaluation Results:")
        for key, value in eval_results.items():
            if isinstance(value, float):
                print_log(f"  {key}: {value:.4f}")
            else:
                print_log(f"  {key}: {value}")
        
        # Log to wandb if available
        if args.wandb_project:
            wandb.init(project=args.wandb_project, config=vars(args))
            wandb.log(eval_results)
            wandb.finish()
            print_log("Results logged to wandb")
    
    print_log("Execution completed!")

if __name__ == "__main__":
    main()
