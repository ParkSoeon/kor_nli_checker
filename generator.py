# generator.py

import os
import sys
import json
import argparse
import torch
import wandb
import numpy as np
import random
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM, 
    TrainingArguments, 
    Trainer, 
    TrainerCallback,
    EarlyStoppingCallback
)

from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType,
    PeftModel 
)

from evaluate import load

from data import (
    EntailmentDataset,
    prepare_fewshot_examples,
    create_datasets,
    create_data_collator
)

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def print_log(message, prefix="LOG"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{prefix}] {message}")

def get_model_id_from_path(model_path):
    return model_path.split('/')[-1].replace('-', '_')

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
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging steps.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save steps.")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluation steps.")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Save total limit.")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps.")

    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for efficient fine-tuning.")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout.")
    parser.add_argument("--lora_target_modules", nargs='+', default=None, help="Target modules for LoRA (e.g., q_proj k_proj v_proj o_proj).")
    parser.add_argument("--lora_task_type", type=str, default="CAUSAL_LM", choices=["CAUSAL_LM", "SEQ_2_SEQ_LM"], help="LoRA task type.")
    parser.add_argument("--lora_model_path", type=str, default=None, help="Path to pre-trained LoRA model for inference.")
        
    # Generation arguments
    parser.add_argument("--num_cands", type=int, default=3, help="Number of candidates to generate.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling.")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty.")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling for generation.")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search (1 for greedy decoding).")
    
    # Few-shot arguments
    parser.add_argument("--num_fewshot", type=int, default=3, help="Number of few-shot examples.")
    parser.add_argument("--fewshot_seed", type=int, default=42, help="Seed for few-shot example selection.")
    
    # General arguments
    parser.add_argument("--mode", type=str, choices=["train", "inf", "test", "train_and_test"], default="train_and_test", help="Mode of operation.")
    parser.add_argument("--wandb_project", type=str, default="2025HCLT", help="WandB project name.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--use_chat_template", action="store_true", default=True, help="Use chat template for formatting.")
    
    # Dynamic padding arguments
    parser.add_argument("--pad_to_multiple_of", type=int, default=8, help="Pad sequences to multiple of this number.")
    
    parser.add_argument("--rouge1_weight", type=float, default=0.6, help="Weight for ROUGE-1 in evaluation.")
    parser.add_argument("--rouge2_weight", type=float, default=0.2, help="Weight for ROUGE-2 in evaluation.")
    parser.add_argument("--rougeL_weight", type=float, default=0.2, help="Weight for ROUGE-L in evaluation.")

    return parser.parse_args()

class CustomCallback(TrainerCallback):
    def __init__(self, args, eval_dataset, tokenizer, fewshot_examples):
        self.args = args
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.fewshot_examples = fewshot_examples
        self.rouge_metric = load("rouge")
        self.evaluation_results = []

        self.best_rouge_score = 0.0
        self.best_checkpoint = None
        self.eval_results = []

    def on_evaluate(self, args, state, control, **kwargs):
        print_log("Starting evaluation...")

        model = kwargs.get('model')
        model.eval()
        predictions = []
        references = []

        # Create evaluation dataset for testing
        eval_dataset_for_eval = EntailmentDataset(
            data_path=self.args.dev_path,
            tokenizer=self.tokenizer,
            max_input_length=self.args.max_input_length,
            max_output_length=self.args.max_output_length,
            model_type=self.args.model_type,
            mode="test",  # Use test mode for evaluation
            use_chat_template=self.args.use_chat_template,
            fewshot_examples=self.fewshot_examples,
            num_fewshot=self.args.num_fewshot
        )

        # Use subset for faster evaluation during training
        eval_subset = torch.utils.data.Subset(eval_dataset_for_eval, range(min(10, len(eval_dataset_for_eval))))
        eval_dataloader = DataLoader(eval_subset, batch_size=1, shuffle=False)

        for batch_index, batch_data in enumerate(eval_dataloader):
            input_ids = batch_data['input_ids'].to(model.device)
            attention_mask = batch_data['attention_mask'].to(model.device)
            target = batch_data['target'][0] if isinstance(batch_data['target'], list) else batch_data['target']
            
            input_text = self.tokenizer.decode(input_ids.squeeze(), skip_special_tokens=False)
            print_log(f"Input text: {input_text}") 
            print_log(f"Target: {target}")

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.args.max_new_tokens,
                    temperature=0.7,
                    do_sample=self.args.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            generated_ids = outputs[0][input_ids.size(1):]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            print_log(f"Generated text: {generated_text}")
            print_log("="*50)

            predictions.append(generated_text)
            references.append(target) 
        
        rouge_results = self.rouge_metric.compute(
            predictions=predictions,
            references=references,
            use_stemmer=False
        )
        
        bertscore_f1 = 0.0  # Placeholder
        
        combined_score = (
            rouge_results['rouge1'] * self.args.rouge1_weight +
            rouge_results['rouge2'] * self.args.rouge2_weight +
            rouge_results['rougeL'] * self.args.rougeL_weight
        )
        
        eval_results = {
            "eval_rouge1": rouge_results['rouge1'],
            "eval_rouge2": rouge_results['rouge2'],
            "eval_rougeL": rouge_results['rougeL'],
            "eval_bertscore_f1": bertscore_f1,
            "eval_combined_score": combined_score,
            "eval_step": state.global_step
        }
        
        if combined_score > self.best_rouge_score:
            self.best_rouge_score = combined_score
            self.best_checkpoint = args.output_dir + f"/checkpoint-{state.global_step}"
            print_log(f"New best checkpoint: {self.best_checkpoint} (Combined Score: {combined_score:.4f})")
        
        self.evaluation_results.append(eval_results)
        
        print_log(f"ROUGE Evaluation - Step {state.global_step}:")
        print_log(f"  ROUGE-1: {rouge_results['rouge1']:.4f}")
        print_log(f"  ROUGE-2: {rouge_results['rouge2']:.4f}")
        print_log(f"  ROUGE-L: {rouge_results['rougeL']:.4f}")
        print_log(f"  Combined Score: {combined_score:.4f}")
        
        model.train()
        
    def run_final_inference(self, model, tokenizer, output_dir):
        print_log("Running final inference with best checkpoint...")
        
        if self.best_checkpoint and os.path.exists(self.best_checkpoint):
            print_log(f"Loading best checkpoint: {self.best_checkpoint}")
            if self.args.use_lora:
                model = PeftModel.from_pretrained(model, self.best_checkpoint)
            else:
                checkpoint_dir = os.path.join(self.best_checkpoint, "adapter_model.bin")
                if os.path.exists(checkpoint_dir):
                    model.load_state_dict(torch.load(checkpoint_dir, map_location=model.device))
                else:
                    print_log(f"Checkpoint directory {checkpoint_dir} does not exist. Loading full model instead.")

        test_dataset = EntailmentDataset(
            self.args.test_path, tokenizer, self.args.max_input_length, self.args.max_output_length,
            self.args.model_type, "test", self.args.use_chat_template, self.fewshot_examples, self.args.num_fewshot
        )

        results = self.generate_candidates_optimized(model, tokenizer, test_dataset)
        
        timestamp = get_timestamp()
        model_id = get_model_id_from_path(self.args.model_name)
        
        output_file = os.path.join(output_dir, f"{model_id}_{timestamp}_final_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        eval_results = self.evaluate_results(results)
        
        eval_file = os.path.join(output_dir, f"{model_id}_{timestamp}_final_evaluation.json")
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=2)
        
        # if self.args.wandb_project:
        #     final_results = {f"final_{k}": v for k, v in eval_results.items()}
        #     wandb.log(final_results)
        
        print_log("Final Evaluation Results:")
        for key, value in eval_results.items():
            if isinstance(value, float):
                print_log(f"  {key}: {value:.4f}")
            else:
                print_log(f"  {key}: {value}")
        
        return results, eval_results
    
    def generate_candidates(self, model, tokenizer, input_ids, attention_mask):

        candidates = []

        # The First candidate(Build the first Cache)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.args.max_new_tokens,
                temperature=self.args.temperature if self.args.do_sample else None,
                top_k=self.args.top_k if self.args.do_sample else None,
                top_p=self.args.top_p if self.args.do_sample else None,
                repetition_penalty=self.args.repetition_penalty,
                do_sample=self.args.do_sample,
                num_beams=self.args.num_beams if not self.args.do_sample else 1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True if self.args.do_sample else False
            )
            
            if hasattr(outputs, 'sequences'):
                generated_ids = outputs.sequences[0][input_ids.size(1):]
            else: 
                generated_ids = outputs[0][input_ids.size(1):]

            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            candidates.append(generated_text)

            # Add Additional Candidates 
            for _ in range(self.args.num_cands - 1):
                temp_variation = self.args.temperature * (0.8 + 0.4 * torch.rand(1).item())

                additional_outputs = model.generate(
                    input_ids =input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.args.max_new_tokens,
                    temperature=temp_variation if self.args.do_sample else None,
                    top_k=self.args.top_k if self.args.do_sample else None,
                    top_p=self.args.top_p if self.args.do_sample else None,
                    repetition_penalty=self.args.repetition_penalty,
                    do_sample=True, # Force to ...
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )

                generated_ids = additional_outputs[0][input_ids.size(1):]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                candidates.append(generated_text)

            return candidates

    def generate_candidates_optimized(self, model, tokenizer, test_dataset):

        print_log("Generating candidates with Cache Optimization...")
        model.eval()
        results = []
        
        # Create data collator for inference
        data_collator = create_data_collator(
            tokenizer, 
            self.args.model_type, 
            self.args.pad_to_multiple_of
        )
        
        dataloader = DataLoader(
            test_dataset, 
            batch_size=1, 
            shuffle=False,
            collate_fn=data_collator
        )
        
        for batch_index, batch_data in enumerate(tqdm(dataloader, desc="Generating")):
            input_ids = batch_data['input_ids'].to(model.device)
            attention_mask = batch_data['attention_mask'].to(model.device)

            def extract_keys(data, key, default_value, batch_index=None):
                if key in data:
                    value = data[key]
                    if isinstance(value, list) and len(value) > 0:
                        return value[0]
                    elif isinstance(value, torch.Tensor):
                        return value.item() if value.numel() == 1 else str(value)
                    else:
                        return str(value) if value is not None else (default_value if batch_index is None else f"{default_value}_{batch_index}")
            
                if 'input' in data:
                    input_data = data['input']
                    if isinstance(input_data, list) and len(input_data) > 0:
                        input_data = input_data[0]

                    if isinstance(input_data, dict) and key in input_data:
                        value = input_data[key]
                        if isinstance(value, list) and len(value) > 0:
                            return value[0]
                        else:
                            return str(value) if value is not None else default_value

            # Extract other information from batch
            example_id = extract_keys(batch_data, 'id', 'example', batch_index)
            premise = extract_keys(batch_data, 'premise', '')
            proposition = extract_keys(batch_data, 'proposition', '')
            label = extract_keys(batch_data, 'label', '')

            print_log(f"Processing Example {batch_index + 1}:")
            print_log(f"[Extracted]  ID: {example_id}")
            print_log(f"[Extracted]  Premise: {premise}")
            print_log(f"[Extracted]  Proposition: {proposition}")
            print_log(f"[Extracted]  Label: {label}")

            candidates = self.generate_candidates(model, tokenizer, input_ids, attention_mask)

            # if len(candidates) > 1 and target:
            #     candidates = self.rank_candidates(candidates, target)

            best_candidate = ""
            
            result = {
                "id": example_id,
                "premise": premise,
                "proposition": proposition,
                "label": label,
                "candidates": candidates,
                "best_candidate": best_candidate # Log the best candidate with the highest PPL score(need to be modified)
            }
            results.append(result)

            # Log first few examples for debugging
            if batch_index < 5:
                print_log(f"Example {batch_index + 1}:")
                print_log(f"  Premise: {premise}")
                print_log(f"  Proposition: {proposition}")
                print_log(f"  Label: {label}")
                for i, candidate in enumerate(candidates[:3]):
                    print_log(f"  Candidate {i + 1}: {candidate}...")
        
        print_log(f"Generated {len(results)} results")
        return results
    
    # # Test Dataset doesn't have Target(GT) -> Cannot use this function only with ROUGE(need to modify)
    # def rank_candidates(self, candidates, target):
    #     if not target or len(candidates) <= 1:
    #         return candidates

    #     scores = []

    #     for candidate in candidates:
    #         rouge_result = self.rouge_metric.compute(
    #             predictions=[candidate],
    #             # references=[target],
    #             use_stemmer=False # Train, Evaluate and Inference are all done with Korean Data
    #         )

    #         score = (
    #             # rouge_result['rouge1'] * 0.5 +
    #             # rouge_result['rouge2'] * 0.3 +
    #             # rouge_result['rougeL'] * 0.2
    #         )

    #         scores.append((score, candidate))

    #     scores.sort(key=lambda x: x[0], reverse=True)
    #     ranked_candidates = [candidate for score, candidate in scores]

    #     return ranked_candidates
            
    # def evaluate_results(self, results):
    #     print_log("Evaluating results...")
        
    #     predictions = [result['candidates'][0] for result in results]
    #     # Evaluate with PPL (Need to Select a Model to Calculate...)

    #     return eval_results

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print_log(f"Seed set to {seed}")

def load_model_and_tokenizer(args):
    tokenizer_name = args.tokenizer_name or args.model_name
    print_log(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=args.cache_dir)

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
    
    if args.use_lora:
        print_log("Applying LoRA configuration...")
        
        if args.lora_target_modules is None:
            if "llama" in args.model_name.lower() or "kanana" in args.model_name.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            elif "qwen" in args.model_name.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            else:
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        else:
            target_modules = args.lora_target_modules
            
        print_log(f"LoRA target modules: {target_modules}")
        
        task_type = TaskType.CAUSAL_LM if args.model_type == "causal" else TaskType.SEQ_2_SEQ_LM
        
        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none"
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print_log("LoRA applied successfully")

    return model, tokenizer

def train_model(args):
    print_log("Starting training mode")
    
    # Prepare few-shot examples
    fewshot_examples = prepare_fewshot_examples(args.train_path, args.fewshot_seed, args.num_fewshot * 3)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)
    
    # Create datasets using the new data module
    train_dataset, eval_dataset, test_dataset = create_datasets(args, tokenizer, fewshot_examples)
    
    # Create dynamic data collator
    data_collator = create_data_collator(tokenizer, args.model_type, args.pad_to_multiple_of)
    
    # Create output directory with timestamp
    timestamp = get_timestamp()
    model_id = get_model_id_from_path(args.model_name)
    run_name = f"{model_id}_{timestamp}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    print_log(f"Output directory: {output_dir}")
    print_log(f"Run name: {run_name}")
    
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
        eval_strategy="steps",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        fp16=args.fp16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        remove_unused_columns=False,
        report_to="wandb",
        dataloader_pin_memory=False,  # For dynamic padding
    )
    
    rouge_callback = CustomCallback(args, eval_dataset, tokenizer, fewshot_examples)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[rouge_callback, EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)],
    )
    
    print_log("Starting training...")
    trainer.train()
    
    print_log("Saving final model...")
    if args.use_lora:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print_log("LoRA weights and tokenizer saved")
    else:
        trainer.save_model()
        print_log("Full model saved")

    if args.mode == "train_and_test":
        print_log("Running automatic inference with best checkpoint...")
        final_results, final_eval = rouge_callback.run_final_inference(model, tokenizer, output_dir)
        
        # Save training summary
        training_summary = {
            "model_name": args.model_name,
            "run_name": run_name,
            "best_checkpoint": rouge_callback.best_checkpoint,
            "best_combined_score": rouge_callback.best_rouge_score,
            "final_evaluation": final_eval,
            "training_args": vars(args),
            "evaluation_history": rouge_callback.evaluation_results
        }
        
        summary_file = os.path.join(output_dir, f"{run_name}_training_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(training_summary, f, ensure_ascii=False, indent=2)
        
        print_log(f"Training summary saved to {summary_file}")
        
        return model, tokenizer, final_results, final_eval
    
    print_log(f"Training complete. Model saved to {output_dir}")
    return model, tokenizer, None, None

def main():
    args = get_parse()
    set_seed(args.seed)
    
    print_log("Starting main execution")
    print_log(f"Mode: {args.mode}")
    print_log(f"Model: {args.model_name}")
    print_log(f"Use chat template: {args.use_chat_template}")
    print_log(f"Dynamic padding: pad_to_multiple_of={args.pad_to_multiple_of}")
    
    # Create timestamped output directory
    model_id = get_model_id_from_path(args.model_name)
    timestamp = get_timestamp()
    run_name = f"{model_id}_{timestamp}"

    # Initialize wandb
    if args.mode in ["train", "train_and_test"]:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=run_name
        )
        print_log("Wandb initialized")
    
    if args.mode == "train":
        model, tokenizer, _, _ = train_model(args)

    elif args.mode == "train_and_test":
        model, tokenizer, final_results, final_eval = train_model(args)
        
        if final_results and final_eval:
            print_log("Training and testing completed successfully!")
            print_log("Final Results Summary:")
            for key, value in final_eval.items():
                if isinstance(value, float):
                    print_log(f"  {key}: {value:.4f}")
                else:
                    print_log(f"  {key}: {value}")

    elif args.mode == "inf" or args.mode == "test":
        # Inference only mode
        print_log("Starting inference mode")
        fewshot_examples = prepare_fewshot_examples(args.train_path, args.fewshot_seed, args.num_fewshot * 3)
        model, tokenizer = load_model_and_tokenizer(args)
        
        if args.lora_model_path:
            print_log(f"Loading LoRA model from {args.lora_model_path}")
            model = PeftModel.from_pretrained(model, args.lora_model_path)
        
        test_dataset = EntailmentDataset(
            args.test_path, tokenizer, args.max_input_length, args.max_output_length,
            args.model_type, "test", args.use_chat_template, fewshot_examples, args.num_fewshot
        )
        
        # Create dummy callback for inference
        callback = CustomCallback(args, test_dataset, tokenizer, fewshot_examples)
        results = callback.generate_candidates(model, tokenizer, test_dataset)
        eval_results = callback.evaluate_results(results)
        
        # Save results
        timestamp = get_timestamp()
        model_id = get_model_id_from_path(args.model_name)
        output_dir = os.path.join(args.output_dir, f"{model_id}_{timestamp}_inference")
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"{model_id}_{timestamp}_inference_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        eval_file = os.path.join(output_dir, f"{model_id}_{timestamp}_inference_evaluation.json")
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=2)
        
        print_log("Inference Results:")
        for key, value in eval_results.items():
            if isinstance(value, float):
                print_log(f"  {key}: {value:.4f}")
            else:
                print_log(f"  {key}: {value}")

if __name__ == "__main__":
    main()
