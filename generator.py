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
    DataCollatorForLanguageModeling,
    TrainerCallback
)

from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType,
    PeftModel 
)

from evaluate import load

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
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")

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
    
    # Few-shot arguments
    parser.add_argument("--num_fewshot", type=int, default=3, help="Number of few-shot examples.")
    parser.add_argument("--fewshot_seed", type=int, default=42, help="Seed for few-shot example selection.")
    
    # General arguments
    parser.add_argument("--mode", type=str, choices=["train", "inf", "test", "train_and_test"], default="train_and_test", help="Mode of operation.")
    parser.add_argument("--wandb_project", type=str, default="2025HCLT", help="WandB project name.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--use_chat_template", action="store_true", default=True, help="Use chat template for formatting.")
    
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
        # self.bertscore_metric = load("bertscore")
        self.evaluation_results = []

        self.best_rouge_score = 0.0
        self.best_checkpoint = None
        self.eval_results = []

    def on_evaluate(self, args, state, control, **kwargs): #model, tokenizer, eval_dataloader, **kwargs):
        print_log("Starting evaluation...")

        model = kwargs.get('model')

        model.eval()
        predictions = []
        references = []

        eval_dataset_for_eval = EntailmentDataset(
            data_path=self.args.dev_path,
            tokenizer=self.tokenizer,
            max_input_length=self.args.max_input_length,
            max_output_length=self.args.max_output_length,
            model_type=self.args.model_type,
            mode="test",
            use_chat_template=self.args.use_chat_template,
            fewshot_examples=self.fewshot_examples,
            num_fewshot=self.args.num_fewshot
        )

        # eval_subset = torch.utils.data.Subset(self.eval_dataset)
        # eval_dataloader = DataLoader(self.eval_dataset, batch_size=1, shuffle=False)#self.args.per_device_eval_batch_size, shuffle=False)

        eval_subset = torch.utils.data.Subset(self.eval_dataset, range(min(10, len(self.eval_dataset))))
        eval_dataloader = DataLoader(eval_subset, batch_size=1, shuffle=False)#self.args.per_device_eval_batch_size, shuffle=False)

        # eos_ids = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []

        for batch_idx, batch_data in enumerate(eval_dataloader):

            item = batch_data[0] if isinstance(batch_data, list) else batch_data

            input_ids = batch_data['input_ids'].to(model.device) #.unsqueeze(0).to(model.device)
            attention_mask = batch_data['attention_mask'].to(model.device) #.unsqueeze(0).to(model.device)
            target = item.get('target', '') 
            
            input_text = self.tokenizer.decode(input_ids.squeeze(), skip_special_tokens=False)
            print_log(f"Input text: {input_text}") 
            print_log(f"Target: {target}")

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.args.max_new_tokens,
                    temperature=0.7,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            generated_ids = outputs[0][input_ids.size(1):]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            print_log(f"Generated text: {generated_text}")
            print_log(f"Generated IDs: {generated_ids.tolist()}")
            print_log("="*50)

            predictions.append(generated_text)
            references.append(target) 
        
        rouge_results = self.rouge_metric.compute(
            predictions=predictions,
            references=references,
            use_stemmer=False
        )
        
        # bertscore_results = self.bertscore_metric.compute(
        #     predictions=predictions,
        #     references=references,
        #     model_type="klue/bert-base",
        #     lang="ko"
        # )
        bertscore_f1 = 0.0#np.mean(bertscore_results['f1'])
        
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
        
        if self.args.wandb_project:
            wandb.log(eval_results)
        
        if combined_score > self.best_rouge_score:
            self.best_rouge_score = combined_score
            self.best_checkpoint = args.output_dir + f"/checkpoint-{state.global_step}"
            print_log(f"New best checkpoint: {self.best_checkpoint} (Combined Score: {combined_score:.4f})")
        
        self.evaluation_results.append(eval_results)
        
        print_log(f"ROUGE Evaluation - Step {state.global_step}:")
        print_log(f"  ROUGE-1: {rouge_results['rouge1']:.4f}")
        print_log(f"  ROUGE-2: {rouge_results['rouge2']:.4f}")
        print_log(f"  ROUGE-L: {rouge_results['rougeL']:.4f}")
        # print_log(f"  BERTScore F1: {bertscore_f1:.4f}")
        print_log(f"  Combined Score: {combined_score:.4f}")
        
        model.train()
        
    def run_final_inference(self, model, tokenizer, output_dir):
        print_log("Running final inference with best checkpoint...")
        
        if self.best_checkpoint and os.path.exists(self.best_checkpoint):
            print_log(f"Loading best checkpoint: {self.best_checkpoint}")
            if self.args.use_lora:
                model = PeftModel.from_pretrained(model, self.best_checkpoint)
            else:
                model.load_state_dict(torch.load(os.path.join(self.best_checkpoint, "pytorch_model.bin")))
        
        test_dataset = EntailmentDataset(
            self.args.test_path, tokenizer, self.args.max_input_length, self.args.max_output_length,
            self.args.model_type, "test", self.args.use_chat_template, self.fewshot_examples, self.args.num_fewshot
        )

        results = self.generate_candidates(model, tokenizer, test_dataset)
        
        timestamp = get_timestamp()
        model_id = get_model_id_from_path(self.args.model_name)
        
        output_file = os.path.join(output_dir, f"{model_id}_{timestamp}_final_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        eval_results = self.evaluate_results(results)
        
        eval_file = os.path.join(output_dir, f"{model_id}_{timestamp}_final_evaluation.json")
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=2)
        
        if self.args.wandb_project:
            final_results = {f"final_{k}": v for k, v in eval_results.items()}
            wandb.log(final_results)
        
        print_log("Final Evaluation Results:")
        for key, value in eval_results.items():
            if isinstance(value, float):
                print_log(f"  {key}: {value:.4f}")
            else:
                print_log(f"  {key}: {value}")
        
        return results, eval_results
    
    def generate_candidates(self, model, tokenizer, test_dataset):
        print_log("Generating candidates...")
        model.eval()
        results = []
        
        dataloader = DataLoader(
            test_dataset, 
            batch_size=1, 
            shuffle=False
        )
        
        for batch_idx, item in enumerate(tqdm(dataloader, desc="Generating")):
            if isinstance(item, dict):
                input_ids = item['input_ids'].unsqueeze(0).to(model.device)
                attention_mask = item['attention_mask'].unsqueeze(0).to(model.device)
                target = item['target']
                example_id = item['id']
                prompt = item.get('prompt', '')
                premise = item.get('premise', '')
                proposition = item.get('proposition', '')
                label = item.get('label', '')
            else:
                item = item[0] if isinstance(item, list) else item
                input_ids = item['input_ids'].unsqueeze(0).to(model.device)
                attention_mask = item['attention_mask'].unsqueeze(0).to(model.device)
                target = item['target']
                example_id = item['id']
                prompt = item.get('prompt', '')
                premise = item.get('premise', '')
                proposition = item.get('proposition', '')
                label = item.get('label', '')
            
            candidates = []
            with torch.no_grad():
                for _ in range(self.args.num_cands):
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.args.max_new_tokens,
                        temperature=self.args.temperature if self.args.do_sample else None,
                        top_k=self.args.top_k if self.args.do_sample else None,
                        top_p=self.args.top_p if self.args.do_sample else None,
                        repetition_penalty=self.args.repetition_penalty,
                        do_sample=self.args.do_sample,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    
                    generated_ids = outputs[0][input_ids.size(1):]
                    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                    candidates.append(generated_text)
            
            result = {
                "id": example_id,
                "premise": premise,
                "proposition": proposition,
                "label": label,
                "prompt": prompt,
                "target": target,
                "candidates": candidates,
                "best_candidate": candidates[0] if candidates else ""
            }
            results.append(result)

            # Log first few examples for debugging
            if batch_idx < 5:
                print_log(f"Example {batch_idx + 1}:")
                print_log(f"  Premise: {premise}...")
                print_log(f"  Proposition: {proposition}...")
                print_log(f"  Label: {label}")
                print_log(f"  Target: {target}...")
                print_log(f"  Generated: {candidates[0]}...")
        
        print_log(f"Generated {len(results)} results")
        return results
    
    def evaluate_results(self, results):
        print_log("Evaluating results...")
        
        targets = [result['target'] for result in results]
        predictions = [result['best_candidate'] for result in results]
        
        rouge_results = self.rouge_metric.compute(
            predictions=predictions,
            references=targets,
        )
        
        # bertscore_results = self.bertscore_metric.compute(
        #     predictions=predictions,
        #     references=targets,
        #     model_type="klue/bert-base",
        #     lang="ko"
        # )
        # bertscore_metrics = {
        #     "bertscore_precision": np.mean(bertscore_results['precision']),
        #     "bertscore_recall": np.mean(bertscore_results['recall']),
        #     "bertscore_f1": np.mean(bertscore_results['f1'])
        # }
        # bertscore_f1=0.0
        
        combined_score = (
            rouge_results['rouge1'] * self.args.rouge1_weight +
            rouge_results['rouge2'] * self.args.rouge2_weight +
            rouge_results['rougeL'] * self.args.rougeL_weight
        )
        
        eval_results = {
            "rouge1": rouge_results['rouge1'],
            "rouge2": rouge_results['rouge2'],
            "rougeL": rouge_results['rougeL'],
            "combined_rouge_score": combined_score,
            # **bertscore_metrics,
            "num_examples": len(results),
            "num_candidates_per_example": self.args.num_cands
        }
        
        return eval_results

class EntailmentDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_input_length=512, max_output_length=256, 
                 model_type="causal", mode="train", use_chat_template=True, 
                 fewshot_examples=None, num_fewshot=1):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.model_type = model_type
        self.mode = mode
        self.use_chat_template = use_chat_template
        self.fewshot_examples = fewshot_examples if fewshot_examples else []
        self.num_fewshot = min(num_fewshot, len(self.fewshot_examples))

        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        print_log(f"Loaded {len(self.data)} examples from {data_path}")
        print_log(f"Using {self.num_fewshot} few-shot examples")
        print_log(f"Chat template: {self.use_chat_template}")

    def create_fewshot_prompt(self, premise, proposition, label=None):
        examples = []

        # Add few-shot examples
        for example in self.fewshot_examples[:self.num_fewshot]:
            ex_premise = example['input']['premise']
            ex_proposition = example['input']['proposition']
            ex_label = example['input']['label']  
            ex_output = example['output']
            examples.append({
                "premise": ex_premise,
                "proposition": ex_proposition,
                "label": ex_label,
                "explanation": ex_output
            })

        if self.use_chat_template:
            messages = [{"role": "system", "content": "당신은 한국어 자연어 추론(NLI) 전문가입니다. 주어진 전제와 가설을 분석하여 함의 관계를 설명해주세요."}]

            if examples:
                example_content = "다음은 자연어 추론 과제의 예시입니다:\n\n"
                for i, ex in enumerate(examples, 1):
                    example_content += f"[예시 {i}]\n"
                    example_content += f"[전제] {ex['premise']}\n" 
                    example_content += f"[가설] {ex['proposition']}\n"
                    example_content += f"[관계] {ex['label']}\n"
                    example_content += f"[설명] {ex['explanation']}\n\n"

                example_content += "이제 새로운 전제와 가설에 대해 관계를 분석하여 설명문을 설명하세요.\n\n"
                messages.append({"role": "user", "content": example_content})
                # messages.append({"role": "assistant", "content": ""}) # 여기에 추가적인 말을 넣는게 좋을까.. 조금 더 고민해볼 필요가 있다 생각

            current_message = f"[전제] {premise}\n[가설] {proposition}\n[관계] {label}"
            messages.append({"role": "user", "content": current_message})

            return messages
        else:
            prompt_parts = ["다음은 자연어 추론 작업입니다. 전제와 가설 사이의 관계를 분석하고 설명해주세요.\n"]
            
            for ex in examples:
                prompt_parts.append(f"[전제] {ex['premise']}")
                prompt_parts.append(f"[가설] {ex['proposition']}")
                prompt_parts.append(f"[관계] {ex['label']}")
                prompt_parts.append(f"[설명] {ex['explanation']}\n")
            
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
                full_message = messages+[{"role": "assistant", "content": output}]
                
                # # apply_chat_template 적용
                # full_text = self.tokenizer.apply_chat_template(
                #     messages, 
                #     tokenize=False,
                # )
            
                # apply_chat_template 적용
                full_text_list = self.tokenizer.apply_chat_template(
                    full_message,
                    tokenize=False,
                    add_generation_prompt=False
                )

                if full_text_list.startswith("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\n<|eot_id|><|start_header_id|>system<|end_header_id|>"):
                    full_text_list = full_text_list.replace("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\n<|eot_id|>", "<|begin_of_text|>", 1)
            
                full_text = self.tokenizer(
                    full_text_list, 
                    # tokenize=True,
                    return_tensors="pt",
                    # return_tensors=None, # Not convert to tensors yet -> since we need Attention mask too
                    truncation=True,
                    max_length=self.max_input_length + self.max_output_length,
                    add_special_tokens=False
                )

                prefix_list = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )

                prefix = self.tokenizer(
                    prefix_list,
                    # tokenize=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_input_length,
                    add_special_tokens=False
                )

                # print(f"Shape of full_text['input_ids']: {full_text['input_ids'].shape}")
                
                # print(full_text.shape()) # (1, L)
                # print(f"Full text shape: {full_text[0].shape}")
                # assert 0

                # print(f"Input IDs shape: {input_ids.shape}")
                # print(f"Input IDs: {input_ids.tolist()}")
                # print(f"Full text shape: {full_text.shape}")
                # assert 0
                # # OK

                input_ids = full_text["input_ids"].squeeze(0)  # (1, L) -> (L,)
                attention_mask = full_text["attention_mask"].squeeze(0)
                prefix_input_ids = prefix["input_ids"].squeeze(0)  # (1, L) -> (L,)

                decoded = self.tokenizer.decode(input_ids, skip_special_tokens=False)
                prefix_decoded = self.tokenizer.decode(prefix_input_ids, skip_special_tokens=False)

                print_log(f"==== Example {idx} ====")
                print_log(f"Premise: {premise}")
                print_log(f"Proposition: {proposition}")
                print_log(f"Label: {label}")
                print_log(f"Target Output: {output}")
                
                print_log(f"Messages count: {len(messages)}")
                print_log(f"Full text length of 'chat_template' format: {len(full_message)}")
                print_log(f"Full text with 'chat_template' format:\n{messages}")
                print_log(f"Full text with Decoded format:\n{decoded}")
                print_log(f"Input IDs: {input_ids.tolist()}")

                labels = input_ids.clone()
                print(f"Shape of Labels: {labels.shape}")
                prefix_len = len(prefix_input_ids)
                print(f"Shape of Prefix Length: {prefix_len}")

                # Mask input tokens
                labels[:prefix_len] = -100 

                # assert (input_ids[:prefix_len] == prefix["input_ids"][0]).all(), "Prefix mis-match"
                num_masked_tokens = (labels == -100).sum().item()

                # print_log(f"Input-only text length: {len(prefix)}")
                # print_log(f"Input-only preview:\n{prefix}...")
                print_log(f"Input tokens length: {prefix_len}")
                print_log(f"Total tokens length: {len(full_text['input_ids'][0])}")
                print_log(f"Target tokens length: {len(full_text['input_ids'][0]) - prefix_len}")
                print_log(f"Input IDs (first 10): {full_text['input_ids'].tolist()}")
                print_log(f"Masked Labels (first 10): {labels.tolist()}")
                print_log(f"Masked Labels (length): {num_masked_tokens}")
                print_log("="*80)

                return {
                    "input_ids": input_ids.squeeze(),
                    "attention_mask": full_text["attention_mask"].squeeze(),
                    "labels": labels.squeeze(),
                }

            else:  # inference/test mode
                inf_ft_list = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )

                if inf_ft_list.startswith("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\n<|eot_id|><|start_header_id|>system<|end_header_id|>"):
                    inf_ft_list = inf_ft_list.replace("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\n<|eot_id|>", "<|begin_of_text|>", 1)
                
                inf_full_text = self.tokenizer(
                    inf_ft_list,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_input_length + self.max_output_length,
                    add_special_tokens=False
                )

                # Log first few examples for debugging
                if idx < 3:
                    print_log(f"Inference example {idx}:")
                    print_log(f"Input prompt:\n{input_text}...")

                return {
                    "input_ids": input_encoding["input_ids"].squeeze(),
                    "attention_mask": input_encoding["attention_mask"].squeeze(),
                    "target": output,
                    "id": item.get('id', idx),
                    "prompt": input_text,
                    "premise": premise,
                    "proposition": proposition,
                    "label": label
                }
        else:
            print("[ERROR] Not Set as Chat Template Mode. Please check the configuration.")
            assert 0
                
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

def prepare_fewshot_examples(train_path, seed, num_examples=3):
    print_log(f"Preparing few-shot examples from {train_path}")
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
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
        print_log(f"  Premise: {first_ex['input']['premise']}")
        print_log(f"  Proposition: {first_ex['input']['proposition']}")
        print_log(f"  Label: {first_ex['input']['label']}")
        print_log(f"  Output: {first_ex['output']}")
    
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

    test_dataset = EntailmentDataset(
        args.test_path, tokenizer, args.max_input_length, args.max_output_length,
        args.model_type, "test", args.use_chat_template, fewshot_examples, args.num_fewshot
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
        # save_on_best_metric=False,  # Ensure the best model is saved based on eval_combined_score
        # metric_for_best_model="eval_combined_score",
        fp16=args.fp16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        remove_unused_columns=False,
        report_to="wandb" if args.wandb_project else "none",
        # greater_is_better=True,
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
        callbacks=[rouge_callback],
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
    
    # Create timestamped output directory
    model_id = get_model_id_from_path(args.model_name)
    timestamp = get_timestamp()
    run_name = f"{model_id}_{timestamp}"

    # Initialize wandb
    if args.wandb_project and args.mode == "train":
        wandb.init(
            project=args.wandb_project,
            config=vars(args)
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

if __name__ == "__main__":
    main()
