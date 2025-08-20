# ./generator.py

import os
import sys
import json
import argparse
import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, Trainer, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling

from evaluate import bertscore, rouge

def get_parse():
    parser = argparse.ArgumentParser(description="Generate text using a pre-trained model.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the pre-trained model.")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Name of the tokenizer.")
    parser.add_argument("--train_path", type=str, required=True, help="Path to the input file containing prompts.")
    parser.add_argument("--dev_path", type=str, required=True, help="Path to the input file containing validation prompts.")
    parser.add_argument("--test_path", type=str, required=True, help="Path to save the generated outputs.")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of generated text.")
    parser.add_argument("--num_cands", type=int, default=3, help="Number of candidates to generate for each prompt.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling parameter.")
    parser.add_argument("--wandb_project", type=str, default="text_generation", help="WandB project name for logging.")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty for text generation.")

    parser.add_argument("--mode", type=str, choices=["train", "inf", "test"], default="train", help="Mode of operation: train or eval.")
    
    return parser.parse_args()

class EntailmentDataset(Dataset):
    def __init__(self, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Add Dev dataset to Train and Split with 8:2 ratio
        with open(self.train_path, 'r') as f:
            self.train_data = f.readlines()
        with open(self.dev_path, 'r') as f:
            self.dev_data = f.readlines()
        self.data = self.train_data + self.dev_data
        np.random.shuffle(self.data)
        split_idx = int(len(self.data) * 0.8)
        self.train_data = self.data[:split_idx]
        self.dev_data = self.data[split_idx:]

        print(f"Loaded {len(self.train_data)} training examples and {len(self.dev_data)} validation examples.")

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        item = self.train_data[idx]

        premise = item['input']['premise']
        proposition = item['input']['proposition']
        label = item['output']['label']
        output = item['output']

        input_text = f"premise: {premise} proposition: {proposition} label: {label}"

        if self.model_type == "causal":
            full_text = f"{input_text}\nExplanation: {output}
            encoding = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding=False, # But if needed, change to Dynamic padding to input
                return_tensors="pt"
            )

            labels = encoding['input_ids'].clone()
            labels[0, :len(input_encoding['input_ids'][0])] = -100


            return {
                "input_ids": encoding['input_ids'].squeeze(),
                "attention_mask": encoding['attention_mask'].squeeze(),
                "labels": labels.squeeze()
            }

        else:
            input_encoding = self.tokenizer(
                input_text,
                truncation=True,
                max_length=self.max_length,
                padding=False, # But if needed, change to Dynamic padding to input
                return_tensors="pt"
            )

            output_encoding = self.tokenizer(
                output,
                truncation=True,
                max_length=self.max_length,
                padding=False, # But if needed, change to Dynamic padding to input
                return_tensors="pt"
            )

            return {
                "input_ids": input_encoding['input_ids'].squeeze(),
                "attention_mask": input_encoding['attention_mask'].squeeze(),
                "labels": output_encoding['input_ids'].squeeze()
            }

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(model_args):
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_args.model_type == "causal":
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            torch_dtype=torch.float16, 
            device_map="auto" 
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    return model, tokenizer

def train_model(model_args, data_args, training_args):
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )

    train_dataset = DataCollator(tokenizer, max_length=data_args.max_input_length)
    dev_dataset = DataCollator(tokenizer, max_length=data_args.max_input_length)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    trainer.train()
    tokenizer.save_pretrained(training_args.output_dir)

    print("Training complete. Model saved to", training_args.output_dir)

    return model, tokenizer

def generate_candidates(model, tokenizer, prompts, generation_args, model_type):


def main():
    args = get_parse()
    wandb.init(project=parse_args.wandb_project, config=parse_args)

    if args.mode == "train":
        wandb.init(
            project=args.wandb_project,
            config={
                "model_name": args.model_name,
                "tokenizer_name": args.tokenizer_name,
                "max_length": args.max_length,
                "num_cands": args.num_cands,
                "batch_size": args.batch_size,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty
            }
        )

    model_args = ModelArguments(
        model_name_or_path=args.model_name,
        tokenizer_name=args.tokenizer_name,
        cache_dir="/home/nlplab/hdd1/cache_dir",
        use_auth_token=False
    )

    data_args = DataArguments(
        train_file = train_ds, # Cause already split in DataCollator class
        validation_file = dev_ds,
        max_input_length = args.max_length,
        max_target_length = args.max_length,
    )

    generation_args = GenerationArguments(
        num_cands=args.num_cands,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty
    )

    if args.mode == "train":
        training_args = TrainingArguments(
            output_dir="./output",
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=3,
            logging_dir="./logs",
            logging_steps=10,
            save_steps=100,
            evaluation_strategy="steps",
            eval_steps=100,
            save_total_limit=2,
            load_best_model_at_end=True,
            report_to_wandb=True,
            remove_unused_columns=False,
            greater_is_better=False,
            load_best_model_at_end=True,
            fp16=True,
        )

        model, tokenizer = train_model(model_args, data_args, training_args)

    else:
        model, tokenizer = load_model_and_tokenizer(model_args)

        with open(args.test_path, 'r') as f:
            prompts = f.readlines()

        results = generate_candidates(
            model, tokenizer, test_data, generation_args, model_args.model_type
        )

        output_file = os.path.join(args.test_path, "generated_candidates.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"Generated candidates saved to {output_file}")

    if args.mode == "train":
        wandb.finish()

if __name__ == "__main__":
    main()

