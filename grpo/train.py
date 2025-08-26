# train.py ...... I don't like the name of this file....

from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, TrainingArguments
import torch
from data import load_data, GRPODataset
from typing import Callable, Dict, List

def creat_grpo_trainer(
    model, tokenizer, dataset, reward_function: Callable,
    output_dir: str, learning_rate: float = 5e-5, batch_size: int = 8, epochs: int = 3, **kwargs
):
    training_args = TrainingArguments(
        output_dir = output_dir,
        learning_rate = learning_rate,
        per_device_train_batch_size = batch_size,
        num_train_epochs = epochs,
        logging_steps = 10,
        save_steps = 100,
        save_total_limit = 2,
        remove_unused_columns = False,
        fp16 = torch.cuda.is_available(),
        report_to = "wandb",
        **kwargs
    )

    grpo_config = GRPOConfig(
        learning_rate = learning_rate,
        batch_size = batch_size,
    )

    grpo_trainer = GRPOTrainer(
        model = model,
        args = training_args,
        train_dataset = dataset,
        tokenizer = tokenizer,
        reward_function = reward_function,
        config = grpo_config,
    )

    return grpo_trainer

def train_adapter_a(self, adapter_a, tokenizer, train_data: List[Dict], val_data: List[Dict], output_dir: str, args) -> torch.nn.Module:
    
    train_dataset = GRPODataset(train_data, tokenizer)

    # Define a Reward Function based on ROUGE(for Adapter A)
    def adapter_a_reward_function(self, generated: str, references: str, query: str) -> float:
        return compute_adapter_a_reward(genertated, references, lambda1=self.lambda1, lambda2=self.lambda2, lambda3=self.lambda3)

    trainer = creat_grpo_trainer(
        model = adapter_a,
        tokenizer = tokenizer,
        dataset = train_dataset,
        reward_function = adapter_a_reward_function,
        output_dir=f"{output_dir}/adapter_a",
        learning_rate = args.learning_rate,
        batch_size = args.batch_size,
        epochs = args.epochs,
    )

    trainer.train()

    return adapter_a

def train_adapter_b(self, adapter_b, tokenizer, train_data: List[Dict], val_data: List[Dict], adapter_a_candidates: Dict[str, List[str]], output_dir: str, args) -> torch.nn.Module:
    
    train_dataset = GRPODataset(train_data, tokenizer)

    # Define a Reward Function based on Interactive BLEU, ROUGE-L, and PPL(for Adapter B)
    def adapter_b_reward_function(self, generated: str, references: str, query: str) -> float:
        key = None
        for k in adapter_a_candidates.keys():
            premise, proposition = k.split(" ||| ")
            if premise in query and proposition in query:
                key = k
                break

        if key is None:
            return 0.0

        a_candidates = adapter_a_candidates[key]

        return compute_adapter_b_reward(
            generated, references, a_candidates, tokenizer=tokenizer, model=ppl_model, lambda1=self.lambda1, lambda2=self.lambda2, lambda3=self.lambda3
        )

    # Create Trainer
    trainer = creat_grpo_trainer(
        model = adapter_b,
        tokenizer = tokenizer,
        dataset = train_dataset,
        reward_function = adapter_b_reward_function,
        output_dir=f"{output_dir}/adapter_b",
        learning_rate = args.learning_rate,
        batch_size = args.batch_size,
        epochs = args.epochs,
    )

    trainer.train()

    return adapter_b
