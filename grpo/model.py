# ./model.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from  peft import LoraConfig, get_peft_model, TaskType
import copy

def load_model_and_tokenizer(self, model_name, device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

def create_lora_config(self, r=8, alpha=16, dropout=0.1):
    config = LoraConfig(
        r=self.r,
        lora_alpha=self.alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"],
        lora_dropout=self.dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    return config

def create_dual_adapters(self, base_model, lora_config):
    adapter_a = get_peft_model(copy.deepcopy(base_model), lora_config)
    adapter_b = get_peft_model(copy.deepcopy(base_model), lora_config)

    # Initialize adapter_b with the same weights as adapter_a
    adapter_b.load_state_dict(adapter_a.state_dict())
    
    return adapter_a, adapter_b

def format_input_prompt(self, premise, proposition, label):
    prompt = """다음 전제와 가설의 관계를 바탕으로 '함의 분석 설명문'을 생성하세요.
    
    [전제] {premise}
    [가설] {proposition}
    [관계] {label}
    
    [함의 분석 설명문]"""
    
    return prompt
