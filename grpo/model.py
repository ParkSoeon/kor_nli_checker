# ./model.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from  peft import LoraConfig, get_peft_model, TaskType
import copy

def load_model_and_tokenizer(model_name, device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map='auto' if torch.cuda.is_available() else None
    )
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

def create_lora_config(r=8, alpha=16, dropout=0.1):
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"],
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    return config

def create_dual_adapters(base_model, lora_config):
    adapter_a = get_peft_model(copy.deepcopy(base_model), lora_config)
    adapter_b = get_peft_model(copy.deepcopy(base_model), lora_config)

    # # Initialize adapter_b with the same weights as adapter_a
    # adapter_b.load_state_dict(adapter_a.state_dict())
    
    return adapter_a, adapter_b

def format_input_prompt(premise, proposition, label):
    prompt = """다음 전제와 가설의 관계를 바탕으로 아래의 형식에 따른 '함의 분석 설명문'을 생성하세요.
    
    [전제]: {premise}
    [가설]: {proposition}
    [관계]: {label}
    
    [함의 분석 설명문]: """
    
    return prompt
