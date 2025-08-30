# ./model.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from  peft import LoraConfig, get_peft_model, TaskType
import copy
import os

def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def print_log(message: str, prefix: str ="LOG") -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def load_model_and_tokenizer(model_name, device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map='auto' if torch.cuda.is_available() else None
    )
    
    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #     model.resize_token_embeddings(len(tokenizer))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print_log(f"[DBG] Set pad_token to eos_token: {tokenizer.pad_token}")
    elif tokenizer.pad_token == '<|end_of_text|>':
        tokenizer.pad_token = tokenizer.eos_token
        print_log(f"[DBG] Changed pad_token from <|end_of_text|> to eos_token: {tokenizer.pad_token}")
    
    return model, tokenizer

def load_adapter_model(base_model_name: str, adapter_path: str, device="cuda"):
    base_model, tokenizer = load_model_and_tokenizer(base_model_name, device)

    if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        adapter_model = PeftModel.from_pretrained(base_model, adapter_path)
        print_log(f"Loaded adapter model from {adapter_path}")
    else:
        adapter_weights_path = os.path.join(adapter_path, "adapter_weights.pth")
        if os.path.exists(adapter_weights_path):
            lora_config = create_lora_config()
            adapter_model = get_peft_model(base_model, lora_config)
            adapter_model.load_state_dict(torch.load(adapter_weights_path, map_location=device))
            print_log(f"Loaded adapter weights from {adapter_weights_path}")
        else:
            raise FileNotFoundError(f"No adapter found at {adapter_path}")

    return adapter_model, tokenizer

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
    
    return adapter_a, adapter_b

def save_adapter_safely(adapter_model, save_path: str, model_name: str = "adapter"):
    os.makedirs(save_path, exist_ok=True)

    timestamp = get_timestamp()

    adapter_model.save_pretrained(save_path)
    print_log(f"Adapter model saved to {save_path}")

    return True

def format_input_prompt(premise, proposition, label):
    prompt = f"""다음 전제와 가설의 관계를 바탕으로 아래의 형식에 따른 '함의 분석 설명문'을 생성하세요.
    
    [전제]: {premise}
    [가설]: {proposition}
    [관계]: {label}
    
    [함의 분석 설명문]: """
    
    return prompt
