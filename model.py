# ./model.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from  peft import LoraConfig, get_peft_model, TaskType
import copy
import os
from datetime import datetime

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
    prompt = f"""당신은 한국어 자연어 추론(NLI) 전문가입니다. 주어진 전제와 가설을 분석하여 주어진 관계에 맞는 함의 분석 설명문을 생성하세요.

**중요한 규칙:**
1. 반드시 '[설명] '으로 시작해서 설명문 생성을 시작하세요.
2. 설명은 한 문장 이상, 세 문장 이하로 작성하고, 마지막에 전제와 가설의 관계가 함의 또는 모순임을 명확히 드러내야 합니다.
   - 예: '함의이다.', '함의에 해당된다.', '모순이다.', '모순에 속한다.' 등
3. 전제와 가설의 관계는 무조건 '함의', '모순' 중 하나입니다. '중립'이나, '특정 관계에 해당되지 않는다.' 등의 표현은 허용되지 않습니다.
4. 설명문은 최대 길이 75토큰을 넘지 않도록 최대한 간결하고 명확하게 작성하세요.
5. 설명문은 한국어로 작성되어야 합니다.

위의 규칙을 엄격히 준수하여 답변해 주세요.
    
    [전제]: {premise}
    [가설]: {proposition}
    [관계]: {label}
    
    [함의 분석 설명문]: """
    
    return prompt
