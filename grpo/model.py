import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import copy

def load_base_model(model_name, device='cuda'):
    """
    Load base SFT model and tokenizer
    Args:
        model_name: str, HuggingFace model name
        device: str, device to load model on
    Returns:
        tuple: (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    
    return model, tokenizer

def create_lora_config(r=16, lora_alpha=32, lora_dropout=0.1):
    """
    Create LoRA configuration
    Args:
        r: int, rank
        lora_alpha: int, scaling factor
        lora_dropout: float, dropout rate
    Returns:
        LoraConfig
    """
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

def create_dual_adapters(base_model, lora_config):
    """
    Create two identical adapters from base model
    Args:
        base_model: base language model
        lora_config: LoRA configuration
    Returns:
        tuple: (adapter_a, adapter_b)
    """
    # Create first adapter
    adapter_a = get_peft_model(copy.deepcopy(base_model), lora_config)
    
    # Create second adapter with same initialization
    adapter_b = get_peft_model(copy.deepcopy(base_model), lora_config)
    
    # Copy weights to ensure identical initialization
    adapter_b.load_state_dict(adapter_a.state_dict())
    
    return adapter_a, adapter_b

def format_input_prompt(premise, proposition):
    """
    Format input for NLI task
    Args:
        premise: str
        proposition: str
    Returns:
        str: formatted prompt
    """
    prompt = f"""다음 전제와 가설의 관계를 분석하고 설명하세요.

전제: {premise}
가설: {proposition}

분석:"""
    
    return prompt
