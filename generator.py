
import torch
import gc
from tqdm import tqdm
from typing import List, Dict
from transformers import AutoTokenizer
from peft import PeftModel
from model import load_model_and_tokenizer
from data import GRPODataset
from torch.utils.data import DataLoader
import os

def clear_memory():
    """메모리 정리"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def load_adapter_for_inference(model_path_or_model, base_model_name=None, tokenizer=None, device="cuda"):
    """어댑터 모델을 인퍼런스용으로 로드 (메모리 효율적)"""
    
    if isinstance(model_path_or_model, str):
        # 경로에서 로드하는 경우
        if base_model_name:
            base_model, tokenizer = load_model_and_tokenizer(base_model_name, device)
        else:
            # 어댑터 경로에서 베이스 모델 정보 추출
            config_path = os.path.join(model_path_or_model, "adapter_config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    adapter_config = json.load(f)
                base_model_name = adapter_config.get('base_model_name_or_path')
                if base_model_name:
                    base_model, tokenizer = load_model_and_tokenizer(base_model_name, device)
                else:
                    raise ValueError("Base model name not found in adapter config")
            else:
                raise ValueError("Adapter config not found")
        
        # PEFT 모델 로드
        model = PeftModel.from_pretrained(base_model, model_path_or_model)
        model.eval()
        
        return model, tokenizer
    else:
        # 이미 로드된 모델인 경우
        model_path_or_model.eval()
        return model_path_or_model, tokenizer

@torch.no_grad()
def generate_candidates_batch(model, tokenizer, prompts: List[str], num_candidates: int = 5, 
                             max_new_tokens: int = 64, temperature: float = 0.7, 
                             top_p: float = 0.95, device="cuda"):
    """배치 단위로 후보 생성 (메모리 효율적)"""
    
    model.eval()
    all_candidates = []
    
    # 토크나이징
    inputs = tokenizer(
        prompts, 
        return_tensors='pt', 
        padding=True, 
        truncation=True, 
        max_length=512
    )
    
    # GPU로 이동
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    batch_candidates = []
    
    for _ in range(num_candidates):
        # 각 후보 생성
        with torch.cuda.amp.autocast():  # 메모리 절약을 위한 mixed precision
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        # 생성된 텍스트 디코딩
        generated_texts = []
        for i, output in enumerate(outputs):
            # 원본 프롬프트 길이만큼 제외
            prompt_length = inputs['input_ids'][i].shape[0]
            generated_part = output[prompt_length:]
            
            generated_text = tokenizer.decode(generated_part, skip_special_tokens=True)
            generated_texts.append(generated_text.strip())
        
        batch_candidates.append(generated_texts)
    
    # 후보들을 샘플별로 재구성
    for i in range(len(prompts)):
        sample_candidates = []
        for j in range(num_candidates):
            sample_candidates.append(batch_candidates[j][i])
        all_candidates.append(sample_candidates)
    
    # 메모리 정리
    del outputs, inputs
    clear_memory()
    
    return all_candidates

def generate_adapter_a_candidates(model_path_or_model, tokenizer, data: List[Dict], 
                                batch_size: int = 4, num_candidates: int = 5, 
                                device="cuda", use_model_path=False, base_model_name=None):
    """Adapter A 후보 생성 (메모리 최적화)"""
    
    print(f"Generating Adapter A candidates with batch_size={batch_size}")
    
    # 모델 로드
    if use_model_path or isinstance(model_path_or_model, str):
        model, tokenizer = load_adapter_for_inference(model_path_or_model, base_model_name, tokenizer, device)
    else:
        model = model_path_or_model
        model.eval()
    
    # 데이터셋 생성
    dataset = GRPODataset(data, tokenizer, use_chat_template=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_candidates = {}
    
    try:
        for batch in tqdm(dataloader, desc="Generating Adapter A candidates"):
            prompts = batch['prompt']
            premises = batch['premise']
            propositions = batch['proposition']
            
            # 배치 생성
            batch_candidates = generate_candidates_batch(
                model, tokenizer, prompts, num_candidates, device=device
            )
            
            # 결과 저장
            for i, (premise, proposition, candidates) in enumerate(zip(premises, propositions, batch_candidates)):
                key = f"{premise} ||| {proposition}"
                all_candidates[key] = candidates
            
            # 배치 처리 후 메모리 정리
            clear_memory()
    
    finally:
        # 모델이 경로에서 로드된 경우 메모리 해제
        if use_model_path or isinstance(model_path_or_model, str):
            del model
            clear_memory()
    
    print(f"Generated candidates for {len(all_candidates)} samples")
    return all_candidates

def generate_adapter_b_candidates(model_path_or_model, tokenizer, data: List[Dict], 
                                batch_size: int = 4, num_candidates: int = 5, 
                                device="cuda", use_model_path=False, base_model_name=None):
    """Adapter B 후보 생성 (메모리 최적화)"""
    
    print(f"Generating Adapter B candidates with batch_size={batch_size}")
    
    # 모델 로드
    if use_model_path or isinstance(model_path_or_model, str):
        model, tokenizer = load_adapter_for_inference(model_path_or_model, base_model_name, tokenizer, device)
    else:
        model = model_path_or_model
        model.eval()
    
    # 데이터셋 생성
    dataset = GRPODataset(data, tokenizer, use_chat_template=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_candidates = {}
    
    try:
        for batch in tqdm(dataloader, desc="Generating Adapter B candidates"):
            prompts = batch['prompt']
            premises = batch['premise']
            propositions = batch['proposition']
            
            # 배치 생성
            batch_candidates = generate_candidates_batch(
                model, tokenizer, prompts, num_candidates, device=device
            )
            
            # 결과 저장
            for i, (premise, proposition, candidates) in enumerate(zip(premises, propositions, batch_candidates)):
                key = f"{premise} ||| {proposition}"
                all_candidates[key] = candidates
            
            # 배치 처리 후 메모리 정리
            clear_memory()
    
    finally:
        # 모델이 경로에서 로드된 경우 메모리 해제
        if use_model_path or isinstance(model_path_or_model, str):
            del model
            clear_memory()
    
    print(f"Generated candidates for {len(all_candidates)} samples")
    return all_candidates

# 스트리밍 생성을 위한 추가 함수 (매우 큰 데이터셋용)
def generate_candidates_streaming(model_path_or_model, tokenizer, data: List[Dict], 
                                output_file: str, batch_size: int = 2, num_candidates: int = 5, 
                                device="cuda", adapter_type="A"):
    """스트리밍 방식으로 후보 생성 (메모리 극도 절약)"""
    
    import json
    
    print(f"Streaming generation for Adapter {adapter_type}")
    
    # 모델 로드
    if isinstance(model_path_or_model, str):
        model, tokenizer = load_adapter_for_inference(model_path_or_model, device=device)
    else:
        model = model_path_or_model
        model.eval()
    
    # 데이터셋 생성
    dataset = GRPODataset(data, tokenizer, use_chat_template=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 파일 스트리밍 쓰기
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('{\n')
        first_item = True
        
        try:
            for batch in tqdm(dataloader, desc=f"Streaming Adapter {adapter_type} candidates"):
                prompts = batch['prompt']
                premises = batch['premise']
                propositions = batch['proposition']
                
                # 배치 생성
                batch_candidates = generate_candidates_batch(
                    model, tokenizer, prompts, num_candidates, device=device
                )
                
                # 결과를 파일에 직접 쓰기
                for i, (premise, proposition, candidates) in enumerate(zip(premises, propositions, batch_candidates)):
                    key = f"{premise} ||| {proposition}"
                    
                    if not first_item:
                        f.write(',\n')
                    else:
                        first_item = False
                    
                    f.write(f'  {json.dumps(key, ensure_ascii=False)}: {json.dumps(candidates, ensure_ascii=False)}')
                
                # 배치 처리 후 메모리 정리
                clear_memory()
        
        finally:
            f.write('\n}')
            
            # 모델 메모리 해제
            if isinstance(model_path_or_model, str):
                del model
                clear_memory()
    
    print(f"Streaming generation completed. Results saved to {output_file}")
