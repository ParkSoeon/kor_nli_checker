# ./generator.py

import torch
from tqdm import tqdm
from typing import List, Dict
from model import format_input_prompt

def print_log(message, prefix="LOG"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def generate_candidates(model, tokenizer, input_text, num_candidates=5, max_new_tokens=90, 
                        temperature=0.7, top_p=0.95, device: str = 'cuda', 
                        use_chat_template=True) -> List[str]:
    model.eval()
    candidates = []

    if use_chat_template:
        messages = [
            {
                "role": "system",
                "content": "당신은 한국어 자연어 추론 전문가입니다."
            },
            {
                "role": "user", 
                "content": input_text
            }
        ]
        
        formatted_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        print_log(f"Chat template applied for generation")
        print_log(f"Formatted input (last 100 chars): {formatted_input[-100:]}")
    else:
        formatted_input = input_text

    print_log(f"==== Generating {num_candidates} candidates for input: \n{input_text} ====")

    with torch.no_grad():
        inputs = tokenizer(
            input_text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
        ).to(device)
        # inputs = {k: v.to(model.device) for k, v in inputs.items()}

        print_log(f"Input tokens (first 20): {inputs['input_ids'][0][:20].tolist()}")
        print_log(f"Decoded with special tokens: {tokenizer.decode(inputs['input_ids'][0][:50], skip_special_tokens=False)}")

        for i in range(num_candidates):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            generated_text = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            candidates.append(generated_text)
            print_log(f"Generated Candidate{i+1}: {generated_text}")

    print_log("="*40)

    return candidates

def generate_adapter_a_candidates(adapter_a, tokenizer, data_samples: List[Dict], 
                                batch_size=1, num_candidates=5, 
                                device: str="cuda") -> Dict[str, List[str]]:
    all_candidates = {}

    print_log(f"Starting candidate generation with Adapter A on {len(data_samples)} samples ===== ")

    for i in tqdm(range(0, len(data_samples), batch_size), desc="Generating Adapter A Candidates"):
        batch_samples = data_samples[i:i+batch_size]

        # for j, sample in batch:
        #     sample_num = i + j + 1
        #     print_log(f"Processing Sample {sample_num}/{len(data_samples)}")
        #     print_log(f"Premise    : {sample['input']['premise']}")
        #     print_log(f"Proposition: {sample['input']['proposition']}")
        #     print_log(f"Label      : {sample['label']}")

        #     input_text = format_input_prompt(sample['input']["premise"], sampele['input']["proposition"], sample['input']["label"])
        #     candidates = generate_candidates(adapter_a, tokenizer, input_text, num_candidates, device=device)

        #     key = f"{sample['input']['premise']} ||| {sample['proposition']}"
        #     all_candidates[key] = candidates

        for sample in batch_samples:
            base_prompt = format_input_prompt(
                sample['input']["premise"], 
                sample['input']["proposition"],
                sample['input']["label"]
            )
            
            candidates = generate_candidates(
                adapter_a, tokenizer, base_prompt, num_candidates, 
                device=device, use_chat_template=use_chat_template
            )

            key = f"{sample['input']['premise']} ||| {sample['input']['proposition']}"
            all_candidates[key] = candidates

        print_log(f"Finished generating candidates with Adapter A ===== ")

        return all_candidates

def generate_adapter_b_candidates(adapter_b, tokenizer, data_samples: List[Dict], batch_size=1, num_candidates=5, device: str = 'cuda') -> Dict[str, List[str]]:
    # return generate_adapterq_a_candidates(adapter_b, tokenizer, data_samples, batch_size, num_candidates, device)

    all_candidates = {}
    print_log(f"Starting candidate generation with Adapter B on {len(data_samples)} samples ===== ")
    for i in tqdm(range(0, len(data_samples), batch_size), desc="Generating Adapter B Candidates"):
        batch_samples = data_samples[i:i+batch_size]

        for j, sample in enumerate(batch_samples):
            sample_num = i + j + 1
            print_log(f"Processing Sample {sample_num}/{len(data_samples)}")
            print_log(f"Premise    : {sample['input']['premise']}")
            print_log(f"Proposition: {sample['input']['proposition']}")
            print_log(f"Label      : {sample['label']}")

            input_text = format_input_prompt(sample['input']["premise"], sample['input']["proposition"], sample['input']["label"])
            candidates = generate_candidates(adapter_b, tokenizer, input_text, num_candidates, device=device)

            key = f"{sample['input']['premise']} ||| {sample['proposition']}"
            all_candidates[key] = candidates
    print_log(f"Finished generating candidates with Adapter B ===== ")
    return all_candidates
