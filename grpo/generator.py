# ./generator.py

import torch
from tqdm import tqdm
from typing import List, Dict
from model import format_input_prompt

def generate_candidates(model, tokenizer, input_text, num_candidates=5, max_new_tokens=64, temperature=0.7, top_p=0.95, device) -> List[str]:
    model.eval()
    candidates = []

    with torch.no_grad():
        inputs = tokenizer(
            input_text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
        ).to(device)
        # inputs = {k: v.to(model.device) for k, v in inputs.items()}

        for _ in range(num_candidates):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_candidates,
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

    return candidates

def generate_adapter_a_candidates(adapter_a, tokenizer, data_smaples: List[Dict], batch_size=1, num_candidates=5) -> Dict[str, List[str]]:
    all_candidates = {}

    for i in tqdm(range(0, len(data_samples), batch_size), desc="Generating Adapter A Candidates"):
        batch_samples = data_samples[i:i+batch_size]

        for sample in batch:
            input_text = format_input_prompt(sample["premise"], sampele["proposition"], sample["label"])
            candidates = generate_candidates(adapter_a, tokenizer, input_text, num_candidates, device=device)

            key = f"{sample['premise']} ||| {sample['proposition']}"
            all_candidates[key] = candidates

        return all_candidates
