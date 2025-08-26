import torch
from tqdm import tqdm

def generate_candidates(model, tokenizer, input_text, num_candidates=5, 
                       max_new_tokens=128, temperature=0.8, top_p=0.9):
    """
    Generate multiple candidates for single input
    Args:
        model: language model
        tokenizer: tokenizer
        input_text: str, input prompt
        num_candidates: int, number of candidates to generate
        max_new_tokens: int, max tokens to generate
        temperature: float, sampling temperature
        top_p: float, nucleus sampling parameter
    Returns:
        list: generated candidates
    """
    model.eval()
    candidates = []
    
    with torch.no_grad():
        inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=384)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        for _ in range(num_candidates):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Decode only the newly generated part
            generated_text = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            candidates.append(generated_text)
    
    return candidates

def batch_generate_adapter_a_candidates(adapter_a, tokenizer, data_samples, 
                                       batch_size=4, num_candidates=5):
    """
    Generate candidates for Adapter A in batches
    Args:
        adapter_a: Adapter A model
        tokenizer: tokenizer
        data_samples: list of data samples
        batch_size: int, batch size
        num_candidates: int, candidates per sample
    Returns:
        dict: input -> candidates mapping
    """
    all_candidates = {}
    
    for i in tqdm(range(0, len(data_samples), batch_size), desc="Generating Adapter A candidates"):
        batch = data_samples[i:i+batch_size]
        
        for sample in batch:
            input_text = format_input_prompt(sample['premise'], sample['proposition'])
            candidates = generate_candidates(
                adapter_a, tokenizer, input_text, num_candidates
            )
            
            # Use premise+proposition as key for uniqueness
            key = f"{sample['premise']}||{sample['proposition']}"
            all_candidates[key] = candidates
    
    return all_candidates
