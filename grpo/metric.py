# ./metrics.py

import numpy as np
from evaluate import load
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
from typing import List, Dict
from datetime import datetime

_rouge_scorer = None

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def print_log(message, prefix="LOG"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def get_rouge_scorer():
    global _rouge_scorer
    if _rouge_scorer is None:
        _rouge_scorer = load("rouge")
    return _rouge_scorer

def compute_interactive_bleu(sentence_a: List[str], sentence_b: List[str]) -> float:
    smoothing_function = SmoothingFunction().method1

    total_bleu = 0.0
    pair_count = 0

    for a in sentence_a:
        for b in sentence_b:
            tokens_a = a.split()
            tokens_b = b.split()

            bleu = sentence_bleu([tokens_a], tokens_b, smoothing_function=smoothing_function)
            
            total_bleu += bleu
            pair_count += 1

    return total_bleu / pair_count if pair_count > 0 else 0.0

def compute_rouge(generated: str, references: str, rouge_types: List[str] = ["rouge1", "rouge2", "rougeL"]) -> Dict[str, float]:

    scorer = load("rouge")
    scores = scorer.compute(predictions=[generated], references=[references], rouge_types=rouge_types)

    return {
        "rouge1": scores["rouge1"],
        "rouge2": scores["rouge2"],
        "rougeL": scores["rougeL"],
        # "combined": lambda1 * scores["rouge1"]["f1"] + lambda2 * scores["rouge2"]["f1"] + lambda3 * scores["rougeL"]["f1"]
        # "combined": lambda1 * scores["rouge1"] + lambda2 * scores["rouge2"] + lambda3 * scores["rougeL"]
    }

def compute_perplexity(model, tokenizer, text: str, device="cuda") -> float:
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss

        perplexity = torch.exp(loss)

    return perplexity.item()

def compute_adapter_a_reward(generated: str, references: str, lambda1: float = 0.0, lambda2: float = 0.0, lambda3: float = 0.0) -> float:
    rouge_scores = compute_rouge(generated, references)

    reward_a = (
        lambda1 * rouge_scores["rouge1"] + 
        lambda2 * rouge_scores["rouge2"] + 
        lambda3 * rouge_scores["rougeL"]
    )

    print_log(f"=== Adapter A Reward ===")
    print_log(f"    Generated: {generated}")
    print_log(f"    Reference: {references}")
    print_log(f"    ROUGE-1  : {rouge_scores['rouge1']:.4f}")
    print_log(f"    ROUGE-2  : {rouge_scores['rouge2']:.4f}")
    print_log(f"    ROUGE-L  : {rouge_scores['rougeL']:.4f}")
    print_log(f"    Final Reward: {reward_a:.4f}")
        
    return reward_a

# def compute_adapter_b_reward(generated: str, references: str, adapter_a_cands: List[str], model=None, tokenizer=None, lambda1: float = 0.0, lambda2: float = 0.0, lambda3: float = 0.0) -> float:

#     interactive_bleu = compute_interactive_bleu([generated], adapter_a_cands)

#     rouge_scores = compute_rouge(generated, references)
#     rouge_l = rouge_scores["rougeL"]

#     ppl_penalty = 0.0
#     if model is not None and tokenizer is not None:
#         ppl = compute_perplexity(model, tokenizer, generated)
#         ppl_penalty = -np.log(ppl) # Lower Perplexity == Higher Reward

#     reward_b = (
#         -lambda1 * interactive_bleu + 
#         lambda2 * rouge_l +
#         -lambda3 * ppl_penalty
#     )

#     print_log(f"=== Adapter B Reward ===")
#     print_log(f"    Generated: {generated}")
#     print_log(f"    Reference: {references}")
#     print_log(f"    Interactive BLEU: {interactive_bleu:.4f}")
#     print_log(f"    ROUGE-L       : {rouge_l:.4f}")
#     print_log(f"    PPL Penalty   : {ppl_penalty:.4f}")
#     print_log(f"    Final Reward : {reward_b:.4f}")
        
#     return reward_b

def compute_adapter_b_reward(generated: str, references: str, adapter_a_cands: List[str], model=None, tokenizer=None, lambda1: float = 0.0, lambda2: float = 0.0, lambda3: float = 0.0) -> float:

    # 1. For GOLD (Quality) - ROUGE-L with Gold
    rouge_scores = compute_rouge(generated, references)
    gold_similarity = rouge_scores["rougeL"]
    
    # 2. Coparing with A (Balance for Diversity vs Consistency)
    # Diversity: Interactive BLEU (More Different -> Higher Score)
    interactive_bleu = compute_interactive_bleu([generated], adapter_a_cands)
    diversity_score = -interactive_bleu  # Negative Score -> More different, Higher Score
    
    # Consistency: Avg ROUGE-L with Candidates from A(More Similar -> Higher Score)
    a_rouge_scores = []
    for cand in adapter_a_cands:
        rouge_score = compute_rouge(generated, cand)["rougeL"]
        a_rouge_scores.append(rouge_score)
    consistency_score = sum(a_rouge_scores) / len(a_rouge_scores) if a_rouge_scores else 0.0
    
    # Balance for Diversity and Consistency!!!! with Candidates from A (Trade-off)
    diversity_weight = 0.7  # Higher weight for diversity
    consistency_weight = 0.3  # Lower weight for consistency
    a_interaction = diversity_weight * diversity_score + consistency_weight * consistency_score
    
    # 3. Fluency - Perplexity (Lower PPL -> Higher Score)
    fluency = 0.0
    if model is not None and tokenizer is not None:
        ppl = compute_perplexity(model, tokenizer, generated, model.device)
        fluency = -np.log(ppl)  
    
    reward_b = (
        lambda1 * a_interaction +     
        lambda2 * gold_similarity +   
        lambda3 * fluency        
    )

    print_log(f"=== Adapter B Reward (Balanced) ===")
    print_log(f"    Generated: {generated}")
    print_log(f"    Reference: {references}")
    print_log(f"    Gold ROUGE-L: {gold_similarity:.4f}")
    print_log(f"    Interactive BLEU: {interactive_bleu:.4f}")
    print_log(f"    Diversity Score: {diversity_score:.4f}")
    print_log(f"    Consistency Score: {consistency_score:.4f}")
    print_log(f"    A Interaction: {a_interaction:.4f}")
    print_log(f"    Fluency (PPL): {fluency:.4f}")
    print_log(f"    Final Reward: {reward_b:.4f}")
        
    return reward_b
