# ./metrics.py - 개선된 버전

import numpy as np
from evaluate import load
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
from typing import List, Dict
from datetime import datetime
import gc

def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def print_log(message: str, prefix: str="LOG") -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

# Cache for ROUGE scorer to avoid reloading
_rouge_scorer = None

def get_rouge_scorer():
    """Get cached ROUGE scorer"""
    global _rouge_scorer
    if _rouge_scorer is None:
        _rouge_scorer = load("rouge")
    return _rouge_scorer

def compute_interactive_bleu(generated_candidates: List[str], adapter_a_candidates: List[str]) -> float:
    """
    Compute Interactive BLEU between Adapter B candidates and Adapter A candidates
    Lower values indicate higher diversity (better for Adapter B)
    """
    if not generated_candidates or not adapter_a_candidates:
        return 0.0
    
    smoothing_function = SmoothingFunction().method1
    total_bleu = 0.0
    pair_count = 0

    # Compare every generated candidate with every Adapter A candidate
    for gen_cand in generated_candidates:
        for a_cand in adapter_a_candidates:
            try:
                tokens_gen = gen_cand.strip().split()
                tokens_a = a_cand.strip().split()
                
                if not tokens_gen or not tokens_a:
                    continue
                
                bleu = sentence_bleu([tokens_a], tokens_gen, smoothing_function=smoothing_function)
                total_bleu += bleu
                pair_count += 1
            except Exception as e:
                print_log(f"Error computing BLEU: {e}")
                continue

    return total_bleu / pair_count if pair_count > 0 else 0.0

def compute_rouge(generated: str, references: str, rouge_types: List[str] = ["rouge1", "rouge2", "rougeL"]) -> Dict[str, float]:
    """Compute ROUGE scores with caching and error handling"""
    if not generated.strip() or not references.strip():
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    try:
        scorer = get_rouge_scorer()
        scores = scorer.compute(
            predictions=[generated.strip()], 
            references=[references.strip()], 
            rouge_types=rouge_types
        )
        
        return {
            "rouge1": float(scores.get("rouge1", 0.0)),
            "rouge2": float(scores.get("rouge2", 0.0)),
            "rougeL": float(scores.get("rougeL", 0.0)),
        }
    except Exception as e:
        print_log(f"Error computing ROUGE: {e}")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

def compute_perplexity(model, tokenizer, text: str, device="cuda") -> float:
    """Compute perplexity with proper error handling and memory management"""
    if not text.strip():
        return float('inf')
    
    try:
        model.eval()
        with torch.no_grad():
            # Tokenize with proper truncation
            inputs = tokenizer(
                text.strip(), 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Compute loss
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss

            # Compute perplexity
            perplexity = torch.exp(loss).item()
            
            # Clean up
            del inputs, outputs
            
            return min(perplexity, 1000.0)  # Cap extremely high values
            
    except Exception as e:
        print_log(f"Error computing perplexity: {e}")
        return 100.0  # Return reasonable default
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def compute_adapter_a_reward(generated: str, references: str, lambda1: float = 0.5, lambda2: float = 0.3, lambda3: float = 0.2) -> float:
    """
    Compute reward for Adapter A based on ROUGE scores
    Higher ROUGE scores = Higher reward (better accuracy)
    """
    if not generated.strip() or not references.strip():
        return 0.0
    
    try:
        rouge_scores = compute
