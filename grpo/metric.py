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
        rouge_scores = compute_rouge(generated, references)

        reward_a = (
            lambda1 * rouge_scores["rouge1"] + 
            lambda2 * rouge_scores["rouge2"] + 
            lambda3 * rouge_scores["rougeL"]
        )

        # Ensure reward is bounded
        reward_a = max(0.0, min(1.0, reward_a))
        
        return float(reward_a)
        
    except Exception as e:
        print_log(f"Error computing Adapter A reward: {e}")
        return 0.0

def compute_adapter_b_reward(generated: str, references: str, adapter_a_cands: List[str], 
                           model=None, tokenizer=None, device="cuda",
                           lambda1: float = 1.0, lambda2: float = 0.5, lambda3: float = 0.1) -> float:
    """
    Compute reward for Adapter B based on diversity and quality
    
    Reward formula: R_B = λ1·(-interactive_BLEU) + λ2·(ROUGE-L) + λ3·(-PPL_penalty)
    
    Args:
        generated: Generated text from Adapter B
        references: Ground truth reference
        adapter_a_cands: List of candidates from Adapter A
        model: Model for perplexity calculation
        tokenizer: Tokenizer for perplexity calculation
        device: Device for computation
        lambda1: Weight for negative interactive BLEU (diversity reward)
        lambda2: Weight for ROUGE-L (quality reward)
        lambda3: Weight for negative PPL penalty (fluency reward)
    """
    if not generated.strip():
        return 0.0
    
    try:
        # 1. Interactive BLEU (diversity component - lower is better)
        interactive_bleu = 0.0
        if adapter_a_cands:
            # Compare generated text with all Adapter A candidates
            interactive_bleu = compute_interactive_bleu([generated], adapter_a_cands)
        
        # 2. ROUGE-L (quality component - higher is better)
        rouge_l = 0.0
        if references.strip():
            rouge_scores = compute_rouge(generated, references, rouge_types=["rougeL"])
            rouge_l = rouge_scores["rougeL"]
        
        # 3. Perplexity penalty (fluency component - lower PPL is better)
        ppl_penalty = 0.0
        if model is not None and tokenizer is not None:
            try:
                ppl = compute_perplexity(model, tokenizer, generated, device)
                # Convert to penalty: higher PPL = higher penalty
                # Use log to normalize and cap the penalty
                ppl_penalty = np.log(max(1.0, min(ppl, 1000.0))) / np.log(100.0)  # Normalize to [0, 2] range
            except Exception as e:
                print_log(f"Error computing PPL penalty: {e}")
                ppl_penalty = 0.5  # Default penalty
        
        # Compute final reward
        # Note: interactive_bleu and ppl_penalty are negated because lower values are better
        reward_b = (
            lambda1 * (-interactive_bleu) +     # Diversity: lower BLEU with A candidates is better
            lambda2 * rouge_l +                 # Quality: higher ROUGE-L with reference is better
            lambda3 * (-ppl_penalty)            # Fluency: lower perplexity is better
        )
        
        # Optional: Add small bonus for reasonable length
        gen_len = len(generated.split())
        if 5 <= gen_len <= 50:  # Reasonable length range
            reward_b += 0.1
        
        # Ensure reward is reasonable (but don't strictly bound it for GRPO)
        reward_b = max(-2.0, min(2.0, reward_b))
        
        return float(reward_b)
        
    except Exception as e:
        print_log(f"Error computing Adapter B reward: {e}")
        return 0.0

def compute_batch_rewards_a(generated_list: List[str], references_list: List[str], 
                           lambda1: float = 0.5, lambda2: float = 0.3, lambda3: float = 0.2) -> List[float]:
    """Compute Adapter A rewards for a batch"""
    rewards = []
    for generated, reference in zip(generated_list, references_list):
        reward = compute_adapter_a_reward(generated, reference, lambda1, lambda2, lambda3)
        rewards.append(reward)
    return rewards

def compute_batch_rewards_b(generated_list: List[str], references_list: List[str], 
                           adapter_a_cands_list: List[List[str]], model=None, tokenizer=None, 
                           device="cuda", lambda1: float = 1.0, lambda2: float = 0.5, lambda3: float = 0.1) -> List[float]:
    """Compute Adapter B rewards for a batch"""
    rewards = []
    for generated, reference, a_cands in zip(generated_list, references_list, adapter_a_cands_list):
        reward = compute_adapter_b_reward(
            generated, reference, a_cands, model, tokenizer, device, lambda1, lambda2, lambda3
        )
        rewards.append(reward)
    return rewards

# Utility functions for debugging
def log_reward_components(generated: str, references: str, adapter_a_cands: List[str] = None, 
                         reward_type: str = "A", **kwargs):
    """Log detailed reward components for debugging"""
    if reward_type == "A":
        rouge_scores = compute_rouge(generated, references)
        print_log(f"=== Adapter A Reward Components ===")
        print_log(f"Generated: {generated[:50]}...")
        print_log(f"Reference: {references[:50]}...")
        print_log(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
        print_log(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
        print_log(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
        
        reward = compute_adapter_a_reward(generated, references, **kwargs)
        print_log(f"Final Reward: {reward:.4f}")
        
    elif reward_type == "B" and adapter_a_cands:
        interactive_bleu = compute_interactive_bleu([generated], adapter_a_cands)
        rouge_scores = compute_rouge(generated, references, rouge_types=["rougeL"])
        rouge_l = rouge_scores["rougeL"]
        
        print_log(f"=== Adapter B Reward Components ===")
        print_log(f"Generated: {generated[:50]}...")
        print_log(f"Reference: {references[:50]}...")
        print_log(f"Adapter A Cands: {len(adapter_a_cands)} candidates")
        print_log(f"Interactive BLEU: {interactive_bleu:.4f}")
        print_log(f"ROUGE-L: {rouge_l:.4f}")
        
        model = kwargs.get('model')
        tokenizer = kwargs.get('tokenizer')
        if model and tokenizer:
            ppl = compute_perplexity(model, tokenizer, generated, kwargs.get('device', 'cuda'))
            print_log(f"Perplexity: {ppl:.4f}")
        
        reward = compute_adapter_b_reward(generated, references, adapter_a_cands, **kwargs)
        print_log(f"Final Reward: {reward:.4f}")
