# But Not Used//////.....
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Any, List, Tuple
import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def print_log(message, prefix="LOG"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{prefix}] {message}")

def get_model_id_from_path(model_path):
    return model_path.split('/')[-1].replace('-', '_')

class RewardCalculator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing_function = SmoothingFunction().method1

    # For 'Expert A'
    def consistency_reward(self, premise: str, proposition: str, explanation: str, predicted_label: str, true_label: str, alpha: int) -> float:
        label_reward = 1.0 if predicted_label == true_label else -1.0

        # Heuristic: Check if explanation contains key terms from premise and proposition
        explanation_consistency = self.check_if_explanation_consistent(explanation, predicted_label)

        return alpha * label_reward + (1 - alpha) * explanation_consistency

    # For 'Expert B'
    def factual_reward(self, premise: str, proposition: str, explanation: str, reference_explanations: List[str] = None, beta: float = 0.7) -> float:

        # Heuristic: overlap with premise & proposition
        overlap_score = self.text_overlap(explanation, premise + " " + proposition)

        rouge_score = 0.0
        if reference_explanations:
            scores = [self.rouge_scorer.score(ref, explanation) for ref in reference_explanations]
            rouge_score = np.mean([s["rougeL"].fmeasure for s in scores])

        return beta * overlap_score + (1 - beta) * rouge_score
        
    
    # For 'Expert C'
    def diversity_reward(self, explanation: str, other_explanations: List[str], gamma1: float = 0.4, gamma2: float = 0.3, gamma3: float = 0.3) -> float:
        self_bleu = np.mean([
            sentence_bleu([other.split()], explanation.split(), smoothing_function=self.smoothing_function)
            for other in other_explanations if other.strip()
        ]) if other_explanations else 0.0

        kl_score = self.approx_kl_divergence(explanation, other_explanations)

        ppl_score = 1.0 / (1.0 + len(explanation.split()))

        return -gamma1 * self_bleu - gamma2 * kl_score + gamma3 * ppl_score

    def approx_kl_divergence(self, explanations: str, others: List[str]) -> float:
        def word_dist(text):
            words = text.split()
            total = len(words) + 1e-8
            return {w: words.count(w)/total for w in set(words)}

        explanation = word_dist(explanations)
        if not others:
            return 0.0

        q = word_dist(" ".join(others))
        kl = 0.0
        for w, explanation_w in explanation.items():
            q_w = q.get(w, 1e-8)
            kl += p_w * np.log(p_w / explanation_w)

        return kl
