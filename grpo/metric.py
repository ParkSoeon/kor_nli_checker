# ./reward.py

import numpy as np
from evaluate import load
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch

def compute_interactive_bleu(sentence_a, sentence_b):
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

def compute_rouge(generated, references, rouge_types=["rouge1", "rouge2", "rougeL"]):

    scorer = load("rouge")
    scores = scorer.compute(predictions=[generated], references=[references], rouge_types=rouge_types)

    return {
        "rouge1": scores["rouge1"]["f1"],
        "rouge2": scores["rouge2"]["f1"],
        "rougeL": scores["rougeL"]["f1"],
        # "combined": alpha * scores["rouge1"]["f1"] + beta * scores["rouge2"]["f1"] + gamma * scores["rougeL"]["f1"]
        # "combined": alpha * scores["rouge1"] + beta * scores["rouge2"] + gamma * scores["rougeL"]
    }

def compute_perplexity(model, tokenizer, texts, device):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss

        perplexity = torch.exp(loss)

    return perplexity.item()

def compute_adapter_a_reward(generated, references, alpha=0.0, beta=0.0, gamma=0.0):
    rouge_scores = compute_rouge(generated, references)

    reward_a = (
        alpha * rouge_scores["rouge1"] + 
        beta * rouge_scores["rouge2"] + 
        gamma * rouge_scores["rougeL"]
    )

    return reward_a

def compute_adapter_b_reward(generated, references, adapter_a_cands, model=None, tokenizer=None, alpha=0.0, beta=0.0, gamma=0.0):

    interactive_bleu = compute_interactive_bleu([generated], adapter_a_cands)

    rouge_scores = compute_rouge(generated, references)
    rouge_l = rouge_scores["rougeL"]

    ppl_penalty = 0.0
    if model is not None and tokenizer is not None:
        ppl = compute_perplexity(model, tokenizer, generated)
        ppl_penalty = -np.log(ppl) # Lower Perplexity == Higher Reward

    reward_b = (
        - alpha * interactive_bleu + 
        beta * rouge_l -
        gamma * ppl_penalty
    )

    return reward_b
