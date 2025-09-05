# train.py ...... I don't like the name of this file....

from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, TrainingArguments
import torch
from data import load_data, GRPODataset
from typing import Callable, Dict, List
from model import format_input_prompt
from datetime import datetime
from metric import compute_adapter_a_reward, compute_adapter_b_reward

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def print_log(message, prefix="LOG"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def create_grpo_trainer(
    model, tokenizer, dataset, reward_function: Callable,
    output_dir: str, learning_rate: float = 5e-5, batch_size: int = 8, epochs: int = 3, 
    num_candidates: int = 5, **kwargs
):

    num_generations = num_candidates

    if batch_size % num_generations != 0:
        generation_batch_size = ((batch_size // num_generations) + 1) * num_generations
        print_log(f"Adjusting generation_batch_size from {batch_size} to {generation_batch_size} to be divisible by num_generations ({num_generations})")
    else:
        generation_batch_size = batch_size

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_steps=10,
        save_steps=100,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        report_to="wandb",
        dataloader_drop_last=True,
        
        num_generations=num_generations,
        generation_batch_size=generation_batch_size,
        **kwargs
    )

    grpo_trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_function,
    )

    return grpo_trainer

def train_adapter_a(adapter_a, tokenizer, train_data: List[Dict], val_data: List[Dict], 
                    output_dir: str, args) -> torch.nn.Module:
    
    use_chat_template = args.use_chat_template
    print_log(f"Use Chat Template: {use_chat_template}")

    train_dataset = GRPODataset(train_data, tokenizer, use_chat_template=use_chat_template)

    print_log(f"Train Dataset Size: {len(train_dataset)}")
    print_log(f"=== Train Dataset Samples ===")

    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        print_log(f"\n--- Sample {i+1} ---")
        print_log(f"Prompt     : \n{sample['prompt']}")
        print_log(f"Premise    : {sample['premise']}")
        print_log(f"Proposition: {sample['proposition']}")
        print_log(f"Label      : {sample['labels']}")
        print_log(f"Reference  : {sample['reference']}")

    reference_map = {}
    for sample in train_data:
        base_query = format_input_prompt(
            sample['input']["premise"], 
            sample['input']["proposition"], 
            sample['input']["label"]
        )
        
        # Chat template 적용
        if use_chat_template:
            # SFT 베이스라인과 동일한 system prompt 사용
            system_content = """당신은 한국어 자연어 추론(NLI) 전문가입니다. 주어진 전제와 가설을 분석하여 함의 관계를 설명해주세요.

    **중요한 규칙:**
    1. 반드시 '[설명] '으로 시작해서 설명문 생성을 시작하세요.
    2. 설명은 한 문장 이상, 세 문장 이하로 작성하고, 마지막에 전제와 가설의 관계가 함의 또는 모순임을 명확히 드러내야 합니다.
    - 예: '함의이다.', '함의에 해당된다.', '모순이다.', '모순에 속한다.' 등
    3. 전제와 가설의 관계는 무조건 '함의', '모순' 중 하나입니다. '중립'이나, '특정 관계에 해당되지 않는다.' 등의 표현은 허용되지 않습니다.
    4. 설명문은 최대 길이 75토큰을 넘지 않도록 최대한 간결하고 명확하게 작성하세요.
    5. 설명문은 한국어로 작성되어야 합니다.

위의 규칙을 엄격히 준수하여 답변해 주세요."""
            
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": base_query}
            ]
            formatted_query = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            formatted_query = formatted_query.replace('<|end_of_text|>', '')
            empty_system_content = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\n<|eot_id|><|start_header_id|>system<|end_header_id|>"
            if formatted_query.startswith(empty_system_content):
                formatted_query = formatted_query.replace("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\n<|eot_id|>", "<|begin_of_text|>", 1)

        else:
            formatted_query = base_query
            
        reference_map[formatted_query] = sample.get("output", "")

    print_log(f"Reference Map Size: {len(reference_map)}")

    # Define a Reward Function based on ROUGE(for Adapter A)
    def adapter_a_reward_function(completions, **kwargs):
        rewards = []
        
        # Get prompts from kwargs
        prompts = kwargs.get('prompts', [])
        
        # Handle different completion formats
        if isinstance(completions[0], dict):
            completion_texts = [comp.get("content", comp.get("text", str(comp))) for comp in completions]
        else:
            completion_texts = completions
        
        print_log(f"Adapter A Reward Function called with {len(completion_texts)} completions")

        for i, completion_text in enumerate(completion_texts):
            # Get corresponding prompt
            prompt = prompts[i] if i < len(prompts) else ""
            reference = reference_map.get(prompt, "")

            # # Debugging key matching issues(Just in case)
            # if not reference:
            #     print_log(f"=== KEY MATCHING DEBUG ===")
            #     print_log(f"Current prompt length: {len(prompt)}")
            #     print_log(f"Current prompt (first 100 chars): {prompt}...")
            #     print_log(f"Reference map keys count: {len(reference_map)}")
            
            # # Compare with the first key with reference_map
            # if reference_map:
            #     first_key = list(reference_map.keys())[0]
            #     print_log(f"First ref_map key length: {len(first_key)}")
            #     print_log(f"First ref_map key (first 100 chars): {first_key}...")
            #     print_log(f"Keys match: {prompt == first_key}")

            clean_completion = completion_text
            if clean_completion.startswith("[설명] "):
                clean_completion = clean_completion[len("[설명] "):].strip()        
            elif clean_completion.startswith("설명] "):
                clean_completion = clean_completion[len("설명] "):].strip()
            elif clean_completion.startswith("설명 "):
                clean_completion = clean_completion[len("설명 "):].strip()
            elif clean_completion.startswith("[설명 ]"):
                clean_completion = clean_completion[len("[설명 ]"):].strip()

            clean_reference = reference
            if clean_reference.startswith('[설명]'):
                clean_reference = clean_reference[4:].strip()
                
            reward = compute_adapter_a_reward(
                clean_completion, clean_reference, 
                lambda1=args.lambda1, 
                lambda2=args.lambda2, 
                lambda3=args.lambda3
            )
            
            # print_log(f"=== Adapter A Reward {i} ===")
            # print_log(f"    Query    : {prompt}")
            # print_log(f"    Generated: {completion_text}")
            # print_log(f"    Reference: {reference}")
            # print_log(f"    Reward   : {reward:.4f}")
            
            rewards.append(reward)

        return rewards

    trainer = create_grpo_trainer(
        model=adapter_a,
        tokenizer=tokenizer,
        dataset=train_dataset,
        reward_function=adapter_a_reward_function,
        output_dir=f"{output_dir}/adapter_a",
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_candidates=args.num_candidates
    )

    print_log(">> Starting Adapter A Training")
    trainer.train()
    print_log(">> Finished Adapter A Training")
    return adapter_a

def train_adapter_b(adapter_b, tokenizer, train_data: List[Dict], val_data: List[Dict], adapter_a_candidates: Dict[str, List[str]], output_dir: str, args, ppl_model=None) -> torch.nn.Module:

    print_log(f"=== Adapter B Training Setup ===")
    print_log(f"Adapter A Candidates Samples: {len(adapter_a_candidates)}")

    for i, (key, cands) in enumerate(list(adapter_a_candidates.items())[:5]):
        print_log(f"\n--- Adapter A Candidates Sample {i+1} ---")
        print_log(f"Key: {key}")
        for j, cand in enumerate(cands):
            print_log(f"Candidate {j+1}: {cand}")

    train_dataset = GRPODataset(train_data, tokenizer)

    reference_map = {}
    for sample in train_data:
        query = format_input_prompt(sample['input']["premise"], sample['input']["proposition"], sample['input']["label"])
        reference_map[query] = sample.get("output", "")

    # Define a Reward Function based on Interactive BLEU, ROUGE-L, and PPL(for Adapter B)
    def adapter_b_reward_function(completions, **kwargs):
        """
        completions: List of completion dictionaries
        **kwargs: Contains additional info like prompts
        """
        rewards = []
        
        # Get prompts from kwargs
        prompts = kwargs.get('prompts', [])
        
        # Handle different completion formats
        if isinstance(completions[0], dict):
            completion_texts = [comp.get("content", comp.get("text", str(comp))) for comp in completions]
        else:
            completion_texts = completions

        print_log(f"Adapter B Reward Function called with {len(completion_texts)} completions")

        for i, completion_text in enumerate(completion_texts):
            prompt = prompts[i] if i < len(prompts) else ""
            
            key = None
            for k in adapter_a_candidates.keys():
                premise, proposition = k.split(" ||| ")
                if premise in prompt and proposition in prompt:
                    key = k
                    break

            if key is None:
                rewards.append(0.0)
                continue

            a_candidates = adapter_a_candidates[key]
            reference = reference_map.get(prompt, "")

            reward = compute_adapter_b_reward(
                completion_text, reference, a_candidates, 
                tokenizer=tokenizer, model=ppl_model, 
                lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3
            )

            # print_log(f"=== Adapter B Reward {i} ===")
            # print_log(f"    Query    : {prompt}")
            # print_log(f"    Generated: {completion_text}")
            # print_log(f"    Reference: {reference}")
            # for j, cand in enumerate(a_candidates):
            #     print_log(f"    Adapter A Candidates{j+1}: {cand}")
            # print_log(f"    Reward   : {reward:.4f}")
            
            rewards.append(reward)
        
        return rewards

    # Create Trainer
    trainer = create_grpo_trainer(
        model=adapter_b,
        tokenizer=tokenizer,
        dataset=train_dataset,
        reward_function=adapter_b_reward_function,
        output_dir=f"{output_dir}/adapter_b",
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_candidates=args.num_candidates
    )

    print_log(">> Starting Adapter B Training")
    trainer.train()
    print_log(">> Finished Adapter B Training")
    return adapter_b
