# README.md
"""
# Dual Adapter GRPO Training

하나의 SFT 모델을 기반으로 두 개의 어댑터를 사용해 정확성과 다양성을 각각 특화시키는 GRPO 학습 구현

## 파일 구조

```
├── metrics.py          # 평가 지표 (Interactive BLEU, ROUGE, PPL)
├── model_utils.py      # 모델 로딩 및 어댑터 생성
├── data_utils.py       # 데이터 전처리 및 저장/로딩
├── generation.py       # 텍스트 생성 유틸리티
├── training.py         # GRPO 학습 함수들
└── main.py            # 메인 훈련 스크립트
```

## 설치 요구사항

```bash
pip install torch transformers peft trl datasets rouge-score nltk numpy tqdm
```

## 사용법

### 1. 기본 훈련 실행
```bash
python main.py \
    --model_name "your_sft_model_name" \
    --data_path "path/to/nli_data.json" \
    --output_dir "./outputs" \
    --epochs 3 \
    --batch_size 8 \
    --lr 5e-5
```

### 2. 데이터 형식
```json
[
  {
    "premise": "또한 '브람스를 좋아하세요?'는 SBS 단편드라마...",
    "proposition": "박은빈이 출연한 SBS 월화 드라마의 제목에는 작곡가 이름이 들어간다.",
    "label": "entailment",
    "output": "정답 설명문 (optional)"
  }
]
```

### 3. 단계별 실행

#### Step 1: 모델 및 어댑터 생성
```python
from model_utils import load_base_model, create_lora_config, create_dual_adapters

base_model, tokenizer = load_base_model("your_model_name")
lora_config = create_lora_config()
adapter_a, adapter_b = create_dual_adapters(base_model, lora_config)
```

#### Step 2: Adapter A 훈련 (정확성 특화)
```python
from training import train_adapter_grpo
from metrics import compute_adapter_a_reward

adapter_a = train_adapter_grpo(
    adapter_a, tokenizer, data_samples,
    reward_function=compute_adapter_a_reward
)
```

#### Step 3: Adapter A 후보 생성 및 저장
```python
from generation import batch_generate_adapter_a_candidates
from data_utils import save_candidates_to_json

candidates = batch_generate_adapter_a_candidates(adapter_a, tokenizer, data_samples)
save_candidates_to_json(candidates, "adapter_a_candidates.json")
```

#### Step 4: Adapter B 훈련 (다양성 특화)
```python
from metrics import compute_adapter_b_reward

adapter_b_reward_fn = lambda gen, ref, a_cands: compute_adapter_b_reward(
    gen, ref, a_cands, adapter_b, tokenizer,
    lambda_1=0.4,  # Interactive BLEU 가중치
    lambda_2=0.4,  # ROUGE-L 가중치  
    lambda_3=0.2   # Perplexity 가중치
)

adapter_b = train_adapter_grpo(
    adapter_b, tokenizer, data_samples,
    reward_function=adapter_b_reward_fn,
    adapter_a_candidates=candidates
)
```
