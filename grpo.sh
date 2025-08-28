wandb login --relogin 8232a08ec8f9d8913ab5a533db16b5c5ced16fe7

MODEL_NAME="/home/nlplab/hdd1/hclt_ss/output/kanana_1.5_8b_instruct_2505_20250827_014625/checkpoint-5300"
TRAIN_DATA="../dataset/함의분석_train.json"
VAL_DATA="../dataset/함의분석_valid.json"
OUTPUT_DIR="../output/grpo_$(date +%Y%m%d_%H%M%S)"
PPL_MODEL_PATH="kakaocorp/kanana-1.5-8b-instruct-2505"

EPOCHS=3
BATCH_SIZE=8
LEARNING_RATE=2e-5
NUM_CANDIDATES=5

# Adapter A reward weights (ROUGE-based)
LAMBDA1=0.5  # ROUGE-1 weight
LAMBDA2=0.3  # ROUGE-2 weight
LAMBDA3=0.2  # ROUGE-L weight

# LoRA configuration
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.1

DEVICE="cuda"

mkdir -p $OUTPUT_DIR

python main.py \
    --model_name $MODEL_NAME \
    --train_data $TRAIN_DATA \
    --val_data $VAL_DATA \
    --output_dir $OUTPUT_DIR \
    --ppl_model $PPL_MODEL_PATH \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_candidates $NUM_CANDIDATES \
    --lambda1 $LAMBDA1 \
    --lambda2 $LAMBDA2 \
    --lambda3 $LAMBDA3 \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --device $DEVICE