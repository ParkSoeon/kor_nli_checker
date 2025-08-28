export CUDA_VISIBLE_DEVICES="0,1,2"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

nohup bash grpo.sh > ${TIMESTAMP}_train.out 2>&1 &
