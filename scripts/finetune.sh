export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))



MAX_SEQ_LENGTH=2048
EPOCH=3
MAX_TRAIN_STEPS=100

##################### Raw
MODEL_NAME=mistralai/Mistral-7B-v0.1
TRAIN_FILE=/data/data/mshen16/fine_nwp/data/ToW-pretrain/Raw.jsonl
OUT_DIR=/data/data/mshen16/fine_nwp/output/mistral_7b/Raw

MODEL_NAME=meta-llama/Llama-2-7b-hf
TRAIN_FILE=/data/data/mshen16/fine_nwp/data/ToW-pretrain/Raw.jsonl
OUT_DIR=/data/data/mshen16/fine_nwp/output/llama2_7b/Raw

MODEL_NAME=meta-llama/Meta-Llama-3-8B
TRAIN_FILE=/data/data/mshen16/fine_nwp/data/ToW-pretrain/Raw.jsonl
OUT_DIR=/data/data/mshen16/fine_nwp/output/llama3_8b/Raw


##################### ToW-NoDeN
MAX_SEQ_LENGTH=3072

MODEL_NAME=mistralai/Mistral-7B-v0.1
TRAIN_FILE=/data/data/mshen16/fine_nwp/data/ToW-pretrain/ToW-NoDeN.jsonl
OUT_DIR=/data/data/mshen16/fine_nwp/output/mistral_7b/ToW-NoDeN

MODEL_NAME=meta-llama/Llama-2-7b-hf
TRAIN_FILE=/data/data/mshen16/fine_nwp/data/ToW-pretrain/ToW-NoDeN.jsonl
OUT_DIR=/data/data/mshen16/fine_nwp/output/llama2_7b/ToW-NoDeN

MODEL_NAME=meta-llama/Meta-Llama-3-8B
TRAIN_FILE=/data/data/mshen16/fine_nwp/data/ToW-pretrain/ToW-NoDeN.jsonl
OUT_DIR=/data/data/mshen16/fine_nwp/output/llama3_8b/ToW-NoDeN


##################### ToW-PartDeN
MODEL_NAME=mistralai/Mistral-7B-v0.1
TRAIN_FILE=/data/data/mshen16/fine_nwp/data/ToW-pretrain/ToW-PartDeN.jsonl
OUT_DIR=/data/data/mshen16/fine_nwp/output/mistral_7b/ToW-PartDeN

MODEL_NAME=meta-llama/Llama-2-7b-hf
TRAIN_FILE=/data/data/mshen16/fine_nwp/data/ToW-pretrain/ToW-PartDeN.jsonl
OUT_DIR=/data/data/mshen16/fine_nwp/output/llama2_7b/ToW-PartDeN

MODEL_NAME=meta-llama/Meta-Llama-3-8B
TRAIN_FILE=/data/data/mshen16/fine_nwp/data/ToW-pretrain/ToW-PartDeN.jsonl
OUT_DIR=/data/data/mshen16/fine_nwp/output/llama3_8b/ToW-PartDeN


##################### ToW
MODEL_NAME=mistralai/Mistral-7B-v0.1
TRAIN_FILE=/data/data/mshen16/fine_nwp/data/ToW-pretrain/ToW.jsonl
OUT_DIR=/data/data/mshen16/fine_nwp/output/mistral_7b/ToW

MODEL_NAME=meta-llama/Llama-2-7b-hf
TRAIN_FILE=/data/data/mshen16/fine_nwp/data/ToW-pretrain/ToW.jsonl
OUT_DIR=/data/data/mshen16/fine_nwp/output/llama2_7b/ToW

MODEL_NAME=meta-llama/Meta-Llama-3-8B
TRAIN_FILE=/data/data/mshen16/fine_nwp/data/ToW-pretrain/ToW.jsonl
OUT_DIR=/data/data/mshen16/fine_nwp/output/llama3_8b/ToW


##################### ToW-em_only
MODEL_NAME=mistralai/Mistral-7B-v0.1 
TRAIN_FILE=/data/data/mshen16/fine_nwp/data/ToW-pretrain/ToW-em_only.jsonl
OUT_DIR=/data/data/mshen16/fine_nwp/output/mistral_7b/ToW-em_only

MODEL_NAME=meta-llama/Llama-2-7b-hf 
TRAIN_FILE=/data/data/mshen16/fine_nwp/data/ToW-pretrain/ToW-em_only.jsonl
OUT_DIR=/data/data/mshen16/fine_nwp/output/llama2_7b/ToW-em_only

MODEL_NAME=meta-llama/Meta-Llama-3-8B 
TRAIN_FILE=/data/data/mshen16/fine_nwp/data/ToW-pretrain/ToW-em_only.jsonl
OUT_DIR=/data/data/mshen16/fine_nwp/output/llama3_8b/ToW-em_only

##################### ToW-no_unpred
MODEL_NAME=mistralai/Mistral-7B-v0.1
TRAIN_FILE=/data/data/mshen16/fine_nwp/data/ToW-pretrain/ToW-no_unpred.jsonl
OUT_DIR=/data/data/mshen16/fine_nwp/output/mistral_7b/ToW-no_unpred

MODEL_NAME=meta-llama/Llama-2-7b-hf  
TRAIN_FILE=/data/data/mshen16/fine_nwp/data/ToW-pretrain/ToW-no_unpred.jsonl
OUT_DIR=/data/data/mshen16/fine_nwp/output/llama2_7b/ToW-no_unpred

MODEL_NAME=meta-llama/Meta-Llama-3-8B 
TRAIN_FILE=/data/data/mshen16/fine_nwp/data/ToW-pretrain/ToW-no_unpred.jsonl
OUT_DIR=/data/data/mshen16/fine_nwp/output/llama3_8b/ToW-no_unpred


accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file /data/data/mshen16/fine_nwp/configs/stage3_no_offloading_accelerate.conf \
    --main_process_port 8000 \
    fine_nwp/finetune.py \
    --model_name_or_path $MODEL_NAME \
    --use_flash_attn \
    --tokenizer_name $MODEL_NAME \
    --use_slow_tokenizer \
    --train_file $TRAIN_FILE \
    --max_seq_length $MAX_SEQ_LENGTH \
    --preprocessing_num_workers 128 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs $EPOCH \
    --max_train_steps $MAX_TRAIN_STEPS \
    --output_dir $OUT_DIR \
    --logging_steps 1 





