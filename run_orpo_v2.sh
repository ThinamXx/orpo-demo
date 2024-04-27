#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --cpus-per-gpu=8


accelerate launch --config_file ./orpo/src/accelerate/fsdp.yaml orpo_v2.py \
    --lr 5e-6 \
    --warmup_steps 100 \
    --model_name facebook/opt-1.3b \
    --data_name mlabonne/orpo-dpo-mix-40k \
    --num_train_epochs 1 \
    --prompt_max_length 128 \
    --response_max_length 512 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --num_proc 8 \
    --flash_attention_2 \
    --enable_qlora \
    --enable_lora