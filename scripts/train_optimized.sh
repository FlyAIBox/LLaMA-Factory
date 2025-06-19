#!/bin/bash

# 海光DCU优化训练脚本
# 适用于DeepSeek-R1-32B-Distill模型

echo "=== 开始优化训练流程 ==="

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "1. 清理GPU内存..."
python scripts/clear_gpu_memory.py

echo "2. 开始训练..."

# 方案1: 使用优化配置（推荐首选）
echo "尝试方案1: 优化配置（序列长度512，梯度累积32）"
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --preprocessing_num_workers 8 \
    --finetuning_type lora \
    --template deepseek3 \
    --flash_attn auto \
    --dataset_dir /root/AI-BOX/code/dcu/easy-dataset/local-db/BRSJUcZdjjho \
    --dataset "[Easy Dataset] [BRSJUcZdjjho] Alpaca" \
    --cutoff_len 512 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir saves/DeepSeek-R1-32B-Distill/lora/train_optimized_$(date +%Y-%m-%d-%H-%M-%S) \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all \
    --deepspeed cache/ds_z3_offload_aggressive.json \
    --dataloader_pin_memory False \
    --gradient_checkpointing True

# 如果方案1失败，尝试方案2
if [ $? -ne 0 ]; then
    echo "方案1失败，尝试方案2: 量化训练"
    
    # 清理内存
    python scripts/clear_gpu_memory.py
    
    llamafactory-cli train \
        --stage sft \
        --do_train True \
        --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
        --preprocessing_num_workers 8 \
        --finetuning_type lora \
        --template deepseek3 \
        --flash_attn auto \
        --dataset_dir /root/AI-BOX/code/dcu/easy-dataset/local-db/BRSJUcZdjjho \
        --dataset "[Easy Dataset] [BRSJUcZdjjho] Alpaca" \
        --cutoff_len 512 \
        --learning_rate 5e-05 \
        --num_train_epochs 3.0 \
        --max_samples 100000 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 32 \
        --lr_scheduler_type cosine \
        --max_grad_norm 1.0 \
        --logging_steps 1 \
        --save_steps 100 \
        --warmup_steps 0 \
        --packing False \
        --report_to none \
        --output_dir saves/DeepSeek-R1-32B-Distill/lora/train_quantized_$(date +%Y-%m-%d-%H-%M-%S) \
        --bf16 True \
        --plot_loss True \
        --trust_remote_code True \
        --ddp_timeout 180000000 \
        --include_num_input_tokens_seen True \
        --optim adamw_torch \
        --lora_rank 8 \
        --lora_alpha 16 \
        --lora_dropout 0 \
        --lora_target all \
        --deepspeed cache/ds_z2_offload_config.json \
        --quantization_bit 4 \
        --dataloader_pin_memory False \
        --gradient_checkpointing True
fi

# 如果方案2也失败，尝试方案3
if [ $? -ne 0 ]; then
    echo "方案2失败，尝试方案3: 进一步降低序列长度"
    
    # 清理内存
    python scripts/clear_gpu_memory.py
    
    llamafactory-cli train \
        --stage sft \
        --do_train True \
        --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
        --preprocessing_num_workers 4 \
        --finetuning_type lora \
        --template deepseek3 \
        --flash_attn auto \
        --dataset_dir /root/AI-BOX/code/dcu/easy-dataset/local-db/BRSJUcZdjjho \
        --dataset "[Easy Dataset] [BRSJUcZdjjho] Alpaca" \
        --cutoff_len 256 \
        --learning_rate 5e-05 \
        --num_train_epochs 3.0 \
        --max_samples 100000 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 64 \
        --lr_scheduler_type cosine \
        --max_grad_norm 1.0 \
        --logging_steps 1 \
        --save_steps 100 \
        --warmup_steps 0 \
        --packing False \
        --report_to none \
        --output_dir saves/DeepSeek-R1-32B-Distill/lora/train_minimal_$(date +%Y-%m-%d-%H-%M-%S) \
        --bf16 True \
        --plot_loss True \
        --trust_remote_code True \
        --ddp_timeout 180000000 \
        --include_num_input_tokens_seen True \
        --optim adamw_torch \
        --lora_rank 8 \
        --lora_alpha 16 \
        --lora_dropout 0 \
        --lora_target all \
        --deepspeed cache/ds_z2_offload_config.json \
        --quantization_bit 4 \
        --dataloader_pin_memory False \
        --gradient_checkpointing True
fi

echo "=== 训练流程完成 ===" 