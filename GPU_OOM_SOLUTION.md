# 海光DCU训练DeepSeek-R1-32B-Distill内存不足解决方案

## 问题分析

### 当前问题
- **错误类型**: `HIP out of memory. Tried to allocate 1.45 GiB`
- **环境**: 海光DCU K100 8卡，每卡64GB显存
- **显存使用率**: 76-80%（已接近满载）
- **模型规模**: DeepSeek-R1-32B-Distill（32B参数）

### 根本原因
1. **模型参数量大**: 32B参数的模型即使使用LoRA也需要大量显存
2. **序列长度过长**: cutoff_len=1024占用过多激活值内存
3. **梯度累积不足**: gradient_accumulation_steps=16可能不够
4. **DeepSpeed配置未充分优化**: 参数offload配置有优化空间

## 解决方案

### 方案1: 优化配置（推荐首选）

**关键优化点**:
- ✅ 序列长度: 1024 → 512 (节省50%激活值内存)
- ✅ 梯度累积: 16 → 32 (保持有效批次大小)
- ✅ 激进DeepSpeed配置: 更小的prefetch和persistence阈值
- ✅ 禁用dataloader pin_memory
- ✅ 启用gradient checkpointing

**使用方法**:
```bash
# 使用优化配置文件
llamafactory-cli train config/dtk2504-llamafactory092-ds32b-optimized.yaml

# 或使用优化脚本（自动尝试多种方案）
./scripts/train_optimized.sh
```

### 方案2: 量化训练

**关键优化点**:
- ✅ 4bit量化: 大幅减少模型内存占用
- ✅ ZeRO Stage 2: 更好的量化兼容性
- ✅ 序列长度512: 平衡内存和效果

**使用方法**:
```bash
llamafactory-cli train config/dtk2504-llamafactory092-ds32b-quantized.yaml
```

### 方案3: 极限优化

**关键优化点**:
- ✅ 序列长度: 256 (适合简单任务)
- ✅ 梯度累积: 64 (保持训练稳定性)
- ✅ 4bit量化 + ZeRO Stage 2
- ✅ 减少preprocessing workers

## 立即执行步骤

### 1. 清理当前GPU内存
```bash
# 运行内存清理脚本
python scripts/clear_gpu_memory.py

# 或手动清理
pkill -f llamafactory
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"
```

### 2. 检查GPU状态
```bash
rocm-smi
```

### 3. 执行优化训练
```bash
# 方案1: 优化配置（推荐）
./scripts/train_optimized.sh

# 或单独执行某个方案
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
```

## 配置文件说明

### 1. 优化配置 (`dtk2504-llamafactory092-ds32b-optimized.yaml`)
- 适合大多数场景
- 保持较好的模型效果
- 内存优化适中

### 2. 量化配置 (`dtk2504-llamafactory092-ds32b-quantized.yaml`)
- 大幅节省内存
- 训练速度可能略慢
- 效果损失很小

### 3. 激进DeepSpeed配置 (`ds_z3_offload_aggressive.json`)
- 最大化CPU offload
- 最小化GPU常驻参数
- 可能影响训练速度

## 监控建议

### 训练过程中监控
```bash
# 监控GPU内存使用
watch -n 1 rocm-smi

# 监控训练日志
tail -f logs/train.log

# 监控系统资源
htop
```

### 关键指标
- GPU内存使用率 < 90%
- 训练loss正常下降
- 无OOM错误

## 进一步优化建议

如果上述方案仍有问题，可以考虑：

1. **使用更小的模型**: 切换到7B或13B版本
2. **减少LoRA rank**: 从8降到4
3. **使用gradient accumulation**: 进一步增加到64或128
4. **分阶段训练**: 先训练embeddings，再训练其他层
5. **使用模型并行**: 结合tensor parallel

## 预期效果

使用优化配置后，预期：
- ✅ GPU内存使用率降至60-70%
- ✅ 训练正常启动和进行
- ✅ 保持相似的训练效果
- ✅ 训练速度可能略有下降（可接受） 