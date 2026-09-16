#!/bin/bash
# ==============================================================================
# 端到端串联质谱多标签气味预测模型 - SCENT Open-POM 对齐与云算力训练脚本
# 适用环境: Linux / macOS / Windows, NVIDIA GPU (RTX 3090/4090 / A100) 或 MPS / CPU
# ==============================================================================

set -e

echo "=== 1. 检查运行环境与 GPU 状态 ==="
nvidia-smi || echo "未检测到 nvidia-smi，将以 CPU/MPS 模式运行"

mkdir -p runs checkpoints

# 阶段选择: 
# stage1: 仅运行 SCENT 阶段一多模态对比对齐预训练 (MS <-> Open-POM InfoNCE)
# stage2: 运行下游 138 维气味分类训练 (可加载阶段一预训练权重或开启联合对齐)
# full:   完整依次执行 阶段一 -> 阶段二
STAGE="full"
if [ "$1" != "" ]; then
    STAGE="$1"
fi

if [ "$STAGE" = "stage1" ] || [ "$STAGE" = "full" ]; then
    echo "=================================================================="
    echo "=== 启动 SCENT 阶段一: 质谱与 Open-POM 跨模态对比预训练 ==="
    echo "=================================================================="
    python src/pretrain_align.py \
        --data_dir GS_LF_MS_DATA \
        --epochs 30 \
        --batch_size 64 \
        --lr 1e-4 \
        --tau 0.03 \
        --save_dir checkpoints \
        --pom_cache pom_cache.pt
fi

if [ "$STAGE" = "stage2" ] || [ "$STAGE" = "full" ]; then
    echo "=================================================================="
    echo "=== 启动 SCENT 阶段二: 下游气味标签预测与感知微调 ==="
    echo "=================================================================="
    PRETRAIN_FLAG=""
    if [ -f "checkpoints/scent_aligned_ms_encoder.pth" ]; then
        PRETRAIN_FLAG="--pretrained_align_ckpt checkpoints/scent_aligned_ms_encoder.pth"
        echo "检测到阶段一对齐权重，将自动加载: checkpoints/scent_aligned_ms_encoder.pth"
    fi

    python src/train.py \
        --data_dir GS_LF_MS_DATA \
        --split_by molecule \
        --loss_type asl \
        --batch_size 64 \
        --lr 2e-4 \
        --weight_decay 1e-4 \
        --epochs 100 \
        --d_model 256 \
        --latent_dim 256 \
        --num_layers 4 \
        --nhead 8 \
        --max_seq_len 200 \
        --use_pom_align \
        --lambda_align 0.2 \
        --tau 0.03 \
        --save_path best_model.pth \
        --log_dir runs/odor_scent_experiment \
        $PRETRAIN_FLAG
fi

echo "=== 训练流程全部完成，最佳模型权重与测试报告已保存 ==="
