#!/bin/bash

# 并行训练脚本：使用 GNU Parallel 在多 GPU 上并行运行 AMAPPO / MAPPO 实验
# 用法: bash experiments/para_train.bash [并行数]
# 示例: bash experiments/para_train.bash 4

set -euo pipefail

# ── 配置 ──────────────────────────────────────────────
algos=("amappo" "mappo")
seeds=(42 123 456 789 1024)
epochs=1500
device="cuda"

# 并行任务数，默认 8，可通过命令行参数覆盖
max_jobs="${1:-8}"

# 日志目录
log_dir="logs"
mkdir -p "$log_dir"

# 检测可用 GPU 数量
if command -v nvidia-smi &>/dev/null; then
    num_gpus=$(nvidia-smi -L | wc -l)
    echo "============================================"
    echo "  检测到 ${num_gpus} 块 GPU，最大并行数: ${max_jobs}"
    echo "  算法: ${algos[*]}"
    echo "  种子: ${seeds[*]}"
    echo "  总任务数: $((${#algos[@]} * ${#seeds[@]}))"
    echo "============================================"
else
    echo "警告: 未检测到 nvidia-smi，将使用 CPU 或单 GPU"
    num_gpus=1
fi

# ── 执行 ──────────────────────────────────────────────
# {%} 是 GNU Parallel 的 slot 编号（1, 2, 3, ...），自动轮转分配
# 将 slot 编号映射为 GPU ID: slot 1 -> GPU 0, slot 2 -> GPU 1, ...
# 取模运算确保即使 slot 数超过 GPU 数也能正确分配

echo "开始并行训练..."
echo "日志输出到: ${log_dir}/<algo>_seed<seed>.log"
echo ""

parallel -j "$max_jobs" --eta --joblog "${log_dir}/joblog.csv" '
    gpu_id=$(( ({%} - 1) % {3} ));
    echo "[启动] algo={1}  seed={2}  GPU=$gpu_id  slot={%}";
    CUDA_VISIBLE_DEVICES=$gpu_id \
    python experiments/train.py --algo {1} --epochs {4} --device cuda --seed {2} \
        > "${log_dir}/{1}_seed{2}.log" 2>&1;
    echo "[完成] algo={1}  seed={2}  退出码=$?";
' ::: "${algos[@]}" ::: "${seeds[@]}" ::: "$num_gpus" ::: "$epochs"

# ── 汇总 ──────────────────────────────────────────────
echo ""
echo "============================================"
echo "  所有训练任务已完成！"
echo "  日志目录: ${log_dir}/"
echo "  任务日志: ${log_dir}/joblog.csv"
echo "============================================"

# 检查是否有失败的任务
if [ -f "${log_dir}/joblog.csv" ]; then
    failed=$(awk -F, 'NR>1 && $7!=0 {count++} END {print count+0}' "${log_dir}/joblog.csv")
    if [ "$failed" -gt 0 ]; then
        echo "⚠ 有 ${failed} 个任务失败，请检查日志："
        awk -F, 'NR>1 && $7!=0 {print "  " $2 " -> 退出码 " $7}' "${log_dir}/joblog.csv"
    else
        echo "✓ 所有任务均成功完成"
    fi
fi
