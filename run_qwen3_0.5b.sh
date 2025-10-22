#!/bin/bash
set -x

# 清理显存
echo "清理 GPU 显存..."
nvidia-smi --gpu-reset || true

# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ENDPOINT=https://hf-mirror.com  # 使用镜像加速下载

# 模型路径
MODEL_PATH=Qwen/Qwen3-0.5B

# 停止之前的 Ray
echo "停止之前的 Ray..."
ray stop || true
sleep 2

# 启动 Ray
echo "启动 Ray..."
ray start --head --port=6379 --dashboard-host=0.0.0.0

# 显示 GPU 状态
echo "当前 GPU 状态："
nvidia-smi

# 运行训练
echo "开始训练..."
python3 -m verl.trainer.main \
    config=examples/config_qwen3_0.5b.yaml \
    worker.actor.model.model_path=${MODEL_PATH}

# 清理
echo "训练完成，清理 Ray..."
ray stop
