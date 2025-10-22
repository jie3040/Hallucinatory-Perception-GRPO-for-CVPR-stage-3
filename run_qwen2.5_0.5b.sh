#!/bin/bash
set -x

# 清理显存
nvidia-smi --gpu-reset || true

# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ENDPOINT=https://hf-mirror.com

MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct

# 停止之前的 Ray
ray stop || true
sleep 2

# 启动 Ray
ray start --head --port=6379 --dashboard-host=0.0.0.0

# 显示 GPU 状态
nvidia-smi

# 运行训练
python3 -m verl.trainer.main \
    config=examples/config_qwen2.5_0.5b.yaml \
    worker.actor.model.model_path=${MODEL_PATH}

# 清理
ray stop
