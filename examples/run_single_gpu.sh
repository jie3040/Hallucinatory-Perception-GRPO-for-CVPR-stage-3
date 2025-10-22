#!/bin/bash
set -x

# 模型路径
MODEL_PATH=Qwen/Qwen3-4B

# 停止之前的 Ray
ray stop || true

# 启动 Ray
ray start --head --port=6379 --dashboard-host=0.0.0.0

# 运行训练
python3 -m verl.trainer.main \
    config=examples/config_single_gpu.yaml \
    worker.actor.model.model_path=${MODEL_PATH}

# 清理
ray stop
