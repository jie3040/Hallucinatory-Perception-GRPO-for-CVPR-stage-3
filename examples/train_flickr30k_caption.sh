#!/bin/bash

set -x

echo "================================================================================"
echo "Flickr30k Caption GRPO Training"
echo "================================================================================"
echo "训练数据: /root/autodl-tmp/flickr30k_caption/train (27,912样本)"
echo "验证数据: /root/autodl-tmp/flickr30k_caption/val (3,102样本)"
echo "模型: Qwen2-VL-2B-Instruct"
echo "================================================================================"

# 设置环境变量
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

# 模型路径 (可以改为本地路径)
MODEL_PATH=Qwen/Qwen2-VL-2B-Instruct

# 启动训练
python3 -m verl.trainer.main \
    config=examples/caption_grpo_config.yaml \
    data.train_files=/root/autodl-tmp/flickr30k_caption/train \
    data.val_files=/root/autodl-tmp/flickr30k_caption/val \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=flickr30k_caption_grpo \
    trainer.n_gpus_per_node=1