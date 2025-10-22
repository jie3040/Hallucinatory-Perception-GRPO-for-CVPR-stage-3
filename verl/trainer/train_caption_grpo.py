"""
Flickr30k Caption GRPO训练脚本
"""

import os
import sys
from pathlib import Path

# 添加EasyR1路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from verl.trainer.main import main
from caption_reward import CaptionReward

def setup_training():
    """设置训练环境"""
    
    # 设置环境变量
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # 确保输出目录存在
    output_dir = Path("/root/autodl-tmp/caption_grpo_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Flickr30k Caption GRPO Training")
    print("=" * 80)
    print(f"训练数据: /root/autodl-tmp/flickr30k_caption/train (27,912样本)")
    print(f"验证数据: /root/autodl-tmp/flickr30k_caption/val (3,102样本)")
    print(f"输出目录: {output_dir}")
    print("=" * 80)
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"✓ GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠ 警告: 未检测到GPU")
    
    print("=" * 80)

if __name__ == "__main__":
    setup_training()
    
    # 启动训练
    # 配置文件路径
    config_path = "examples/caption_grpo_config.yaml"
    
    # 使用verl的训练主函数
    main(config_path)