# /root/autodl-tmp/EasyR1/examples/train_difference_aware.py

"""
使用 Difference-Aware Loss 训练的启动脚本
"""

import sys
sys.path.insert(0, '/root/autodl-tmp/EasyR1')

import ray
from omegaconf import OmegaConf
from verl.trainer.ray_trainer import DifferenceAwareRayPPOTrainer

def main():
    # 加载配置
    config = OmegaConf.load("examples/caption_grpo_config_difference_aware.yaml")
    
    # 初始化 Ray
    ray.init(ignore_reinit_error=True)
    
    # 创建训练器
    print("创建 DifferenceAwareRayPPOTrainer...")
    trainer = DifferenceAwareRayPPOTrainer(config)
    
    # 开始训练
    print("开始训练...")
    trainer.fit()
    
    # 关闭 Ray
    ray.shutdown()

if __name__ == "__main__":
    main()