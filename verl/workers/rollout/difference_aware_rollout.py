# /root/autodl-tmp/EasyR1/verl/workers/rollout/difference_aware_rollout.py

"""
扩展 Rollout 以支持参考模型生成
"""

import torch
from typing import Dict, List, Any
from verl.workers.rollout.vllm_rollout_spmd import vLLMRollout

class DifferenceAwareRollout(vLLMRollout):
    """
    扩展 vLLM Rollout，增加参考模型生成功能
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generate_ref_sequences = True
        print("[DifferenceAwareRollout] 初始化，将生成参考序列")
    
    def generate_sequences_with_ref(self, prompts: List[str], images: List = None):
        """
        同时生成 policy 和 reference 序列
        
        Returns:
            {
                'policy_sequences': [...],
                'ref_sequences': [...],
                'policy_token_ids': tensor,
                'ref_token_ids': tensor,
            }
        """
        # 1. 生成 policy 序列（原有逻辑）
        policy_output = super().generate_sequences(prompts, images)
        
        # 2. 生成 reference 序列
        # 这里复用同一个模型，但可以用不同的参数
        # 在实际中，ref 模型应该是冻结的早期检查点
        ref_output = self._generate_ref_sequences(prompts, images)
        
        return {
            'policy_sequences': policy_output['sequences'],
            'ref_sequences': ref_output['sequences'],
            'policy_token_ids': policy_output['token_ids'],
            'ref_token_ids': ref_output['token_ids'],
            'prompts': prompts,
            'images': images,
        }
    
    def _generate_ref_sequences(self, prompts: List[str], images: List = None):
        """
        使用参考模型生成序列
        
        注意：这里需要一个独立的 ref model 实例
        或者使用相同模型但温度不同
        """
        # 临时方案：使用不同的采样参数模拟 ref model
        original_temp = self.sampling_params.get('temperature', 0.8)
        original_top_p = self.sampling_params.get('top_p', 0.9)
        
        # Ref model 用更确定的采样
        self.sampling_params['temperature'] = 0.7
        self.sampling_params['top_p'] = 0.85
        
        ref_output = super().generate_sequences(prompts, images)
        
        # 恢复原参数
        self.sampling_params['temperature'] = original_temp
        self.sampling_params['top_p'] = original_top_p
        
        return ref_output