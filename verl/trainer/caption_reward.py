"""
Caption质量奖励函数
结合多个指标评估生成的caption质量
"""

import torch
from typing import List, Dict
import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider

class CaptionReward:
    def __init__(self, weights=None):
        """
        初始化奖励计算器
        weights: 各指标权重，默认 {'cider': 0.5, 'bleu': 0.2, 'rouge': 0.2, 'meteor': 0.1}
        """
        self.weights = weights or {
            'cider': 0.5,   # CIDEr最适合caption任务
            'bleu': 0.2,
            'rouge': 0.2,
            'meteor': 0.1
        }
        
        # 初始化评估器
        self.scorers = {
            'bleu': Bleu(4),
            'rouge': Rouge(),
            'meteor': Meteor(),
            'cider': Cider()
        }
    
    def compute_reward(
        self, 
        generated_captions: List[str], 
        reference_captions: List[str]
    ) -> torch.Tensor:
        """
        计算奖励分数
        
        Args:
            generated_captions: 生成的captions列表
            reference_captions: 参考captions列表
            
        Returns:
            奖励分数tensor，shape: (batch_size,)
        """
        batch_size = len(generated_captions)
        rewards = torch.zeros(batch_size)
        
        # 准备评估格式
        gts = {i: [ref] for i, ref in enumerate(reference_captions)}
        res = {i: [gen] for i, gen in enumerate(generated_captions)}
        
        # 计算各指标分数
        total_scores = np.zeros(batch_size)
        
        for metric_name, scorer in self.scorers.items():
            score, _ = scorer.compute_score(gts, res)
            
            # 处理BLEU返回多个分数的情况
            if isinstance(score, list):
                score = score[-1]  # 使用BLEU-4
            
            # 确保score是数组
            if not isinstance(score, np.ndarray):
                score = np.array([score] * batch_size)
            
            # 加权累加
            weight = self.weights.get(metric_name, 0)
            total_scores += weight * score
        
        rewards = torch.from_numpy(total_scores).float()
        
        return rewards
    
    def compute_reward_with_details(
        self, 
        generated_captions: List[str], 
        reference_captions: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        计算奖励并返回各指标详情
        """
        batch_size = len(generated_captions)
        
        # 准备评估格式
        gts = {i: [ref] for i, ref in enumerate(reference_captions)}
        res = {i: [gen] for i, gen in enumerate(generated_captions)}
        
        results = {}
        total_scores = np.zeros(batch_size)
        
        for metric_name, scorer in self.scorers.items():
            score, _ = scorer.compute_score(gts, res)
            
            if isinstance(score, list):
                score = score[-1]
            
            if not isinstance(score, np.ndarray):
                score = np.array([score] * batch_size)
            
            results[metric_name] = torch.from_numpy(score).float()
            
            weight = self.weights.get(metric_name, 0)
            total_scores += weight * score
        
        results['total'] = torch.from_numpy(total_scores).float()
        
        return results