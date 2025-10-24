# /root/autodl-tmp/EasyR1/verl/trainer/difference_aware_loss.py

import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

class DifferenceAwareLoss:
    """
    Difference-Aware SPIN Loss with LCS Masking
    
    核心思想：
    1. 用参考模型生成序列
    2. 计算生成序列与真实标签的 LCS
    3. Mask 掉 LCS 部分
    4. 用 SPIN loss 训练 policy 关注差异部分
    """
    
    def __init__(
        self,
        min_beta: float = 0.1,
        max_beta: float = 0.5,
        loss_type: str = "sigmoid",
        pad_token_id: int = 151643,
    ):
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.loss_type = loss_type
        self.pad_token_id = pad_token_id
        self.current_step = 0
        self.max_steps = 1000
        
    def update_step(self, current_step: int, max_steps: int):
        """更新当前步数"""
        self.current_step = current_step
        self.max_steps = max_steps
        
    def compute_beta(self) -> float:
        """动态计算温度参数"""
        if self.max_steps == 0:
            return self.min_beta
        return self.min_beta + self.current_step / self.max_steps * (self.max_beta - self.min_beta)
    
    def longest_common_subsequence(self, seq1: list, seq2: list) -> list:
        """计算最长公共子序列"""
        # 过滤掉 -100 和 pad_token
        seq1 = [x for x in seq1 if x != -100 and x != self.pad_token_id]
        seq2 = [x for x in seq2 if x != -100 and x != self.pad_token_id]
        
        if not seq1 or not seq2:
            return []
        
        dp = [[0] * (len(seq2) + 1) for _ in range(len(seq1) + 1)]
        
        # 填充 DP 表
        for i in range(1, len(seq1) + 1):
            for j in range(1, len(seq2) + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        # 反向查找 LCS
        lcs = []
        i, j = len(seq1), len(seq2)
        
        while i > 0 and j > 0:
            if seq1[i - 1] == seq2[j - 1]:
                lcs.append(seq1[i - 1])
                i -= 1
                j -= 1
            else:
                if dp[i - 1][j] >= dp[i][j - 1]:
                    i -= 1
                else:
                    j -= 1
        
        return lcs[::-1]
    
    def make_masked_labels(
        self, 
        labels: torch.Tensor, 
        real_batch_size: int
    ) -> torch.Tensor:
        """
        基于 LCS 构造 masked labels
        
        Args:
            labels: 合并的标签 [true_labels; generated_labels] (batch_size*2, seq_len)
            real_batch_size: 真实 batch size
            
        Returns:
            masked_labels (batch_size*2, seq_len)
        """
        true_labels = labels[:real_batch_size]
        generated_labels = labels[real_batch_size:]
        
        masked_true = true_labels.clone()
        masked_generated = generated_labels.clone()
        
        # 逐样本处理
        for i in range(real_batch_size):
            true_seq = true_labels[i].tolist()
            gen_seq = generated_labels[i].tolist()
            
            # 计算 LCS
            lcs = self.longest_common_subsequence(true_seq, gen_seq)
            
            if not lcs:
                continue
            
            # Mask 真实标签中的 LCS
            lcs_idx = 0
            for j in range(len(true_seq)):
                if lcs_idx < len(lcs) and true_seq[j] == lcs[lcs_idx]:
                    masked_true[i, j] = -100
                    lcs_idx += 1
            
            # Mask 生成标签中的 LCS
            lcs_idx = 0
            for j in range(len(gen_seq)):
                if lcs_idx < len(lcs) and gen_seq[j] == lcs[lcs_idx]:
                    masked_generated[i, j] = -100
                    lcs_idx += 1
        
        # 合并
        masked_labels = torch.cat([masked_true, masked_generated], dim=0)
        return masked_labels
    
    def compute_log_probs(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        average_log_prob: bool = True
    ) -> torch.Tensor:
        """
        计算 log probabilities
        
        Args:
            logits: (batch_size, seq_len, vocab_size)
            labels: (batch_size, seq_len)
            
        Returns:
            log_probs: (batch_size,)
        """
        # Shift for next token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # Log softmax
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather
        gathered_log_probs = torch.gather(
            log_probs, 
            dim=2, 
            index=shift_labels.unsqueeze(2).clamp(min=0)
        ).squeeze(2)
        
        # Mask
        mask = (shift_labels != -100) & (shift_labels != self.pad_token_id)
        gathered_log_probs = gathered_log_probs * mask.float()
        
        if average_log_prob:
            seq_log_probs = gathered_log_probs.sum(-1) / mask.sum(-1).clamp(min=1)
        else:
            seq_log_probs = gathered_log_probs.sum(-1)
        
        return seq_log_probs
    
    def spin_loss(
        self,
        policy_real_logps: torch.Tensor,
        policy_generated_logps: torch.Tensor,
        ref_real_logps: torch.Tensor,
        ref_generated_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算 SPIN loss"""
        real_rewards = policy_real_logps - policy_generated_logps.detach()
        generated_rewards = ref_real_logps - ref_generated_logps
        
        beta = self.compute_beta()
        
        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(beta * real_rewards)
            losses -= F.logsigmoid(-beta * generated_rewards)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        loss = losses.mean()
        
        metrics = {
            "difference_aware/loss": loss.item(),
            "difference_aware/beta": beta,
            "difference_aware/real_rewards": real_rewards.mean().item(),
            "difference_aware/generated_rewards": generated_rewards.mean().item(),
            "difference_aware/policy_real_logps": policy_real_logps.mean().item(),
            "difference_aware/policy_generated_logps": policy_generated_logps.mean().item(),
            "difference_aware/ref_real_logps": ref_real_logps.mean().item(),
            "difference_aware/ref_generated_logps": ref_generated_logps.mean().item(),
        }
        
        return loss, metrics