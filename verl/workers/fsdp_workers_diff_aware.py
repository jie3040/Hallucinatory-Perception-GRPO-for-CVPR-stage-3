"""
扩展 FSDP Worker 支持 Difference-Aware Loss
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from verl.workers.fsdp_workers import FSDPWorker
from verl.trainer.difference_aware_loss import DifferenceAwareLoss

class DifferenceAwareFSDPWorker(FSDPWorker):
    """
    集成 Difference-Aware Loss 的 FSDP Worker
    
    完全兼容 FSDPWorker 的接口
    """
    
    def __init__(self, config, role: str, *args, **kwargs):
        """
        初始化 Worker
        
        Args:
            config: 配置对象
            role: worker 角色（'actor_rollout_ref', 'critic', 等）
        """
        # 调用父类初始化
        super().__init__(config, role, *args, **kwargs)
        
        # 自动启用 Difference-Aware Loss
        self.use_difference_aware = True
        
        print(f"[DifferenceAwareWorker-{role}] ✓ 启用 Difference-Aware Loss")
        
        # 获取 pad_token_id
        pad_token_id = 151643  # Qwen2 默认
        if hasattr(self, 'processor') and self.processor is not None:
            if hasattr(self.processor, 'tokenizer'):
                pad_token_id = self.processor.tokenizer.pad_token_id
        elif hasattr(self, 'tokenizer') and self.tokenizer is not None:
            pad_token_id = self.tokenizer.pad_token_id
        
        # 初始化 Difference-Aware Loss
        min_beta = 0.1
        max_beta = 0.5
        loss_type = 'sigmoid'
        
        self.diff_aware_loss = DifferenceAwareLoss(
            min_beta=min_beta,
            max_beta=max_beta,
            loss_type=loss_type,
            pad_token_id=pad_token_id,
        )
        
        print(f"[DifferenceAwareWorker-{role}] Loss配置: β=[{min_beta},{max_beta}], type={loss_type}, pad={pad_token_id}")
    
    def update_policy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新策略 - 使用 Difference-Aware Loss
        
        覆盖父类方法，使用我们自定义的 loss
        """
        return self._update_policy_with_difference_aware(data)
    
    def _update_policy_with_difference_aware(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用 Difference-Aware Loss 更新策略
        
        完整流程：
        1. 使用 ref model 生成序列
        2. 合并 real 和 generated
        3. 计算 LCS 并 mask
        4. 计算 SPIN loss
        5. 反向传播
        """
        print("\n[DiffAware] ========== 开始更新 ==========")
        
        # 更新步数
        current_step = data.get('global_step', 0)
        max_steps = data.get('max_steps', 1000)
        self.diff_aware_loss.update_step(current_step, max_steps)
        
        # 提取数据
        input_ids = data['input_ids']  # (batch_size, seq_len)
        attention_mask = data['attention_mask']
        
        print(f"[DiffAware] 输入: batch={input_ids.shape[0]}, seq_len={input_ids.shape[1]}")
        
        # 构造 labels
        labels = input_ids.clone()
        if 'prompt_length' in data:
            for i, prompt_len in enumerate(data['prompt_length']):
                labels[i, :prompt_len] = -100
        else:
            # 假设前一半是 prompt（保守估计）
            prompt_len = input_ids.shape[1] // 2
            labels[:, :prompt_len] = -100
        
        # 1. 生成 ref 序列
        print("[DiffAware] Step 1: 生成 ref 序列...")
        try:
            ref_batch = self._generate_ref_sequences(input_ids, attention_mask, data)
            print(f"[DiffAware] Ref 序列生成成功: {ref_batch['input_ids'].shape}")
        except Exception as e:
            print(f"[DiffAware] ⚠ Ref 生成失败: {e}")
            # 降级到原始 loss
            return super().update_policy(data)
        
        real_batch_size = input_ids.shape[0]
        
        # 2. Pad to same length
        max_len = max(input_ids.shape[1], ref_batch['input_ids'].shape[1])
        print(f"[DiffAware] Step 2: Padding to max_len={max_len}")
        
        input_ids = self._pad_tensor(input_ids, max_len, self.diff_aware_loss.pad_token_id)
        ref_input_ids = self._pad_tensor(ref_batch['input_ids'], max_len, self.diff_aware_loss.pad_token_id)
        attention_mask = self._pad_tensor(attention_mask, max_len, 0)
        ref_attention_mask = self._pad_tensor(ref_batch['attention_mask'], max_len, 0)
        labels = self._pad_tensor(labels, max_len, -100)
        ref_labels = self._pad_tensor(ref_batch['labels'], max_len, -100)
        
        # 3. 合并
        combined_input_ids = torch.cat([input_ids, ref_input_ids], dim=0)
        combined_attention_mask = torch.cat([attention_mask, ref_attention_mask], dim=0)
        combined_labels = torch.cat([labels, ref_labels], dim=0)
        
        print(f"[DiffAware] Step 3: 合并后 batch={combined_input_ids.shape[0]}")
        
        # 4. LCS mask
        print("[DiffAware] Step 4: 计算 LCS 并 mask...")
        try:
            masked_labels = self.diff_aware_loss.make_masked_labels(
                combined_labels, real_batch_size
            )
            
            valid = (masked_labels != -100).sum().item()
            total = masked_labels.numel()
            print(f"[DiffAware] LCS masking: {valid}/{total} ({100*valid/total:.1f}%) 有效 tokens")
        except Exception as e:
            print(f"[DiffAware] ⚠ LCS 失败: {e}")
            masked_labels = combined_labels
        
        # 5. Forward pass
        print("[DiffAware] Step 5: Forward pass...")
        try:
            policy_outputs = self.actor_model(
                input_ids=combined_input_ids,
                attention_mask=combined_attention_mask,
            )
            
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=combined_input_ids,
                    attention_mask=combined_attention_mask,
                )
        except Exception as e:
            print(f"[DiffAware] ⚠ Forward 失败: {e}")
            raise
        
        # 6. 计算 log probs
        print("[DiffAware] Step 6: 计算 log probs...")
        policy_log_probs = self.diff_aware_loss.compute_log_probs(
            policy_outputs.logits, masked_labels
        )
        ref_log_probs = self.diff_aware_loss.compute_log_probs(
            ref_outputs.logits, masked_labels
        )
        
        # 7. 分离
        policy_real = policy_log_probs[:real_batch_size]
        policy_gen = policy_log_probs[real_batch_size:]
        ref_real = ref_log_probs[:real_batch_size]
        ref_gen = ref_log_probs[real_batch_size:]
        
        # 8. SPIN loss
        print("[DiffAware] Step 7: 计算 SPIN loss...")
        loss, metrics = self.diff_aware_loss.spin_loss(
            policy_real, policy_gen, ref_real, ref_gen
        )
        
        print(f"[DiffAware] Loss={loss.item():.4f}, β={metrics['difference_aware/beta']:.3f}")
        print(f"[DiffAware]   Real reward={metrics['difference_aware/real_rewards']:.3f}")
        print(f"[DiffAware]   Gen reward={metrics['difference_aware/generated_rewards']:.3f}")
        
        # 9. 反向传播
        print("[DiffAware] Step 8: 反向传播...")
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if hasattr(self.config.worker.actor, 'max_grad_norm'):
            max_norm = self.config.worker.actor.max_grad_norm
            if max_norm and max_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor_model.parameters(), max_norm
                )
                print(f"[DiffAware] Grad norm: {grad_norm:.3f} (clipped to {max_norm})")
        
        self.optimizer.step()
        
        print("[DiffAware] ========== 更新完成 ==========\n")
        
        return {'loss': loss.item(), **metrics}
    
    def _generate_ref_sequences(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        data: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        使用参考模型生成序列
        """
        with torch.no_grad():
            outputs = self.ref_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.diff_aware_loss.pad_token_id,
            )
        
        # 构造 labels（只对新生成的部分计算 loss）
        prompt_len = input_ids.shape[1]
        labels = outputs.clone()
        labels[:, :prompt_len] = -100
        
        # Attention mask
        gen_mask = (outputs != self.diff_aware_loss.pad_token_id).long()
        
        return {
            'input_ids': outputs,
            'attention_mask': gen_mask,
            'labels': labels,
        }
    
    def _pad_tensor(
        self, 
        tensor: torch.Tensor, 
        target_len: int, 
        pad_value: int
    ) -> torch.Tensor:
        """Pad tensor to target length"""
        if tensor.shape[1] >= target_len:
            return tensor
        
        padding = torch.full(
            (tensor.shape[0], target_len - tensor.shape[1]),
            pad_value,
            dtype=tensor.dtype,
            device=tensor.device
        )
        return torch.cat([tensor, padding], dim=1)
