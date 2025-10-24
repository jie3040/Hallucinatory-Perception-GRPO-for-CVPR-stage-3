#!/usr/bin/env python
"""
Difference-Aware Loss 训练启动脚本
"""

import sys
import os

# !!! 关键：禁用 torch.compile
os.environ['VLLM_TORCH_COMPILE_LEVEL'] = '0'
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'  # 使用 xformers 而不是 FlashAttention

sys.path.insert(0, '/root/autodl-tmp/EasyR1')
os.chdir('/root/autodl-tmp/EasyR1')

print("[Launcher] 准备注入 Difference-Aware Loss...")
print("[Launcher] 已禁用 torch.compile 以解决兼容性问题")

def patch_fsdp_worker():
    """Patch FSDPWorker.update_actor 方法"""
    
    import torch
    from verl.trainer.difference_aware_loss import DifferenceAwareLoss
    from verl.workers import fsdp_workers
    
    _original_update_actor = fsdp_workers.FSDPWorker.update_actor
    _diff_aware_loss_cache = {}
    
    def update_actor_with_difference_aware(self, data):
        """替换的 update_actor 方法"""
        worker_id = id(self)
        
        if worker_id not in _diff_aware_loss_cache:
            print("\n" + "="*70)
            print(f"[DiffAware Worker {worker_id}] 初始化 Difference-Aware Loss")
            print("="*70)
            
            pad_token_id = 151643
            if hasattr(self, 'processor') and self.processor is not None:
                if hasattr(self.processor, 'tokenizer'):
                    pad_token_id = self.processor.tokenizer.pad_token_id
            
            _diff_aware_loss_cache[worker_id] = DifferenceAwareLoss(
                min_beta=0.1,
                max_beta=0.5,
                loss_type='sigmoid',
                pad_token_id=pad_token_id,
            )
            
            print(f"[DiffAware] 配置: β=[0.1, 0.5], pad_token_id={pad_token_id}")
            print("="*70 + "\n")
        
        diff_aware_loss = _diff_aware_loss_cache[worker_id]
        
        print("\n[DiffAware] >>> 开始更新")
        
        try:
            result = _update_with_difference_aware(self, data, diff_aware_loss)
            print("[DiffAware] <<< 更新完成\n")
            return result
        except Exception as e:
            print(f"[DiffAware] ⚠ 失败: {e}")
            import traceback
            traceback.print_exc()
            print("[DiffAware] 降级到原始 update_actor")
            return _original_update_actor(self, data)
    
    def _update_with_difference_aware(worker, data, diff_aware_loss):
        """Difference-Aware 更新逻辑"""
        
        current_step = data.batch.get('global_step', 0) if hasattr(data, 'batch') else 0
        max_steps = 1000
        diff_aware_loss.update_step(current_step, max_steps)
        
        if hasattr(data, 'batch'):
            batch = data.batch
        else:
            batch = data
        
        print(f"[DiffAware] Batch keys: {list(batch.keys())}")
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        print(f"[DiffAware] 输入: batch={input_ids.shape[0]}, len={input_ids.shape[1]}")
        
        labels = input_ids.clone()
        if 'prompt_mask' in batch:
            labels[~batch['prompt_mask']] = -100
        else:
            prompt_len = input_ids.shape[1] // 2
            labels[:, :prompt_len] = -100
        
        print("[DiffAware] (1/7) 生成 ref 序列...")
        with torch.no_grad():
            ref_outputs = worker.ref_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=diff_aware_loss.pad_token_id,
            )
        
        prompt_len = input_ids.shape[1]
        ref_labels = ref_outputs.clone()
        ref_labels[:, :prompt_len] = -100
        ref_mask = (ref_outputs != diff_aware_loss.pad_token_id).long()
        
        real_batch_size = input_ids.shape[0]
        
        print("[DiffAware] (2/7) Padding...")
        max_len = max(input_ids.shape[1], ref_outputs.shape[1])
        
        def pad_tensor(tensor, target_len, pad_value):
            if tensor.shape[1] >= target_len:
                return tensor
            padding = torch.full(
                (tensor.shape[0], target_len - tensor.shape[1]),
                pad_value, dtype=tensor.dtype, device=tensor.device
            )
            return torch.cat([tensor, padding], dim=1)
        
        input_ids = pad_tensor(input_ids, max_len, diff_aware_loss.pad_token_id)
        ref_outputs = pad_tensor(ref_outputs, max_len, diff_aware_loss.pad_token_id)
        attention_mask = pad_tensor(attention_mask, max_len, 0)
        ref_mask = pad_tensor(ref_mask, max_len, 0)
        labels = pad_tensor(labels, max_len, -100)
        ref_labels = pad_tensor(ref_labels, max_len, -100)
        
        print("[DiffAware] (3/7) 合并...")
        combined_input_ids = torch.cat([input_ids, ref_outputs], dim=0)
        combined_attention_mask = torch.cat([attention_mask, ref_mask], dim=0)
        combined_labels = torch.cat([labels, ref_labels], dim=0)
        
        print("[DiffAware] (4/7) LCS masking...")
        masked_labels = diff_aware_loss.make_masked_labels(
            combined_labels, real_batch_size
        )
        
        valid = (masked_labels != -100).sum().item()
        total = masked_labels.numel()
        print(f"[DiffAware]   有效: {valid}/{total} ({100*valid/total:.1f}%)")
        
        print("[DiffAware] (5/7) Forward...")
        policy_outputs = worker.actor_model(
            input_ids=combined_input_ids,
            attention_mask=combined_attention_mask,
        )
        
        with torch.no_grad():
            ref_model_outputs = worker.ref_model(
                input_ids=combined_input_ids,
                attention_mask=combined_attention_mask,
            )
        
        print("[DiffAware] (6/7) Log probs...")
        policy_log_probs = diff_aware_loss.compute_log_probs(
            policy_outputs.logits, masked_labels
        )
        ref_log_probs = diff_aware_loss.compute_log_probs(
            ref_model_outputs.logits, masked_labels
        )
        
        policy_real = policy_log_probs[:real_batch_size]
        policy_gen = policy_log_probs[real_batch_size:]
        ref_real = ref_log_probs[:real_batch_size]
        ref_gen = ref_log_probs[real_batch_size:]
        
        print("[DiffAware] (7/7) SPIN loss...")
        loss, metrics = diff_aware_loss.spin_loss(
            policy_real, policy_gen, ref_real, ref_gen
        )
        
        print(f"[DiffAware] Loss={loss.item():.4f}, β={metrics['difference_aware/beta']:.3f}")
        
        worker.optimizer.zero_grad()
        loss.backward()
        
        if hasattr(worker.config.worker.actor, 'max_grad_norm'):
            max_norm = worker.config.worker.actor.max_grad_norm
            if max_norm:
                torch.nn.utils.clip_grad_norm_(
                    worker.actor_model.parameters(), max_norm
                )
        
        worker.optimizer.step()
        
        return {
            'loss': loss.item(),
            **metrics,
        }
    
    fsdp_workers.FSDPWorker.update_actor = update_actor_with_difference_aware
    
    print("[Launcher] ✓ FSDPWorker.update_actor 已替换为 Difference-Aware 版本")

patch_fsdp_worker()

sys.argv = [
    'train',
    'config=examples/caption_grpo_config_difference_aware.yaml',
]

from verl.trainer.main import main

if __name__ == "__main__":
    main()
