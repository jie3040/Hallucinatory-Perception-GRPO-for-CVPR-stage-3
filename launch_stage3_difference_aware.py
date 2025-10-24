#!/usr/bin/env python
"""第三阶段：Difference-Aware Loss - 禁用 torch.compile"""

# !!! 必须在任何 import 之前设置
import os
os.environ['PYTORCH_JIT'] = '0'
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['VLLM_TORCH_COMPILE_LEVEL'] = '0'

# 禁用 torch.compile 的函数
import torch
_original_compile = torch.compile
def _dummy_compile(fn, *args, **kwargs):
    return fn
torch.compile = _dummy_compile

import sys
sys.path.insert(0, '/root/autodl-tmp/EasyR1')
os.chdir('/root/autodl-tmp/EasyR1')

print("[Stage 3] ✓ Torch compile 已禁用")
print("[Stage 3] 注入 Difference-Aware Loss...")

def patch_fsdp_worker():
    from verl.trainer.difference_aware_loss import DifferenceAwareLoss
    from verl.workers import fsdp_workers
    
    _original_update_actor = fsdp_workers.FSDPWorker.update_actor
    _diff_aware_loss_cache = {}
    
    def update_actor_with_difference_aware(self, data):
        worker_id = id(self)
        
        if worker_id not in _diff_aware_loss_cache:
            pad_token_id = 151643
            if hasattr(self, 'processor') and self.processor is not None:
                if hasattr(self.processor, 'tokenizer'):
                    pad_token_id = self.processor.tokenizer.pad_token_id
            
            _diff_aware_loss_cache[worker_id] = DifferenceAwareLoss(
                min_beta=0.1, max_beta=0.5, loss_type='sigmoid', pad_token_id=pad_token_id
            )
            print(f"[DiffAware] ✓ 初始化 (β=[0.1,0.5])")
        
        diff_aware_loss = _diff_aware_loss_cache[worker_id]
        
        try:
            return _update_with_difference_aware(self, data, diff_aware_loss)
        except Exception as e:
            print(f"[DiffAware] ⚠ {e}, 降级")
            return _original_update_actor(self, data)
    
    def _update_with_difference_aware(worker, data, loss_fn):
        current_step = data.batch.get('global_step', 0) if hasattr(data, 'batch') else 0
        loss_fn.update_step(current_step, 1000)
        
        batch = data.batch if hasattr(data, 'batch') else data
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = input_ids.clone()
        labels[:, :input_ids.shape[1]//2] = -100
        
        with torch.no_grad():
            ref_outputs = worker.ref_model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=256, temperature=0.7, top_p=0.9,
                do_sample=True, pad_token_id=loss_fn.pad_token_id
            )
        
        prompt_len = input_ids.shape[1]
        ref_labels = ref_outputs.clone()
        ref_labels[:, :prompt_len] = -100
        ref_mask = (ref_outputs != loss_fn.pad_token_id).long()
        real_batch_size = input_ids.shape[0]
        
        max_len = max(input_ids.shape[1], ref_outputs.shape[1])
        
        def pad(t, l, v):
            if t.shape[1] >= l: return t
            return torch.cat([t, torch.full((t.shape[0], l-t.shape[1]), v, dtype=t.dtype, device=t.device)], dim=1)
        
        input_ids = pad(input_ids, max_len, loss_fn.pad_token_id)
        ref_outputs = pad(ref_outputs, max_len, loss_fn.pad_token_id)
        attention_mask = pad(attention_mask, max_len, 0)
        ref_mask = pad(ref_mask, max_len, 0)
        labels = pad(labels, max_len, -100)
        ref_labels = pad(ref_labels, max_len, -100)
        
        combined_input_ids = torch.cat([input_ids, ref_outputs], dim=0)
        combined_attention_mask = torch.cat([attention_mask, ref_mask], dim=0)
        combined_labels = torch.cat([labels, ref_labels], dim=0)
        
        masked_labels = loss_fn.make_masked_labels(combined_labels, real_batch_size)
        
        policy_outputs = worker.actor_model(input_ids=combined_input_ids, attention_mask=combined_attention_mask)
        with torch.no_grad():
            ref_model_outputs = worker.ref_model(input_ids=combined_input_ids, attention_mask=combined_attention_mask)
        
        policy_log_probs = loss_fn.compute_log_probs(policy_outputs.logits, masked_labels)
        ref_log_probs = loss_fn.compute_log_probs(ref_model_outputs.logits, masked_labels)
        
        loss, metrics = loss_fn.spin_loss(
            policy_log_probs[:real_batch_size], policy_log_probs[real_batch_size:],
            ref_log_probs[:real_batch_size], ref_log_probs[real_batch_size:]
        )
        
        worker.optimizer.zero_grad()
        loss.backward()
        
        if hasattr(worker.config.worker.actor, 'max_grad_norm'):
            max_norm = worker.config.worker.actor.max_grad_norm
            if max_norm:
                torch.nn.utils.clip_grad_norm_(worker.actor_model.parameters(), max_norm)
        
        worker.optimizer.step()
        return {'loss': loss.item(), **metrics}
    
    fsdp_workers.FSDPWorker.update_actor = update_actor_with_difference_aware
    print("[Stage 3] ✓ Patch 完成")

patch_fsdp_worker()

sys.argv = ['train', 'config=examples/caption_grpo_config_stage3_difference_aware.yaml']

from verl.trainer.main import main
main()
