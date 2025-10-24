#!/usr/bin/env python
"""第三阶段：Difference-Aware Loss - 最终版本"""

import sys
import os

sys.path.insert(0, '/root/autodl-tmp/EasyR1')
os.chdir('/root/autodl-tmp/EasyR1')

print("[Stage 3] 注入 Difference-Aware Loss...")

def patch_worker():
    import torch
    from verl.trainer.difference_aware_loss import DifferenceAwareLoss
    from verl.workers import fsdp_workers
    
    _original = fsdp_workers.FSDPWorker.update_actor
    _cache = {}
    
    def update_actor_diff_aware(self, data):
        wid = id(self)
        if wid not in _cache:
            pad_id = 151643
            if hasattr(self, 'processor') and self.processor:
                if hasattr(self.processor, 'tokenizer'):
                    pad_id = self.processor.tokenizer.pad_token_id
            _cache[wid] = DifferenceAwareLoss(0.1, 0.5, 'sigmoid', pad_id)
            print(f"[DiffAware] ✓ 初始化")
        
        loss_fn = _cache[wid]
        try:
            return _update_diff_aware(self, data, loss_fn)
        except Exception as e:
            print(f"[DiffAware] ⚠ {e}, 降级")
            return _original(self, data)
    
    def _update_diff_aware(w, data, fn):
        fn.update_step(data.batch.get('global_step', 0) if hasattr(data, 'batch') else 0, 1000)
        batch = data.batch if hasattr(data, 'batch') else data
        inp, mask = batch['input_ids'], batch['attention_mask']
        labels = inp.clone()
        labels[:, :inp.shape[1]//2] = -100
        
        with torch.no_grad():
            ref_out = w.ref_model.generate(inp, mask, max_new_tokens=256, temperature=0.7, 
                                          top_p=0.9, do_sample=True, pad_token_id=fn.pad_token_id)
        
        plen = inp.shape[1]
        ref_lab = ref_out.clone()
        ref_lab[:, :plen] = -100
        ref_msk = (ref_out != fn.pad_token_id).long()
        bs = inp.shape[0]
        
        mlen = max(inp.shape[1], ref_out.shape[1])
        def pad(t, l, v):
            if t.shape[1] >= l: return t
            return torch.cat([t, torch.full((t.shape[0], l-t.shape[1]), v, dtype=t.dtype, device=t.device)], 1)
        
        inp = pad(inp, mlen, fn.pad_token_id)
        ref_out = pad(ref_out, mlen, fn.pad_token_id)
        mask = pad(mask, mlen, 0)
        ref_msk = pad(ref_msk, mlen, 0)
        labels = pad(labels, mlen, -100)
        ref_lab = pad(ref_lab, mlen, -100)
        
        comb_inp = torch.cat([inp, ref_out])
        comb_mask = torch.cat([mask, ref_msk])
        comb_lab = torch.cat([labels, ref_lab])
        
        m_lab = fn.make_masked_labels(comb_lab, bs)
        
        p_out = w.actor_model(comb_inp, comb_mask)
        with torch.no_grad():
            r_out = w.ref_model(comb_inp, comb_mask)
        
        p_lp = fn.compute_log_probs(p_out.logits, m_lab)
        r_lp = fn.compute_log_probs(r_out.logits, m_lab)
        
        loss, metrics = fn.spin_loss(p_lp[:bs], p_lp[bs:], r_lp[:bs], r_lp[bs:])
        
        w.optimizer.zero_grad()
        loss.backward()
        if hasattr(w.config.worker.actor, 'max_grad_norm'):
            mn = w.config.worker.actor.max_grad_norm
            if mn: torch.nn.utils.clip_grad_norm_(w.actor_model.parameters(), mn)
        w.optimizer.step()
        
        return {'loss': loss.item(), **metrics}
    
    fsdp_workers.FSDPWorker.update_actor = update_actor_diff_aware
    print("[Stage 3] ✓ Patch 完成")

patch_worker()

sys.argv = ['train', 'config=examples/caption_grpo_config_stage3.yaml']

from verl.trainer.main import main
main()
