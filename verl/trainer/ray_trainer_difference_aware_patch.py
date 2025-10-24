# /root/autodl-tmp/EasyR1/verl/trainer/ray_trainer_difference_aware_patch.py

"""
为 Ray Trainer 应用完整的 Difference-Aware 补丁
"""

def create_difference_aware_trainer():
    """
    创建一个支持 Difference-Aware Loss 的训练器类
    """
    
    code = '''
# 在 ray_trainer.py 中添加

class DifferenceAwareRayPPOTrainer(RayPPOTrainer):
    """
    支持 Difference-Aware Loss 的 Ray PPO Trainer
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # 检查是否启用 Difference-Aware
        self.use_difference_aware = config.algorithm.get('use_difference_aware', False)
        
        if self.use_difference_aware:
            print("[Trainer] 启用 Difference-Aware Loss")
            from verl.trainer.difference_aware_loss import DifferenceAwareLoss
            
            self.difference_aware_loss = DifferenceAwareLoss(
                min_beta=config.algorithm.get('min_beta', 0.1),
                max_beta=config.algorithm.get('max_beta', 0.5),
                loss_type=config.algorithm.get('loss_type', 'sigmoid'),
            )
    
    def _training_step(self, experience_batch):
        """
        训练步骤（覆盖）
        """
        if not self.use_difference_aware:
            return super()._training_step(experience_batch)
        
        # 1. 生成参考序列
        ref_batch = self._generate_ref_sequences(experience_batch)
        
        # 2. 合并到 experience_batch
        experience_batch['ref_input_ids'] = ref_batch['ref_input_ids']
        experience_batch['ref_attention_mask'] = ref_batch['ref_attention_mask']
        experience_batch['ref_labels'] = ref_batch['ref_labels']
        
        # 3. 计算 Difference-Aware Loss
        loss_output = self.actor_rollout_ref_wg.compute_loss(experience_batch)
        
        # 4. 反向传播和优化
        self._backward_and_optimize(loss_output['loss'])
        
        # 5. 记录指标
        self._log_metrics(loss_output['metrics'])
        
        return loss_output
    
    def _generate_ref_sequences(self, batch):
        """
        使用参考模型生成序列
        """
        return self.actor_rollout_ref_wg.generate_ref_sequences(batch)
    
    def _backward_and_optimize(self, loss):
        """
        反向传播和优化
        """
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.config.worker.actor.get('max_grad_norm', None):
            torch.nn.utils.clip_grad_norm_(
                self.actor_model.parameters(),
                self.config.worker.actor.max_grad_norm
            )
        
        self.optimizer.step()
    
    def _log_metrics(self, metrics):
        """
        记录指标
        """
        for key, value in metrics.items():
            self.logger.log({key: value})
'''
    
    return code


def apply_full_patch():
    """
    应用完整补丁
    """
    import os
    import shutil
    from datetime import datetime
    
    print("="*80)
    print("开始应用 Difference-Aware Loss 完整补丁")
    print("="*80)
    
    # 1. 备份原文件
    files_to_backup = [
        "/root/autodl-tmp/EasyR1/verl/trainer/ray_trainer.py",
        "/root/autodl-tmp/EasyR1/verl/workers/fsdp_workers.py",
    ]
    
    for filepath in files_to_backup:
        if os.path.exists(filepath):
            backup_path = f"{filepath}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(filepath, backup_path)
            print(f"✓ 备份: {backup_path}")
    
    # 2. 创建新文件
    print("\n创建新模块...")
    
    # difference_aware_loss.py 已经创建
    print("✓ difference_aware_loss.py")
    
    # 创建扩展的 worker
    worker_code = open('/root/autodl-tmp/EasyR1/verl/workers/fsdp_workers_difference_aware.py', 'w')
    # （写入上面的 DifferenceAwareFSDPWorker 代码）
    worker_code.close()
    print("✓ fsdp_workers_difference_aware.py")
    
    # 3. 修改 ray_trainer.py
    print("\n修改 ray_trainer.py...")
    
    trainer_path = "/root/autodl-tmp/EasyR1/verl/trainer/ray_trainer.py"
    
    with open(trainer_path, 'r') as f:
        content = f.read()
    
    # 在文件末尾添加新的 Trainer 类
    trainer_code = create_difference_aware_trainer()
    
    with open(trainer_path, 'a') as f:
        f.write("\n\n")
        f.write("# " + "="*70 + "\n")
        f.write("# Difference-Aware Loss Support\n")
        f.write("# " + "="*70 + "\n")
        f.write(trainer_code)
    
    print("✓ 添加 DifferenceAwareRayPPOTrainer")
    
    print("\n" + "="*80)
    print("补丁应用完成！")
    print("="*80)
    print("\n使用方法:")
    print("1. 在配置文件中设置: algorithm.use_difference_aware = True")
    print("2. 使用 DifferenceAwareRayPPOTrainer 而不是 RayPPOTrainer")
    print()

if __name__ == "__main__":
    apply_full_patch()