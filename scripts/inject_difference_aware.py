"""
注入 Difference-Aware Worker 到 EasyR1
"""

import os
import shutil
from datetime import datetime

def backup_file(filepath):
    """备份文件"""
    if not os.path.exists(filepath):
        print(f"⚠ 文件不存在: {filepath}")
        return
    
    backup_path = f"{filepath}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filepath, backup_path)
    print(f"✓ 备份: {backup_path}")

def inject_imports():
    """在 ray_trainer.py 中添加 import"""
    trainer_path = "/root/autodl-tmp/EasyR1/verl/trainer/ray_trainer.py"
    
    with open(trainer_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if "DifferenceAwareFSDPWorker" in content:
        print("✓ Import 已存在，跳过")
        return content
    
    # 找到 FSDPWorker 的 import
    import_pos = content.find("from verl.workers.fsdp_workers import FSDPWorker")
    
    if import_pos == -1:
        print("⚠ 找不到 FSDPWorker import")
        return content
    
    # 在下一行添加
    line_end = content.find("\n", import_pos)
    new_import = "\nfrom verl.workers.fsdp_workers_diff_aware import DifferenceAwareFSDPWorker"
    
    content = content[:line_end] + new_import + content[line_end:]
    print("✓ 添加 DifferenceAwareFSDPWorker import")
    
    return content

def inject_worker_selection(content):
    """在创建 worker 时根据配置选择类"""
    
    # 这里需要修改 _create_workers 或类似方法
    # 由于 EasyR1 的结构，我们采用更简单的方式：
    # 直接替换 FSDPWorker 的使用
    
    # 方案：在 main.py 中根据配置选择
    print("✓ Worker 选择逻辑将在启动脚本中处理")
    
    return content

def apply_injection():
    """应用注入"""
    
    print("="*70)
    print("开始注入 Difference-Aware Worker")
    print("="*70)
    
    trainer_path = "/root/autodl-tmp/EasyR1/verl/trainer/ray_trainer.py"
    
    # 备份
    backup_file(trainer_path)
    
    # 注入 imports
    content = inject_imports()
    
    # 注入 worker 选择
    content = inject_worker_selection(content)
    
    # 写回
    with open(trainer_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ 注入完成")

def create_patched_main():
    """
    创建一个修改版的 main.py，使用 DifferenceAwareFSDPWorker
    """
    
    main_patch = """
# 在 main.py 中，根据配置选择 Worker 类

# 原始代码可能类似：
# from verl.workers.fsdp_workers import FSDPWorker

# 修改为：
from verl.workers.fsdp_workers import FSDPWorker
from verl.workers.fsdp_workers_diff_aware import DifferenceAwareFSDPWorker

# 在创建 worker 时：
if ppo_config.algorithm.get('use_difference_aware', False):
    worker_cls = DifferenceAwareFSDPWorker
    print("[Main] 使用 DifferenceAwareFSDPWorker")
else:
    worker_cls = FSDPWorker
    print("[Main] 使用 FSDPWorker")
"""
    
    print("\n注意：需要手动修改 verl/trainer/main.py")
    print("在创建 worker 的地方，根据配置选择 Worker 类")
    print(main_patch)

if __name__ == "__main__":
    apply_injection()
    create_patched_main()
    
    print("\n" + "="*70)
    print("注入完成！")
    print("\n下一步：创建启动脚本")
    print("="*70)
