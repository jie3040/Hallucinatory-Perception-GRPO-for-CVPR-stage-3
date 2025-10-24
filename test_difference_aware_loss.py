import torch
import sys
sys.path.insert(0, '/root/autodl-tmp/EasyR1')

from verl.trainer.difference_aware_loss import DifferenceAwareLoss

def test_lcs():
    """测试 LCS 计算"""
    print("测试 LCS...")
    loss_fn = DifferenceAwareLoss()
    
    seq1 = [1, 2, 3, 4, 5]
    seq2 = [1, 3, 4, 6]
    
    lcs = loss_fn.longest_common_subsequence(seq1, seq2)
    print(f"  Seq1: {seq1}")
    print(f"  Seq2: {seq2}")
    print(f"  LCS: {lcs}")
    assert lcs == [1, 3, 4], f"Expected [1, 3, 4], got {lcs}"
    print("  ✓ 通过")

def test_masked_labels():
    """测试 Masked Labels"""
    print("\n测试 Masked Labels...")
    loss_fn = DifferenceAwareLoss()
    
    true_labels = torch.tensor([[1, 2, 3, 4, 5], [10, 11, 12, 13, 14]])
    gen_labels = torch.tensor([[1, 3, 4, 6, 7], [10, 12, 13, 15, 16]])
    
    labels = torch.cat([true_labels, gen_labels], dim=0)
    masked = loss_fn.make_masked_labels(labels, real_batch_size=2)
    
    print(f"  原始标签:\n{labels}")
    print(f"  Masked 标签:\n{masked}")
    print("  ✓ 通过")

def test_spin_loss():
    """测试 SPIN Loss"""
    print("\n测试 SPIN Loss...")
    loss_fn = DifferenceAwareLoss()
    loss_fn.update_step(50, 100)
    
    policy_real = torch.randn(4)
    policy_gen = torch.randn(4)
    ref_real = torch.randn(4)
    ref_gen = torch.randn(4)
    
    loss, metrics = loss_fn.spin_loss(
        policy_real, policy_gen, ref_real, ref_gen
    )
    
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Beta: {metrics['difference_aware/beta']:.4f}")
    print("  ✓ 通过")

if __name__ == "__main__":
    print("="*60)
    print("Testing Difference-Aware Loss Module")
    print("="*60)
    
    test_lcs()
    test_masked_labels()
    test_spin_loss()
    
    print("\n" + "="*60)
    print("所有测试通过! ✓")
    print("="*60)
