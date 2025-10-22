"""
Caption质量奖励函数
符合 EasyR1 的 reward_function 接口
"""

from typing import List, Dict, Any

def compute_score(data: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """
    计算caption质量分数
    
    Args:
        data: 数据列表，每个元素包含:
            - 'prompt': 输入prompt
            - 'response': 生成的caption
            - 'caption': 参考caption (ground truth)
            - 'images': 图像列表
            
    Returns:
        scores: 分数列表，每个元素是包含 'overall' 键的字典
    """
    scores = []
    
    for item in data:
        generated = item.get('response', '').strip()
        reference = item.get('caption', '').strip()
        
        # 基础规则评分
        score = 0.0
        
        # 1. 长度合理性 (10-100字符为最佳)
        gen_len = len(generated)
        if 10 <= gen_len <= 100:
            score += 0.3
        elif gen_len < 10:
            score -= 0.3  # 惩罚过短
        elif gen_len > 150:
            score -= 0.2  # 惩罚过长
        
        # 2. 词汇重叠 (简单的词汇匹配)
        gen_words = set(generated.lower().split())
        ref_words = set(reference.lower().split())
        
        if len(gen_words) > 0 and len(ref_words) > 0:
            # Jaccard相似度
            overlap = len(gen_words & ref_words) / len(gen_words | ref_words)
            score += overlap * 0.5
        
        # 3. 完整性检查 (是否是完整句子)
        if generated and generated[-1] in '.!?':
            score += 0.1
        
        # 4. 避免重复词 (多样性)
        words = generated.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            score += unique_ratio * 0.1
        
        # 5. 基本质量检查
        if generated:  # 非空
            score += 0.1
        if len(generated) > 5:  # 有实际内容
            score += 0.1
        
        # 归一化到 [0, 1]
        score = max(0.0, min(1.0, score))
        
        # 重要：返回字典格式
        scores.append({"overall": score})
    
    return scores
