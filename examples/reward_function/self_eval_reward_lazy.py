# /root/EasyR1/examples/reward_function/self_eval_reward_lazy.py

import torch
from typing import List, Dict, Any
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import re
import gc
import os

def get_best_available_model_path():
    """
    智能选择最佳可用模型：
    1. 优先使用第一阶段训练好的最新检查点
    2. 如果没有，使用原始模型
    """
    # 第一阶段检查点目录
    checkpoint_dir = "/root/autodl-tmp/EasyR1/checkpoints/easy_r1/flickr30k_caption_grpo_2x3090"
    
    if os.path.exists(checkpoint_dir):
        epochs = [d for d in os.listdir(checkpoint_dir) if d.startswith('epoch_')]
        if epochs:
            # 找最新的 epoch
            latest_epoch = sorted(epochs, key=lambda x: int(x.split('_')[1]))[-1]
            model_path = os.path.join(checkpoint_dir, latest_epoch, 'actor')
            
            # 验证模型文件存在
            if os.path.exists(os.path.join(model_path, 'config.json')):
                print(f"[奖励函数] ✓ 使用第一阶段训练好的模型: {model_path}")
                return model_path
            else:
                print(f"[奖励函数] ⚠ 检查点目录存在但模型文件不完整: {model_path}")
    
    # 回退到原始模型
    original_model = "/root/autodl-tmp/models/Qwen/Qwen2-VL-2B-Instruct"
    print(f"[奖励函数] ✓ 使用原始模型: {original_model}")
    return original_model


def batch_score_with_lazy_model(model_path: str, items: List[Dict]) -> List[float]:
    """
    批量加载模型评分，用完立即释放
    
    Args:
        model_path: 模型路径
        items: 包含 image 和 caption 的列表
        
    Returns:
        分数列表 (0-1)
    """
    if not items:
        return []
    
    try:
        print(f"[奖励函数] 临时加载评分模型进行批量评分 ({len(items)} 个样本)...")
        print(f"[奖励函数] 模型路径: {model_path}")
        
        # 临时加载模型到双卡
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",  # 自动分配到两张卡
            trust_remote_code=True
        )
        
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        model.eval()
        
        scores = []
        
        # 批量处理（每次处理4个）
        batch_size = 4
        for i in range(0, len(items), batch_size):
            batch_items = items[i:i+batch_size]
            
            # 准备批量输入
            texts = []
            images = []
            
            for item in batch_items:
                caption = item['caption']
                image = item['image']
                
                scoring_prompt = f"""<image>Rate this caption from 0-10.

Caption: {caption}

Rating:"""
                
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": scoring_prompt}
                    ]
                }]
                
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                texts.append(text)
                images.append(image)
            
            # 批量推理
            inputs = processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True
            ).to("cuda:0")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    temperature=0.1,
                    do_sample=False
                )
            
            # 解析每个样本的分数
            for j, output in enumerate(outputs):
                response = processor.decode(
                    output[inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                score = extract_score(response)
                scores.append(score / 10.0)
            
            # 清理中间结果
            del inputs
            del outputs
            torch.cuda.empty_cache()
        
        # 释放模型
        print(f"[奖励函数] 评分完成，释放模型显存...")
        del model
        del processor
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"[奖励函数] 模型已释放，显存已清理")
        
        return scores
        
    except Exception as e:
        print(f"[奖励函数] 批量评分失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 失败时返回默认分数
        return [0.5] * len(items)


def extract_score(text: str) -> float:
    """提取 0-10 的分数"""
    matches = re.findall(r'\b([0-9]|10)(?:\.\d+)?\b', text)
    if matches:
        try:
            score = float(matches[0])
            return max(0.0, min(10.0, score))
        except:
            pass
    return 5.0


def compute_score(data: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """
    混合奖励：规则 (40%) + 模型评分 (60%)
    使用延迟加载策略
    """
    # 智能选择最佳可用模型
    model_path = get_best_available_model_path()
    
    # 计算规则分数
    rule_scores = []
    model_items = []  # 需要模型评分的项
    
    for item in data:
        generated = item.get('response', '').strip()
        reference = item.get('caption', '').strip()
        images = item.get('images', [])
        
        # 规则分
        rule_score = compute_rule_score(generated, reference)
        rule_scores.append(rule_score)
        
        # 准备模型评分项
        if images and len(images) > 0:
            model_items.append({
                'caption': generated,
                'image': images[0]
            })
        else:
            model_items.append(None)
    
    # 批量模型评分
    valid_items = [item for item in model_items if item is not None]
    
    if valid_items:
        model_scores_valid = batch_score_with_lazy_model(model_path, valid_items)
        
        # 重新分配分数
        model_scores = []
        valid_idx = 0
        for item in model_items:
            if item is not None:
                model_scores.append(model_scores_valid[valid_idx])
                valid_idx += 1
            else:
                model_scores.append(rule_scores[len(model_scores)])
    else:
        model_scores = rule_scores
    
    # 混合分数
    scores = []
    for i in range(len(data)):
        final_score = 0.4 * rule_scores[i] + 0.6 * model_scores[i]
        
        scores.append({
            "overall": final_score,
            "rule_based": rule_scores[i],
            "model_based": model_scores[i]
        })
    
    return scores


def compute_rule_score(generated: str, reference: str) -> float:
    """规则评分"""
    score = 0.0
    gen_len = len(generated)
    
    if 10 <= gen_len <= 100:
        score += 0.3
    elif gen_len < 10:
        score -= 0.3
    
    gen_words = set(generated.lower().split())
    ref_words = set(reference.lower().split())
    
    if len(gen_words) > 0 and len(ref_words) > 0:
        overlap = len(gen_words & ref_words) / len(gen_words | ref_words)
        score += overlap * 0.5
    
    if generated and generated[-1] in '.!?':
        score += 0.1
    
    words = generated.lower().split()
    if len(words) > 0:
        unique_ratio = len(set(words)) / len(words)
        score += unique_ratio * 0.1
    
    if generated and len(generated) > 5:
        score += 0.2
    
    return max(0.0, min(1.0, score))