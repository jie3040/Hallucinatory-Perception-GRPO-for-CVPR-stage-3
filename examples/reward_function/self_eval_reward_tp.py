# /root/EasyR1/examples/reward_function/self_eval_reward_tp.py

import torch
from typing import List, Dict, Any
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import re

class SelfEvaluator:
    """使用模型的独立副本进行自评（双卡并行）"""
    
    def __init__(self, model_path: str):
        print(f"[SelfEvaluator] 加载评分模型: {model_path}")
        print(f"[SelfEvaluator] 使用双卡并行加载")
        
        # 方法1：使用 device_map="auto" 自动分配到两张卡
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",  # 自动分配到所有可用GPU
            trust_remote_code=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        self.model.eval()
        
        print(f"[SelfEvaluator] 模型已加载，设备分布:")
        for name, param in self.model.named_parameters():
            if param.device.type == 'cuda':
                print(f"  {name}: GPU {param.device.index}")
                break
        
        self.scoring_prompt = """<image>Rate this image caption's quality from 0-10.
Consider: accuracy, completeness, clarity, fluency.

Caption: {caption}

Rating (0-10):"""
    
    def score(self, image: Image.Image, caption: str) -> float:
        """评分"""
        prompt = self.scoring_prompt.format(caption=caption)
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # 输入放到第一张卡
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        ).to("cuda:0")  # 输入到第一张卡，模型会自动处理跨卡通信
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.1,
                do_sample=False
            )
        
        response = self.processor.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        score = self._extract_score(response)
        return score / 10.0
    
    def _extract_score(self, text: str) -> float:
        """提取分数"""
        matches = re.findall(r'\b([0-9]|10)(?:\.\d+)?\b', text)
        if matches:
            try:
                return float(matches[0])
            except:
                pass
        return 5.0


# 全局评分器
_evaluator = None

def get_evaluator():
    global _evaluator
    if _evaluator is None:
        import os
        checkpoint_dir = "/root/autodl-tmp/EasyR1/checkpoints/easy_r1/flickr30k_caption_grpo_2x3090"
        
        # 找最新的 epoch
        epochs = [d for d in os.listdir(checkpoint_dir) if d.startswith('epoch_')]
        if epochs:
            latest_epoch = sorted(epochs)[-1]
            model_path = os.path.join(checkpoint_dir, latest_epoch, 'actor')
            print(f"[奖励函数] 使用检查点: {model_path}")
        else:
            # 如果没有检查点，使用原始模型
            model_path = "/root/autodl-tmp/models/Qwen/Qwen2-VL-2B-Instruct"
            print(f"[奖励函数] 使用原始模型: {model_path}")
        
        _evaluator = SelfEvaluator(model_path)
    
    return _evaluator


def compute_score(data: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """
    混合奖励：规则 (30%) + 模型自评 (70%)
    评分模型使用双卡并行
    """
    evaluator = get_evaluator()
    scores = []
    
    for item in data:
        generated = item.get('response', '').strip()
        reference = item.get('caption', '').strip()
        images = item.get('images', [])
        
        # 1. 规则分
        rule_score = compute_rule_score(generated, reference)
        
        # 2. 模型自评分
        if images and len(images) > 0:
            try:
                model_score = evaluator.score(images[0], generated)
            except Exception as e:
                print(f"[奖励函数] 评分失败: {e}")
                import traceback
                traceback.print_exc()
                model_score = rule_score
        else:
            model_score = rule_score
        
        # 3. 混合
        final_score = 0.3 * rule_score + 0.7 * model_score
        
        scores.append({
            "overall": final_score,
            "rule_based": rule_score,
            "model_based": model_score
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