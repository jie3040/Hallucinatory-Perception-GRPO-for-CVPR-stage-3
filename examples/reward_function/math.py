# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any

# mathruler 是一个数学答案评分库
from mathruler.grader import extract_boxed_content, grade_answer
# - extract_boxed_content: 从 LaTeX 格式中提取 \boxed{答案} 里的内容
# - grade_answer: 判断提取的答案是否与标准答案相符


def format_reward(response: str) -> float:
    """
    检查响应是否符合期望的格式
    
    期望格式：<think>推理过程</think> ... \boxed{最终答案}
    这是 DeepSeek-R1 和类似推理模型的标准输出格式
    
    Returns:
        1.0 如果格式正确，0.0 如果格式错误
    """

    # 定义正则表达式模式
    # <think>.*</think>  : 必须有思考标签包裹的推理过程
    # .*                 : 中间可以有任意内容
    # \\boxed\{.*\}      : 必须有 \boxed{...} 包裹的最终答案
    # .*                 : 后面可以有任意内容
    # re.DOTALL          : 让 . 匹配包括换行符在内的所有字符
    
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    # re.fullmatch: 检查整个字符串是否完全匹配模式
    format_match = re.fullmatch(pattern, response)
    
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:

    """
    检查答案的准确性
    
    Args:
        response: 模型生成的完整响应
        ground_truth: 标准答案（通常也是 LaTeX 格式）
    
    Returns:
        1.0 如果答案正确，0.0 如果答案错误
    """
    
    answer = extract_boxed_content(response)

    # 使用 mathruler.grader 的评分函数判断答案是否正确
    # grade_answer 会处理各种数学表达式的等价性：
    # - 数值等价：0.5 == 1/2
    # - 代数等价：x^2 - 1 == (x-1)(x+1)
    # - LaTeX 格式差异：\frac{1}{2} == 1/2
    
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:

    """
    批量计算 reward 分数（对应配置中的 reward_type: batch）
    
    Args:
        reward_inputs: 一批输入数据，每个元素包含：
            {
                "response": "模型生成的响应",
                "response_length": 42,
                "ground_truth": "标准答案"
            }
        format_weight: 格式分数的权重（默认 0.1，即 10%）
                      剩余权重 (1 - 0.1 = 0.9，即 90%) 给准确性
    
    Returns:
        分数列表，每个元素包含：
            {
                "overall": 加权总分,
                "format": 格式分数,
                "accuracy": 准确性分数
            }
    """
    
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:

        # ⚠️ 关键预处理步骤：标准化空格
        # 原始响应：  "< think >推理</ think  > \boxed{ 42 }"
        # 处理后：    "<think>推理</think> \boxed{42}"
        # 
        # 这是为了处理某些模型（如 qwen2.5vl-32b）在标签周围添加额外空格的问题
        # re.sub(pattern, replacement, string):
        #   pattern: r"\s*(<|>|/)\s*"  匹配 <, >, / 前后的任意空格
        #   replacement: r"\1"          只保留捕获组1（即 <, >, / 本身）
        
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format

        # 计算格式分数
        format_score = format_reward(response)

        # 计算准确性分数
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])

        
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores
