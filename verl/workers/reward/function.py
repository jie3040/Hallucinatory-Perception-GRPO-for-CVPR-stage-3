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

import importlib.util
import os
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import Callable, Optional, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


class RewardInput(TypedDict):
    """传递给 reward 函数的输入数据结构"""
    response: str        # 模型生成的响应文本
    response_length: int  # 响应的 token 长度
    ground_truth: str      # 标准答案/真实标签


class RewardScore(TypedDict):
    overall: float     # 总体奖励分数（必须有）
    format: Optional[float]   # 格式分数（可选）
    accuracy: Optional[float]   # 准确性分数（可选）
    # 可以根据需求添加更多维度的分数

# 顺序处理：每次处理一个样本
SequentialRewardFunction = Callable[[RewardInput], RewardScore]

# 批量处理：一次处理多个样本，效率更高
BatchRewardFunction = Callable[[list[RewardInput]], list[RewardScore]]


class FunctionRewardManager(ABC):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.config = config
        self.tokenizer = tokenizer

    @abstractmethod
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        """Compute reward for a batch of data."""
        ...


class SequentialFunctionRewardManager(FunctionRewardManager):
    """顺序处理每个样本的 reward manager"""
    reward_fn: SequentialRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        
        # 初始化 reward tensor，shape: [batch_size, seq_len]
        # 默认全为 0，只在响应的最后一个 token 位置填充实际 reward
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]

        # 计算每个样本的有效响应长度（非 padding 部分）
        # response_mask 中 1 表示有效 token，0 表示 padding
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        
        for i in range(len(data)):
            
            # 获取当前样本的响应长度，转为 Python int 避免 tensor 索引错误
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error

            # 截取有效的响应 token（去除 padding）
            valid_response_ids = response_ids[i][:cur_response_length]

            # 将 token ids 解码为文本
            # skip_special_tokens: 是否跳过特殊 token（如 [PAD], [EOS]）
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )

            # 调用自定义 reward 函数计算分数
            score = self.reward_fn(
                {
                    "response": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                }
            )

            # ⚠️ 关键：将 overall 分数放在响应的最后一个 token 位置
            # 这是因为 GRPO 只在序列结束时给予 reward
            reward_tensor[i, cur_response_length - 1] = score["overall"]

            # 收集所有维度的分数用于监控
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


class BatchFunctionRewardManager(FunctionRewardManager):
    reward_fn: BatchRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_inputs = []
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            reward_inputs.append(
                {
                    "response": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                }
            )

        scores = self.reward_fn(reward_inputs)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i, score in enumerate(scores):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics
