"""
Embedding 相关的数据结构：向量化结果。
"""

from dataclasses import dataclass
from typing import List


@dataclass
class EmbeddingResult:
    """
    vector: 向量
    model: 生成该向量的模型名
    dimension: 维度
    """
    vector: List[float]
    model: str
    dimension: int

