from dataclasses import dataclass
from typing import List


@dataclass
class EmbeddingResult:
    """
    vector: 向量
    dimension: 维度
    """
    vector: List[float]
    model: str
    dimension: int

