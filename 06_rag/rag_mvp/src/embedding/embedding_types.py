from dataclasses import dataclass
from typing import List


@dataclass
class EmbeddingResult:
    vector: List[float]
    model: str
    dimension: int

