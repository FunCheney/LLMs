from typing import List
import pickle
from pathlib import Path

from vector_store_types import VectorRecord

def dot_product(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class LocalVectorIndex:
    def __init__(self):
        self.records: List[VectorRecord] = []

    def add(self, vector: VectorRecord):
        self.records.append(vector)

    def count(self) -> int:
        return len(self.records)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[tuple[VectorRecord, float]]:
        scored_vectors = []
        for record in self.records:
            score = dot_product(query_embedding, record.embedding)

            scored_vectors.append(
                (record, score)
            )

        scored_vectors.sort(key=lambda x: x[1], reverse=True)

        return scored_vectors[:top_k]

