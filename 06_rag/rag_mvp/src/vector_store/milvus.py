from pathlib import Path
from typing import List

from vector_store import VectorRecord
from vector_store.base import VectorOpt


class MilvusVectorOpt(VectorOpt):

    def __init__(self):
        self.records: List[VectorRecord] = []

    def add(self, vector: VectorRecord):
        self.records.append(vector)

    def count(self) -> int:
        return len(self.records)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[tuple[VectorRecord, float]]:
        scored_vectors = []

        scored_vectors.sort(key=lambda x: x[1], reverse=True)

        return scored_vectors[:top_k]

    def save(self, path: str| Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)