from abc import ABC, abstractmethod

from retrieval.types import RetrievalResult


class Retriever(ABC):
    """检索器统一接口。不关心向量库或 BM25 细节。"""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        raise NotImplementedError
