from retrieval.base import Retriever
from retrieval.rrf import reciprocal_rank_fusion
from retrieval.types import RetrievalResult


class HybridRetriever(Retriever):
    SOURCE = "hybrid"

    def __init__(
        self,
        dense_retriever: Retriever,
        sparse_retriever: Retriever,
        candidate_multiplier: int = 2,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.candidate_multiplier = candidate_multiplier

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query = query.strip()
        if not query:
            return []

        candidate_k = max(top_k * self.candidate_multiplier, top_k)
        dense_results = self.dense_retriever.retrieve(query, top_k=candidate_k)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=candidate_k)

        return reciprocal_rank_fusion(
            [dense_results, sparse_results],
            top_k=top_k,
            source=self.SOURCE,
        )
