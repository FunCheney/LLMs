
from retrieval import RetrievalResult
from rrf import reciprocal_rank_fusion
class HybridRetriever:

    def __init__(
        self,
        dense_retriever,
        bm25_retriever,
    ):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[RetrievalResult]:

        dense_results = self.dense_retriever.search(
            query,
            top_k=candidate_k,
        )

        bm25_results = self.bm25_retriever.search(
            query,
            top_k=candidate_k,
        )

        fused_results = reciprocal_rank_fusion(
            [
                dense_results,
                bm25_results,
            ]
        )

        return fused_results[:top_k]