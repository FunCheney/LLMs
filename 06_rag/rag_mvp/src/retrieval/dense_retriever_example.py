from typing import List
from embedding import EmbeddingProvider
from retrieval.base import Retriever
from vector_store import LocalVectorIndex
from retrieval.types import RetrievalResult

class MyRetriever(Retriever):
    def __init__(
            self,
            embedding: EmbeddingProvider,
            index: LocalVectorIndex
    ):
        self.embedding = embedding
        self.index = index

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        query_embedding = self.embedding.embed_text(query)
        results = self.index.search(query_embedding.vector, top_k)

        return [
            RetrievalResult(
                chunk_id=record.id,
                text=record.text,
                metadata=record.metadata,
                score=score
            )
            for record, score in results
        ]

