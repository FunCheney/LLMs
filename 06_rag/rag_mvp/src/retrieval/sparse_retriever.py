from pathlib import Path
from typing import List

from indexing.bm25_index import BM25Index
from retrieval import RetrievalResult
from retrieval.base import Retriever


class SparseRetriever(Retriever):
    SOURCE = "sparse"

    def __init__(
            self,
            index_path: str | Path,
            index: BM25Index,
    ) -> None:
        self.index_path = Path(index_path)
        self.index = index

    def retrieve(self, query, top_k=3) -> List[RetrievalResult]:

        results = self.index.search(query, top_k)

        return [
            RetrievalResult(
                chunk_id=record.id,
                text=record.text,
                metadata=record.metadata,
                score=score
            )

            for record, score in results
        ]
