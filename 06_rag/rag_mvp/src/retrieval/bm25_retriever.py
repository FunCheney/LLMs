from pathlib import Path

from indexing.bm25_index import BM25Index
from retrieval.base import Retriever
from retrieval.mappers import vector_record_to_hit
from retrieval.types import RetrievalResult


class SparseRetriever(Retriever):
    SOURCE = "sparse"

    def __init__(self, index_path: str | Path) -> None:
        self.index_path = Path(index_path)
        self._index: BM25Index | None = None

    def _load_index(self) -> BM25Index:
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found: {self.index_path}. "
                "Run scripts/build_bm25_index.py first."
            )
        if self._index is None:
            self._index = BM25Index.load(self.index_path)
        return self._index

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query = query.strip()
        if not query:
            return []

        index = self._load_index()
        if index.count() == 0:
            return []

        raw_hits = index.search(query, top_k)
        hits = [
            vector_record_to_hit(record, score, self.SOURCE)
            for record, score in raw_hits
        ]
        for rank, hit in enumerate(hits, start=1):
            hit.rank = rank
        return hits
