from pathlib import Path

from embedding.base import EmbeddingProvider
from retrieval.base import Retriever
from retrieval.mappers import vector_record_to_hit
from retrieval.types import RetrievalResult
from vector_store.local import LocalVectorIndex


class DenseRetriever(Retriever):
    """基于本地向量索引的稠密检索。"""

    SOURCE = "dense"

    def __init__(
        self,
        index_path: str | Path,
        embedder: EmbeddingProvider,
    ) -> None:
        self.index_path = Path(index_path)
        self.embedder = embedder
        self._index: LocalVectorIndex | None = None

    def _load_index(self) -> LocalVectorIndex:
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Vector index not found: {self.index_path}. "
                "Run scripts/build_vector_index.py first."
            )
        if self._index is None:
            self._index = LocalVectorIndex.load(self.index_path)
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

        query_vector = self.embedder.embed_text(query).vector
        raw_hits = index.search(query_vector, top_k=top_k)

        hits = [
            vector_record_to_hit(record, score, self.SOURCE)
            for record, score in raw_hits
        ]
        for rank, hit in enumerate(hits, start=1):
            hit.rank = rank
        return hits
