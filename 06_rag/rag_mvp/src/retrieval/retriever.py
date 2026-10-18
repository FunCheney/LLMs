"""
稠密检索器：查询向量 → 向量库 Top-K。

依赖注入设计：DenseRetriever 不再自己创建/加载具体后端，
而是被动接收一个 VectorReader 实例（由 vector_store.factory.open_reader 构造），
因此 local 与 milvus 的切换对检索层完全透明。

向后兼容：仍接受 index_path= 关键字（内部懒加载 LocalVectorIndex），
旧调用（search_local_index / 既有测试）无需修改。
"""

from pathlib import Path

from embedding.base import EmbeddingProvider
from retrieval.base import Retriever
from retrieval.mappers import vector_record_to_hit
from retrieval.types import RetrievalResult
from vector_store.base import VectorReader


class DenseRetriever(Retriever):
    """基于向量库（local / milvus）的稠密检索。"""

    SOURCE = "dense"

    def __init__(
        self,
        index_path: str | Path | None = None,
        embedder: EmbeddingProvider | None = None,
        store: VectorReader | None = None,
    ) -> None:
        """
        两种构造方式（二选一）：
        - store=VectorReader 实例：推荐，后端由工厂决定；
        - index_path=本地 pickle 路径：兼容旧调用，首次检索时懒加载 LocalVectorIndex。
        """
        if store is None and index_path is None:
            raise ValueError("DenseRetriever requires either store= or index_path=")
        if embedder is None:
            raise ValueError("DenseRetriever requires embedder=")
        self.index_path = Path(index_path) if index_path is not None else None
        self.embedder = embedder
        self._store: VectorReader | None = store

    def _load_store(self) -> VectorReader:
        """返回读侧实例；index_path 模式下首次调用时加载本地索引并缓存。"""
        if self._store is None:
            assert self.index_path is not None
            if not self.index_path.exists():
                raise FileNotFoundError(
                    f"Vector index not found: {self.index_path}. "
                    "Run scripts/build_vector_index.py first."
                )
            from vector_store.local import LocalVectorIndex

            self._store = LocalVectorIndex.load(self.index_path)
        return self._store

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """把 query 编码成向量后检索，并转换为统一的 RetrievalResult。"""
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query = query.strip()
        if not query:
            return []

        store = self._load_store()
        if store.count() == 0:
            return []

        query_vector = self.embedder.embed_text(query).vector
        raw_hits = store.search(query_vector, top_k=top_k)

        hits = [
            vector_record_to_hit(record, score, self.SOURCE)
            for record, score in raw_hits
        ]
        # 记录名次，供展示与评测使用
        for rank, hit in enumerate(hits, start=1):
            hit.rank = rank
        return hits
