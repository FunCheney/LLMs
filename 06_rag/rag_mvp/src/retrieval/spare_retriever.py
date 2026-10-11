"""
稀疏检索器：BM25 关键词检索。

与 DenseRetriever 同构，只是把「向量索引」换成「BM25 索引」，
并且查询直接传原始字符串（分词在 BM25 内部完成）。

命名说明：文件名拼写为 spare_retriever（应为 sparse），
为保持向后兼容未重命名，import 时注意路径。
"""

from pathlib import Path

from embedding.bm25 import BM25Index
from retrieval.base import Retriever
from retrieval.mappers import vector_record_to_hit
from retrieval.types import RetrievalResult


class SparseRetriever(Retriever):
    """基于 BM25 索引的稀疏检索。"""

    SOURCE = "sparse"

    def __init__(self, index_path: str | Path) -> None:
        self.index_path = Path(index_path)
        self._index: BM25Index | None = None  # 懒加载缓存

    def _load_index(self) -> BM25Index:
        """首次调用时加载 BM25 索引并缓存；文件缺失给出可操作的提示。"""
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found: {self.index_path}. "
                "Run scripts/build_bm25_index.py first."
            )
        if self._index is None:
            self._index = BM25Index.load(self.index_path)
        return self._index

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """用原始 query 检索 BM25 索引，并转换为统一的 RetrievalResult。"""
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
