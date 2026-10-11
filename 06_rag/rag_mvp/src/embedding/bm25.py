"""
BM25 稀疏索引（关键词检索）。

与稠密向量的互补性：
- 稠密向量擅长语义相似，但对关键词、专有名词、编号不敏感；
- BM25 基于词频，擅长精确匹配关键词（如 "CLAUDE.md"、"offset"）；
- 两者用 RRF 融合，就是检索层默认的 hybrid 模式。

中文分词用 jieba（见 embedding.tokenizer），索引内容与查询都走同一套分词。
索引对象整体用 pickle 序列化，落盘到 data/index/bm25.pkl。
"""

import pickle
from pathlib import Path
from typing import List
from rank_bm25 import BM25Okapi
from embedding.tokenizer import tokenize
from indexing.chunk_metadata import chunk_to_metadata
from ingestion.storage import load_chunks_jsonl
from vector_store.vector_store_types import VectorRecord


class BM25Index:
    """封装 BM25Okapi 的构建、检索与持久化。"""

    def __init__(self) -> None:
        self.records: List[VectorRecord] = []   # 与 BM25 词频矩阵行号一一对应的原始记录
        self.bm25: BM25Okapi | None = None       # 构建前的占位为空

    def build(self, records: List[VectorRecord]) -> None:
        """用一批记录构建索引：对每条文本分词后喂给 BM25Okapi。"""
        self.records = records
        tokenized_documents = [tokenize(record.text) for record in records]
        self.bm25 = BM25Okapi(tokenized_documents)

    def count(self) -> int:
        """返回索引中的文档数。"""
        return len(self.records)

    def search(self, query: str, top_k: int) -> List[tuple[VectorRecord, float]]:
        """检索并返回 (记录, 得分) 列表，按得分从高到低截取 top_k。"""
        if self.bm25 is None:
            raise RuntimeError("BM25 index is not built yet")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # 对全部文档按分数降序排序（rank-bm25 不提供排序，需自行处理）
        rank_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        return [
            (self.records[index], float(scores[index]))
            for index in rank_indices[:top_k]
        ]

    def save(self, path: str | Path) -> None:
        """把整个索引对象 pickle 落盘（自动创建父目录）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        """从 pickle 文件恢复索引，并校验类型。"""
        path = Path(path)
        with path.open("rb") as file:
            index = pickle.load(file)
        if not isinstance(index, cls):
            raise TypeError(f"Expected BM25Index at {path}")
        return index


def build_bm25_index(
    chunks_path: str | Path,
    index_path: str | Path,
) -> BM25Index:
    """
    便捷函数：从 chunks.jsonl 构建 BM25 索引并落盘。

    步骤：读 jsonl → 转成 VectorRecord（只填文本与元数据，无 embedding）→ build → save。
    """
    chunks = load_chunks_jsonl(chunks_path)
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")

    records = [
        VectorRecord(
            id=chunk["chunk_id"],
            text=chunk["content"],
            metadata=chunk_to_metadata(chunk),
            embedding=[],  # BM25 不需要向量
        )
        for chunk in chunks
    ]

    index = BM25Index()
    index.build(records)
    index.save(index_path)
    return index
