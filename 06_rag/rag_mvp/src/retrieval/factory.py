"""
检索器工厂：根据配置创建 Dense / Sparse / Hybrid 检索器。

统一入口 create_retriever(config)，让 CLI、评测脚本、问答层都能
「传一个 mode 字符串」就拿到正确的检索器，避免到处写 if/else。

    RetrieverConfig(mode="dense"  / "sparse" / "hybrid")
"""

from dataclasses import dataclass
from pathlib import Path

from embedding import BGEM3EmbeddingProvider
from retrieval.base import Retriever
from retrieval.spare_retriever import SparseRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.retriever import DenseRetriever


@dataclass
class RetrieverConfig:
    """检索器配置：模式、索引路径、设备与候选放大倍数。"""
    mode: str = "dense"
    dense_index_path: Path = Path("data/index/local.pkl")
    bm25_index_path: Path = Path("data/index/bm25.pkl")
    device: str | None = None
    candidate_multiplier: int = 2


def create_retriever(config: RetrieverConfig) -> Retriever:
    """
    按 config.mode 返回对应检索器。

    注意：这里会无条件构建 BGE-M3（用于给查询编码），
    即使 sparse 模式其实用不到 embedding，仍会加载模型，
    属于已知的轻微浪费，后续可按 mode 做懒加载优化。
    """
    mode = config.mode.lower()
    embedder = BGEM3EmbeddingProvider(device=config.device)
    dense = DenseRetriever(
        index_path=config.dense_index_path,
        embedder=embedder,
    )

    if mode == "dense":
        return dense

    sparse = SparseRetriever(index_path=config.bm25_index_path)

    if mode == "sparse":
        return sparse

    if mode == "hybrid":
        return HybridRetriever(
            dense_retriever=dense,
            sparse_retriever=sparse,
            candidate_multiplier=config.candidate_multiplier,
        )

    raise ValueError(f"Unsupported retriever mode: {config.mode}")
