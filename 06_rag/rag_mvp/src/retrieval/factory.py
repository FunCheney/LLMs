from dataclasses import dataclass
from pathlib import Path

from embedding import BGEM3EmbeddingProvider
from retrieval.base import Retriever
from retrieval.bm25_retriever import SparseRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.retriever import DenseRetriever


@dataclass
class RetrieverConfig:
    mode: str = "dense"
    dense_index_path: Path = Path("data/index/local.pkl")
    bm25_index_path: Path = Path("data/index/bm25.pkl")
    device: str | None = None
    candidate_multiplier: int = 2


def create_retriever(config: RetrieverConfig) -> Retriever:
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
