"""
检索器工厂：根据配置创建 Dense / Sparse / Hybrid 检索器。

统一入口 create_retriever(config)，让 CLI、评测脚本、问答层都能
「传一个 mode 字符串」就拿到正确的检索器，避免到处写 if/else。

    RetrieverConfig(mode="dense"  / "sparse" / "hybrid")
"""

from dataclasses import dataclass
from pathlib import Path

from retrieval.base import Retriever
from retrieval.spare_retriever import SparseRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.retriever import DenseRetriever


@dataclass
class RetrieverConfig:
    """检索器配置：模式、索引路径、向量库后端、设备与候选放大倍数。"""
    mode: str = "dense"
    dense_index_path: Path = Path("data/index/local.pkl")
    bm25_index_path: Path = Path("data/index/bm25.pkl")
    device: str | None = None
    candidate_multiplier: int = 2
    # 向量库后端：local（本地 pickle，默认）| milvus（Milvus / Milvus Lite）
    store_type: str = "local"
    milvus_uri: str | None = None        # None 时用 VectorStoreConfig 的 env 默认值
    milvus_token: str | None = None
    milvus_collection: str | None = None
    # 重排序：None 表示不重排；目前支持 "bge"（BAAI/bge-reranker-v2-m3）
    reranker: str | None = None
    rerank_model: str | None = None       # 自定义 reranker 模型名（默认 bge-reranker-v2-m3）
    rerank_candidate_multiplier: int = 10  # 召回放大倍数（top_k * N，至少 20 条进精排）


def _vector_store_config(config: RetrieverConfig) -> "VectorStoreConfig":
    """把检索配置中的后端字段收敛为 VectorStoreConfig（未传字段走 env 默认值）。"""
    from vector_store.config import VectorStoreConfig

    kwargs: dict = {
        "store_type": config.store_type,
        "index_path": config.dense_index_path,
    }
    if config.milvus_uri is not None:
        kwargs["uri"] = config.milvus_uri
    if config.milvus_token is not None:
        kwargs["token"] = config.milvus_token
    if config.milvus_collection is not None:
        kwargs["collection"] = config.milvus_collection
    return VectorStoreConfig(**kwargs)


def create_retriever(config: RetrieverConfig) -> Retriever:
    """
    按 config.mode 返回对应检索器；config.reranker 非空时再包一层重排。

    dense/hybrid 模式通过 vector_store.factory.open_reader 打开后端
    （store_type=local 读本地 pickle；store_type=milvus 连 Milvus），
    DenseRetriever 本身不感知后端差异。
    sparse 模式不需要 embedding 与向量库，完全不加载模型/不连接 Milvus。
    """
    mode = config.mode.lower()

    if mode == "sparse":
        retriever: Retriever = SparseRetriever(index_path=config.bm25_index_path)
    elif mode in {"dense", "hybrid"}:
        # 惰性导入：embedding 会连带加载 sentence-transformers/torch（很重），
        # 只有真正构建 dense 检索器时才引入
        from embedding import BGEM3EmbeddingProvider
        from vector_store.factory import open_reader

        embedder = BGEM3EmbeddingProvider(device=config.device)
        dense = DenseRetriever(
            store=open_reader(_vector_store_config(config)),
            embedder=embedder,
        )
        if mode == "dense":
            retriever = dense
        else:
            retriever = HybridRetriever(
                dense_retriever=dense,
                sparse_retriever=SparseRetriever(index_path=config.bm25_index_path),
                candidate_multiplier=config.candidate_multiplier,
            )
    else:
        raise ValueError(f"Unsupported retriever mode: {config.mode}")

    if config.reranker is not None:
        retriever = _wrap_with_reranker(retriever, config)

    return retriever


def _wrap_with_reranker(retriever: Retriever, config: RetrieverConfig) -> Retriever:
    """把基础检索器包装成「召回放大 + 重排精排」的组合检索器。"""
    from reranker import BGEReranker, RerankedRetriever

    if config.reranker.lower() != "bge":
        raise ValueError(f"Unsupported reranker: {config.reranker}")

    reranker_kwargs: dict = {}
    if config.rerank_model:
        reranker_kwargs["model_name"] = config.rerank_model
    reranker = BGEReranker(device=config.device, **reranker_kwargs)

    return RerankedRetriever(
        base_retriever=retriever,
        reranker=reranker,
        candidate_multiplier=config.rerank_candidate_multiplier,
    )
