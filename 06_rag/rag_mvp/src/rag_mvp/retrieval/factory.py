"""
检索器工厂：根据配置创建 Dense / Sparse / Hybrid 检索器。

统一入口 create_retriever(config)，让 CLI、评测脚本、问答层都能
「传一个 mode 字符串」就拿到正确的检索器，避免到处写 if/else。

    RetrieverConfig(mode="dense"  / "sparse" / "hybrid")
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rag_mvp.retrieval.base import Retriever
from rag_mvp.retrieval.spare_retriever import SparseRetriever
from rag_mvp.retrieval.hybrid import HybridRetriever
from rag_mvp.retrieval.retriever import DenseRetriever

if TYPE_CHECKING:
    from rag_mvp.reranker.base import Reranker


@dataclass
class RetrieverConfig:
    """检索器配置：模式、索引路径、向量库后端、设备与候选放大倍数。"""
    mode: str = "dense"
    dense_index_path: Path = Path("data/index/local.pkl")
    bm25_index_path: Path = Path("data/index/bm25.pkl")
    device: str | None = None
    candidate_multiplier: int = 2
    # 向量库后端：local（本地 pickle，默认）| milvus | pgvector
    store_type: str = "local"
    milvus_uri: str | None = None        # None 时用 VectorStoreConfig 的 env 默认值
    milvus_token: str | None = None
    milvus_collection: str | None = None
    pgvector_dsn: str | None = None      # None 时读 PGVECTOR_DSN
    # 重排序：None 表示不重排；目前支持 "bge"（BAAI/bge-reranker-v2-m3）
    reranker: str | None = None
    rerank_model: str | None = None       # 自定义 reranker 模型名（默认 bge-reranker-v2-m3）
    rerank_candidate_multiplier: int = 10  # 召回放大倍数（top_k * N，至少 20 条进精排）


def _vector_store_config(config: RetrieverConfig) -> "VectorStoreConfig":
    """把检索配置中的后端字段收敛为 VectorStoreConfig（未传字段走 env 默认值）。"""
    from rag_mvp.vector_store.config import VectorStoreConfig

    kwargs: dict = {
        "store_type": config.store_type,
        "index_path": config.dense_index_path,
    }
    if config.pgvector_dsn is not None:
        kwargs["dsn"] = config.pgvector_dsn
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

    这是 CLI / 评测脚本用的**一体化**入口：拿到一个对象就能检索。
    需要"召回与精排分开、各自可降级"的场景（如 agent_framework 的
    RetrievalPipeline）请用 `create_base_retriever` + `create_reranker`，
    否则精排超时后没有 RRF 结果可以回退——一体化检索器里那一层被包住了。
    """
    retriever = create_base_retriever(config)
    if config.reranker is not None:
        retriever = _wrap_with_reranker(retriever, config)
    return retriever


def create_base_retriever(config: RetrieverConfig) -> Retriever:
    """只构造召回层（dense / sparse / hybrid），**不含精排**。

    dense/hybrid 模式通过 vector_store.factory.open_reader 打开后端
    （store_type=local 读本地 pickle；milvus 连 Milvus；pgvector 连 postgres），
    DenseRetriever 本身不感知后端差异。
    sparse 模式不需要 embedding 与向量库，完全不加载模型/不连接数据库。
    """
    mode = config.mode.lower()

    if mode == "sparse":
        return SparseRetriever(index_path=config.bm25_index_path)

    if mode in {"dense", "hybrid"}:
        # 惰性导入：embedding 会连带加载 sentence-transformers/torch（很重），
        # 只有真正构建 dense 检索器时才引入
        from rag_mvp.embedding import BGEM3EmbeddingProvider
        from rag_mvp.vector_store.factory import open_reader

        embedder = BGEM3EmbeddingProvider(device=config.device)
        dense = DenseRetriever(
            store=open_reader(_vector_store_config(config)),
            embedder=embedder,
        )
        if mode == "dense":
            return dense
        return HybridRetriever(
            dense_retriever=dense,
            sparse_retriever=SparseRetriever(index_path=config.bm25_index_path),
            candidate_multiplier=config.candidate_multiplier,
        )

    raise ValueError(f"Unsupported retriever mode: {config.mode}")


def create_reranker(config: RetrieverConfig) -> "Reranker | None":
    """按 config.reranker 构造精排器；None 表示不精排。

    单独暴露出来是为了让调用方自己控制超时与降级：精排是整条链路里最慢、
    最容易抖的一段，把它和召回绑在一个对象里，就没法"超时了回退到召回结果"。
    """
    if config.reranker is None:
        return None
    from rag_mvp.reranker import BGEReranker, Reranker

    if config.reranker.lower() != "bge":
        raise ValueError(f"Unsupported reranker: {config.reranker}")

    reranker_kwargs: dict = {}
    if config.rerank_model:
        reranker_kwargs["model_name"] = config.rerank_model
    reranker: Reranker = BGEReranker(device=config.device, **reranker_kwargs)
    return reranker


def _wrap_with_reranker(retriever: Retriever, config: RetrieverConfig) -> Retriever:
    """把基础检索器包装成「召回放大 + 重排精排」的组合检索器。"""
    from rag_mvp.reranker import RerankedRetriever

    reranker = create_reranker(config)
    assert reranker is not None  # config.reranker 非空才会走到这里
    return RerankedRetriever(
        base_retriever=retriever,
        reranker=reranker,
        candidate_multiplier=config.rerank_candidate_multiplier,
    )
