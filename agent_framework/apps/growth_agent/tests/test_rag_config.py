"""装配层的配置映射测试。

这类测试的价值不在覆盖率，而在**防止静默回退**：`Settings` 的字段名、env 变量名、
`RetrieverConfig` 的字段名是三份独立的字符串，任一处写错都不会报错——pydantic 会
用默认值、dataclass 会用默认值，最后表现是"明明配了 pgvector，跑起来还在读 pickle"。

所以这里断言的是**值真的传过去了**，而不是"函数没抛异常"。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from agent_framework.config import Settings
from growth_agent.rag import retriever_config
from rag_mvp.retrieval.factory import _vector_store_config


@pytest.fixture()
def rag_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把 RAG_ROOT 指到临时目录：装配层只在启动时推导一次索引路径。"""
    (tmp_path / "data" / "index").mkdir(parents=True)
    monkeypatch.setenv("RAG_ROOT", str(tmp_path))
    return tmp_path


def test_pgvector_settings_reach_retriever_config(
    rag_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VECTOR_STORE", "pgvector")
    monkeypatch.setenv("PGVECTOR_DSN", "postgresql://rag:rag@db:5432/rag")
    monkeypatch.setenv("RETRIEVAL_MODE", "dense")

    config = retriever_config(Settings())

    assert config.store_type == "pgvector"
    assert config.pgvector_dsn == "postgresql://rag:rag@db:5432/rag"
    assert config.mode == "dense"


def test_defaults_fall_back_to_local(rag_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """不配就是 local + 不精排。CI 与"只想跑通对话"的机器都依赖这个默认。"""
    for name in ("VECTOR_STORE", "PGVECTOR_DSN", "RETRIEVAL_RERANK"):
        monkeypatch.delenv(name, raising=False)

    config = retriever_config(Settings())

    assert config.store_type == "local"
    assert config.pgvector_dsn is None
    assert config.reranker is None


def test_rerank_flag_maps_to_bge(rag_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`retrieval_rerank` 是布尔开关，`RetrieverConfig.reranker` 是模型名。

    这两者语义不同却必须一一对应：开了精排但没传模型名，工厂会静默不精排。
    """
    monkeypatch.setenv("RETRIEVAL_RERANK", "true")

    config = retriever_config(Settings())

    assert config.reranker == "bge"


def test_dsn_is_carried_into_vector_store_config(
    rag_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`RetrieverConfig` → `VectorStoreConfig` 这一跳也要测。

    这里断层的话，症状是"连的是 env 里的 PGVECTOR_DSN，而不是我传进去的那个"——
    在多环境（dev/staging）下会连到错误的库，而且从日志上看不出来。
    """
    monkeypatch.setenv("VECTOR_STORE", "pgvector")
    monkeypatch.setenv("PGVECTOR_DSN", "postgresql://rag:rag@db:5432/rag")

    store_config = _vector_store_config(retriever_config(Settings()))

    assert store_config.store_type == "pgvector"
    assert store_config.dsn == "postgresql://rag:rag@db:5432/rag"
    assert store_config.table == "rag_chunks"


def test_index_paths_follow_rag_root(rag_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """索引路径必须来自 RAG_ROOT，不能硬编码。

    硬编码的相对路径在容器里会解析到 `/app/data/...`，报的是 FileNotFoundError，
    看起来像"索引没构建"，实际是"找错了地方"。
    """
    monkeypatch.delenv("VECTOR_STORE", raising=False)

    config = retriever_config(Settings())

    assert config.dense_index_path == rag_root / "data" / "index" / "local.pkl"
    assert config.bm25_index_path == rag_root / "data" / "index" / "bm25.pkl"
