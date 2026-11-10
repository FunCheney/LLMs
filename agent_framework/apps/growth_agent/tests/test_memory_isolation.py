"""记忆与知识库的表级隔离（Gate 7 #5）。

判据文件是 `docs/w7/memory-vs-knowledge.md`，这里只守一条**机器可判**的：
记忆不许落进知识库的表。这条断言的价值不在"现在对不对"，而在"以后有人
为了省事把 `MEMORY_TABLE` 配成 `rag_chunks` 时，进程起不来"——
配置错误在启动时炸，比在检索结果里出现一条来源不明的"资料"要好得多。
"""

from __future__ import annotations

import pytest
from growth_agent.memory import KNOWLEDGE_TABLE, PgVectorMemoryStore


class _NoopEmbedder:
    """构造期用不到 embedder：校验失败在建连之前就抛了。"""

    dimension = 8

    def embed_documents(self, texts: list[str]) -> list[object]:  # pragma: no cover
        raise AssertionError("不该被调用")


def test_memory_table_cannot_be_knowledge_table() -> None:
    with pytest.raises(ValueError, match="表级隔离"):
        PgVectorMemoryStore(
            "postgresql://unused/unused",
            _NoopEmbedder(),
            table=KNOWLEDGE_TABLE,
            dimension=8,
        )


def test_invalid_identifier_rejected() -> None:
    """表名会拼进 SQL，只允许标识符字符——注入面在这里关掉。"""

    with pytest.raises(ValueError, match="非法表名"):
        PgVectorMemoryStore(
            "postgresql://unused/unused",
            _NoopEmbedder(),
            table="memories; DROP TABLE rag_chunks",
            dimension=8,
        )
