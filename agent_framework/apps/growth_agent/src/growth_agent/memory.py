"""W7 记忆运行时（产品层的**唯一装配点**）。

依赖方向与 `rag.py` 相同：growth_agent → rag_mvp（embedding/pgvector），
框架侧只拿到 `MemoryStore` 协议的实现。

三个刻意决策：

1. **记忆用独立的表和独立的 schema**（`agent_memories`，字段按 MemoryRecord
   逐列建），而不是塞进 `rag_chunks`。namespace 隔离的最强形态是"表级"：
   同库不同表，检索互不污染，也不用借 metadata 字段模拟结构。
2. **复用检索链路已加载的 bge-m3**（`AsyncRetriever` 内部的 embedder），
   不再实例化第二份模型——那是 18 秒冷启动 ×2，只为了让记忆用一个
   "自己的" embedder，不值得。sparse 模式没有 embedder：记忆降级关闭
   并打 WARNING（CI 的 sparse 档本来就跑不到真记忆）。
3. **提取只在"显式记忆指令"出现时跑 LLM**（"记住/以后要/不要…"的字面
   匹配，零成本正则）。每轮都提取是 W7 任务清单点名的成本陷阱；
   字面预筛 + 命中才调 cheap 档，把提取成本压到"只在用户真的想记住时"。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from agent_framework.config import Settings
from agent_framework.llm.client import LLMGateway
from agent_framework.llm.types import Message, MessageRole
from agent_framework.memory import (
    MemoryExtractor,
    MemoryItem,
    MemoryRecord,
    SummarizerUsage,
    WriteOutcome,
)
from agent_framework.memory.extract import _EXPLICIT_MARKERS
from agent_framework.memory.long_term import LongTermMemory
from agent_framework.memory.working import WorkingMemory, make_gateway_summarizer
from agent_framework.session.factory import build_session_store

logger = logging.getLogger(__name__)

# 提取的字面预筛（与 extract.py 的显式指令识别同源——
# 这里导入同一个正则，不允许两份"什么算显式"各自演化）。
EXPLICIT_MEMORY_MARKER = _EXPLICIT_MARKERS


def _vec_literal(vector: Sequence[float]) -> str:
    """pgvector 的文本向量形态（"[1.0,2.0]"）。json.dumps 恰好同构。"""
    return json.dumps(list(vector))


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS {table} (
    id TEXT PRIMARY KEY,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    confidence REAL NOT NULL,
    source_session TEXT NOT NULL,
    trust TEXT NOT NULL,
    created_at DOUBLE PRECISION NOT NULL,
    expires_at DOUBLE PRECISION,
    dedupe_hash TEXT NOT NULL,
    superseded_by TEXT,
    embedding vector({dimension})
);
CREATE INDEX IF NOT EXISTS {table}_embedding_idx
    ON {table} USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS {table}_key_idx ON {table} (key);
"""

_UPSERT_SQL = """
INSERT INTO {table} (id, key, value, confidence, source_session, trust,
                     created_at, expires_at, dedupe_hash, superseded_by, embedding)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
ON CONFLICT (id) DO UPDATE SET
    key = EXCLUDED.key,
    value = EXCLUDED.value,
    confidence = EXCLUDED.confidence,
    source_session = EXCLUDED.source_session,
    trust = EXCLUDED.trust,
    expires_at = EXCLUDED.expires_at,
    dedupe_hash = EXCLUDED.dedupe_hash,
    superseded_by = EXCLUDED.superseded_by,
    embedding = EXCLUDED.embedding
"""

_SEARCH_SQL = """
SELECT id, key, value, confidence, source_session, trust, created_at,
       expires_at, dedupe_hash, superseded_by,
       1 - (embedding <=> %s::vector) AS score
FROM {table}
ORDER BY embedding <=> %s::vector
LIMIT %s
"""

_SELECT_ALL_SQL = """
SELECT id, key, value, confidence, source_session, trust, created_at,
       expires_at, dedupe_hash, superseded_by
FROM {table}
"""

_DELETE_SQL = "DELETE FROM {table} WHERE id = ANY(%s)"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9_]+$")

# 知识库的表名（W5/W6 的 rag_chunks）。只用于"记忆不许借住"这条断言，
# 不在这里读写它——跨表操作会让两条链路重新耦合。
KNOWLEDGE_TABLE = "rag_chunks"

_COLUMNS = (
    "id", "key", "value", "confidence", "source_session", "trust",
    "created_at", "expires_at", "dedupe_hash", "superseded_by",
)


class PgVectorMemoryStore:
    """`MemoryStore` 协议的 pgvector 实现。

    与 `rag_chunks` 同一个数据库、**不同的表**（namespace 隔离）。
    写路径单事务批量提交（顶替记录 + 新记录一起写，失败整体回滚）——
    对齐框架协议注释里的"单次 upsert 是一个提交单元"。
    """

    def __init__(
        self,
        dsn: str,
        embedder: Any,
        *,
        table: str = "agent_memories",
        dimension: int = 1024,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        if not _IDENTIFIER.match(table):
            raise ValueError(f"非法表名：{table!r}")
        # 记忆与知识库**必须**不同表（Gate 7 #5 的隔离判据）。把它们配成同一张表
        # 不会立刻报错，只会让记忆项混进检索候选、编号指到记忆上——症状是
        # "引用里出现了一条谁都没写过的资料"。在构造函数里挡住这类配置。
        if table == KNOWLEDGE_TABLE:
            raise ValueError(
                f"记忆表不能是知识库表 {KNOWLEDGE_TABLE!r}："
                f"记忆与知识库必须表级隔离（见 docs/w7/memory-vs-knowledge.md）"
            )
        self._dsn = dsn
        self._embedder = embedder
        self._table = table
        self._dimension = dimension
        self._owns_executor = executor is None
        # 单线程：写库串行（同表同索引），与 AsyncDocumentStore 同一取舍。
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="memory-store"
        )
        self._conn: Any = None
        self._connect_and_init()

    # -- 协议实现 ---------------------------------------------------------- #

    @property
    def name(self) -> str:
        return f"pgvector:{self._table}"

    async def upsert(self, records: Sequence[MemoryRecord]) -> int:
        if not records:
            return 0
        texts = [f"{r.key}: {r.value}" for r in records]
        vectors = await self._embed(texts)
        rows = [
            (
                r.id, r.key, r.value, r.confidence, r.source_session, r.trust,
                r.created_at, r.expires_at, r.dedupe_hash, r.superseded_by,
                _vec_literal(v),
            )
            for r, v in zip(records, vectors, strict=True)
        ]
        return await self._run(self._upsert_sync, rows)

    async def search(self, query: str, *, top_k: int) -> list[tuple[MemoryRecord, float]]:
        if top_k <= 0 or not query.strip():
            return []
        vectors = await self._embed([query])
        literal = _vec_literal(vectors[0])
        return await self._run(self._search_sync, literal, top_k)

    async def delete(self, memory_ids: Sequence[str]) -> int:
        if not memory_ids:
            return 0
        return await self._run(self._delete_sync, list(memory_ids))

    async def all_records(self) -> list[MemoryRecord]:
        return await self._run(self._select_all_sync)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        if self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)

    # -- 同步部分（线程池里跑） -------------------------------------------- #

    def _connect_and_init(self) -> None:
        import psycopg

        self._conn = psycopg.connect(self._dsn, autocommit=False)
        self._conn.execute(_SCHEMA_SQL.format(table=self._table, dimension=self._dimension))
        self._conn.commit()

    def _upsert_sync(self, rows: list[tuple]) -> int:
        sql = _UPSERT_SQL.format(table=self._table)
        # 单事务：新记录与顶替记录一起提交或一起回滚（不留半条）。
        try:
            for row in rows:
                self._conn.execute(sql, row)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return len(rows)

    def _search_sync(self, literal: str, top_k: int) -> list[tuple[MemoryRecord, float]]:
        sql = _SEARCH_SQL.format(table=self._table)
        cursor = self._conn.execute(sql, [literal, literal, top_k])
        return [(self._to_record(row), float(row[10])) for row in cursor.fetchall()]

    def _delete_sync(self, ids: list[str]) -> int:
        cursor = self._conn.execute(_DELETE_SQL.format(table=self._table), [ids])
        self._conn.commit()
        return cursor.rowcount

    def _select_all_sync(self) -> list[MemoryRecord]:
        cursor = self._conn.execute(_SELECT_ALL_SQL.format(table=self._table))
        return [self._to_record(row) for row in cursor.fetchall()]

    @staticmethod
    def _to_record(row: Any) -> MemoryRecord:
        values = dict(zip(_COLUMNS, row[:10], strict=False))
        return MemoryRecord(**values)

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            self._executor, self._embedder.embed_documents, texts
        )
        return [r.vector for r in results]

    async def _run(self, func: Any, *args: Any) -> Any:
        import functools

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, functools.partial(func, *args)
        )


@dataclass
class MemoryRuntime:
    """装配结果。持有需要显式释放的资源（pgvector 连接、线程池）。"""

    store: PgVectorMemoryStore
    long_term: LongTermMemory
    extractor: MemoryExtractor
    working: WorkingMemory
    gateway: LLMGateway
    settings: Settings

    async def recall_for_query(self, question: str) -> list[MemoryItem]:
        """按当前问题召回活跃记忆（受 recall_k 与置信度闸约束）。"""
        try:
            return await self.long_term.recall(
                question, k=self.settings.memory_recall_k
            )
        except Exception:  # noqa: BLE001 - 记忆不可用不挡回答（降级语义）
            logger.warning("memory.recall 失败，本次回答不带记忆注入", exc_info=True)
            return []

    async def session_view(self, session_id: str) -> list[Message]:
        """工作记忆视图：[摘要] + 最近原文。失败时退回原样历史——
        多轮上下文是增强，不是正确性依赖。"""
        try:
            return await self.working.view(session_id)
        except Exception:  # noqa: BLE001
            logger.warning("memory.session_view 失败，退回原始历史", exc_info=True)
            return []

    async def record_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
        *,
        request_id: str | None = None,
    ) -> None:
        """回答落地后的三件事：追加本轮 → 超阈值压缩 → 显式指令才提取。

        追加用 SessionStore 的幂等（同一 request_id 不翻倍）；压缩每轮
        都调（幂等，未达阈值就是 skipped）；提取只在字面预筛命中时调 LLM。
        全部失败都只记日志——回答已经给出去了，落地失败不该变成 500。
        """
        try:
            await self.working._store.append(  # noqa: SLF001 - 同包内部接线
                session_id,
                [
                    Message(role=MessageRole.USER, content=question),
                    Message(role=MessageRole.ASSISTANT, content=answer),
                ],
                request_id=request_id,
            )
        except Exception:  # noqa: BLE001
            logger.warning("memory.record_turn: 会话追加失败", exc_info=True)
            return

        try:
            await self.working.compress(session_id)
        except Exception:  # noqa: BLE001
            logger.warning("memory.record_turn: 压缩失败（下一轮会重试）", exc_info=True)

        if self.should_extract(question):
            messages = [
                Message(role=MessageRole.USER, content=question),
                Message(role=MessageRole.ASSISTANT, content=answer),
            ]
            try:
                outcome = await self.extract_and_write(messages, source_session=session_id)
            except Exception:  # noqa: BLE001
                logger.warning("memory.record_turn: 提取失败", exc_info=True)
                return
            if outcome is not None:
                logger.info("memory.extract: %s", outcome.as_dict())

    def should_extract(self, question: str) -> bool:
        """提取的字面预筛：只有出现显式记忆指令才调 LLM 提取。"""
        return bool(EXPLICIT_MEMORY_MARKER.search(question or ""))

    async def extract_and_write(
        self, messages: list[Message], *, source_session: str
    ) -> WriteOutcome | None:
        """提取 → 写入（会话结束或显式触发时调用；answer 链路按预筛触发）。"""
        try:
            result = await self.extractor.extract(messages, source_session=source_session)
        except Exception:  # noqa: BLE001 - 提取失败按空结果处理
            logger.warning("memory.extract 失败", exc_info=True)
            return None
        if not result.candidates:
            return None
        return await self.long_term.write(result.candidates)

    def close(self) -> None:
        self.store.close()


def build_memory_runtime(
    settings: Settings,
    gateway: LLMGateway,
    *,
    embedder: Any,
    session_store: Any = None,
) -> MemoryRuntime:
    """装配记忆链路。同步、慢（建连接），只在启动/惰性装配时调一次。

    `embedder` 是**同步**的 bge provider（从检索链路借来复用）；
    `session_store` 缺省按 Settings 建（memory 后端足够 demo，
    多副本生产换 redis——`build_session_store` 只认 Settings）。
    """
    if embedder is None:
        raise ValueError(
            "memory 运行时需要 embedder（dense 检索链路的 bge-m3）。"
            "sparse 模式没有 embedder，MEMORY=governed 需要配 dense。"
        )
    dsn_value = settings.pgvector_dsn
    if dsn_value is None:
        raise ValueError("MEMORY=governed 需要 PGVECTOR_DSN（记忆存 pgvector）")
    dsn = dsn_value.get_secret_value()

    store = PgVectorMemoryStore(
        dsn,
        embedder,
        table=settings.memory_table,
        dimension=getattr(embedder, "dimension", None)
        or getattr(getattr(embedder, "model", None), "dim", 1024)
        or 1024,
    )
    long_term = LongTermMemory(
        store,
        min_confidence=settings.memory_min_confidence,
    )
    extractor = MemoryExtractor(gateway)

    resolved_store = session_store or build_session_store(settings)
    # 摘要用量与摘要器共用同一个累计器：压缩是一次真实 LLM 调用，
    # 不记账就会在成本报告里变成"免费"。
    summary_usage = SummarizerUsage()
    working = WorkingMemory(
        resolved_store,
        summarizer=make_gateway_summarizer(gateway, usage=summary_usage),
        token_threshold=settings.working_memory_token_threshold,
        summarize_after=settings.working_memory_summarize_after,
        keep_recent_messages=settings.working_memory_keep_recent,
        summarizer_usage=summary_usage,
    )

    logger.info(
        "memory ready: table=%s recall_k=%d min_confidence=%.2f threshold=%dtok",
        settings.memory_table,
        settings.memory_recall_k,
        settings.memory_min_confidence,
        settings.working_memory_token_threshold,
    )
    return MemoryRuntime(
        store=store,
        long_term=long_term,
        extractor=extractor,
        working=working,
        gateway=gateway,
        settings=settings,
    )
