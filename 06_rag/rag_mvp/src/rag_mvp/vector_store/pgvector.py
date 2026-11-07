"""pgvector 后端：把向量放进"已经在跑的那个 postgres"。

选它的理由不是性能，是**运维面**：项目里已经有 postgres（Temporal 用它），
加一个 `vector` 扩展就是同库不同表；Milvus 要多一份常驻服务、多一份内存、
多一套备份策略，而它的收益（亿级向量、原生分区）在万级 chunk 上用不到。

四条实现约束，都来自 `vector_store/base.py` 的语义契约：

- `add()` 只进内存缓冲，不碰数据库——否则批量导入会变成 N 次往返；
- `save()` 才落库，且**失败时不清空缓冲**（否则一次网络抖动就丢整批数据）；
- `count()` 数已持久化的行，不是缓冲里的；
- `upsert()` 按 chunk_id 幂等，重复执行不产生重复数据。

两个刻意的选择：

1. **embedding 以文本字面量传入**（`%s::vector`），不装 `pgvector` 这个 Python 包。
   少一个依赖，且省掉"驱动注册"这一步——注册漏了会在第一次查询时报一个
   与向量毫无关系的错。
2. **不使用连接池**。检索是低频调用且已有自己的线程池限流，再加一层池会让
   "到底是哪层在排队"变得不可归因。连接断了就重连，代价可接受。
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

from rag_mvp.vector_store.base import VectorStore
from rag_mvp.vector_store.vector_store_types import VectorRecord

# 表名要拼进 SQL，只能用标识符字符。不做这层校验的话，
# `table="x; DROP TABLE chunks"` 会被当成合法表名执行。
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

DEFAULT_TABLE = "rag_chunks"
DEFAULT_DIMENSION = 1024  # bge-m3

_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS {table} (
    chunk_id       text PRIMARY KEY,
    document_id    text NOT NULL,
    text           text NOT NULL,
    metadata       jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    embedding      vector({dimension}),
    content_hash   text,
    model_snapshot text,
    updated_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS {table}_doc_idx ON {table} (document_id);
CREATE INDEX IF NOT EXISTS {table}_embedding_idx
    ON {table} USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS {table}_metadata_idx ON {table} USING gin (metadata);

-- 文档注册表（W6 增量索引）。存在的唯一理由是让"这份文档变没变"变成一次主键查询，
-- 而不是"扫该文档的所有 chunk 取一个 hash"。后者在文档被清空后查不到任何东西，
-- 于是内容没变也会被判成首次入库，每次全量重嵌。
CREATE TABLE IF NOT EXISTS {table}_documents (
    document_id    text PRIMARY KEY,
    content_hash   text NOT NULL,
    version        integer NOT NULL,
    model_snapshot text,
    chunk_count    integer NOT NULL DEFAULT 0,
    updated_at     timestamptz NOT NULL DEFAULT now()
);
"""

_DOC_UPSERT_SQL = """
INSERT INTO {table}_documents
    (document_id, content_hash, version, model_snapshot, chunk_count)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (document_id) DO UPDATE SET
    content_hash   = EXCLUDED.content_hash,
    version        = EXCLUDED.version,
    model_snapshot = EXCLUDED.model_snapshot,
    chunk_count    = EXCLUDED.chunk_count,
    updated_at     = now()
"""

_UPSERT_SQL = """
INSERT INTO {table} (chunk_id, document_id, text, metadata, embedding, content_hash, model_snapshot)
VALUES (%s, %s, %s, %s::jsonb, %s::vector, %s, %s)
ON CONFLICT (chunk_id) DO UPDATE SET
    document_id    = EXCLUDED.document_id,
    text           = EXCLUDED.text,
    metadata       = EXCLUDED.metadata,
    embedding      = EXCLUDED.embedding,
    content_hash   = EXCLUDED.content_hash,
    model_snapshot = EXCLUDED.model_snapshot,
    updated_at     = now()
"""


def to_vector_literal(embedding: list[float]) -> str:
    """向量 → pgvector 能接受的文本字面量 `[1,2,3]`。

    不用 numpy 的 repr：它会输出 `np.float32(0.1)` 这种非字面量，
    postgres 侧报的是语法错误，看起来跟向量完全无关。
    """
    return "[" + ",".join(f"{float(value):.7g}" for value in embedding) + "]"


class PgVectorStore(VectorStore):
    """基于 postgres + pgvector 的向量库。

    用法::

        store = PgVectorStore("postgresql://user:pw@localhost:5432/rag")
        store.upsert(VectorRecord(id="c1", text="...", metadata={...}, embedding=[...]))
        hits = store.search(query_vector, top_k=5)
    """

    def __init__(
        self,
        dsn: str | None = None,
        *,
        table: str = DEFAULT_TABLE,
        dimension: int = DEFAULT_DIMENSION,
        connect: Callable[[], Any] | None = None,
        ensure_schema: bool = True,
    ) -> None:
        if not _IDENTIFIER.match(table):
            raise ValueError(f"非法表名：{table!r}（只允许字母、数字、下划线）")
        self.table = table
        self.dimension = dimension
        self._dsn = dsn or os.getenv("PGVECTOR_DSN", "")
        if not self._dsn and connect is None:
            raise ValueError(
                "缺少连接串：传 dsn= 或设环境变量 PGVECTOR_DSN。"
                "模板见 ops/.env.example。"
            )
        self._connect_impl = connect
        self._conn: Any | None = None
        self._buffer: list[VectorRecord] = []
        if ensure_schema:
            self.ensure_schema()

    # -- 连接 --------------------------------------------------------------- #

    def _connection(self) -> Any:
        if self._conn is not None and not getattr(self._conn, "closed", False):
            return self._conn

        if self._connect_impl is not None:
            conn = self._connect_impl()
        else:
            try:
                import psycopg
            except ImportError as exc:  # pragma: no cover - 取决于安装环境
                raise ImportError(
                    "pgvector 后端需要 psycopg。执行 `uv sync --extra pgvector` 后重试。"
                ) from exc
            # autocommit：写路径都是一条语句一个语义单元，
            # 显式事务留给 W6 的增量索引（那里"删旧写新"必须原子）。
            conn = psycopg.connect(self._dsn, autocommit=True)
        self._conn = conn
        return conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> PgVectorStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- schema ------------------------------------------------------------- #

    def ensure_schema(self) -> None:
        """建扩展、表与索引。全部 IF NOT EXISTS，可重复执行。"""
        sql = _SCHEMA_SQL.format(table=self.table, dimension=self.dimension)
        self._connection().execute(sql)

    # -- 写侧 --------------------------------------------------------------- #

    def add(self, vector: VectorRecord) -> None:
        """只进缓冲。**不落库**——批量导入时逐条往返会让耗时随条数线性上升。"""
        self._buffer.append(vector)

    def save(self, path: str | Path = "") -> None:  # noqa: ARG002 - 契约要求该参数
        """把缓冲写入数据库。成功后清空缓冲；失败时**保留**，可重试。"""
        if not self._buffer:
            return
        sql = _UPSERT_SQL.format(table=self.table)
        params = [self._row(v) for v in self._buffer]
        # 逐条 execute 而不是 executemany：psycopg 的 executemany 在
        # ON CONFLICT 场景下不返回 rowcount，而"写了几条"是导入脚本唯一的自检。
        conn = self._connection()
        for row in params:
            conn.execute(sql, row)
        self._buffer.clear()

    def upsert(self, vector: VectorRecord) -> None:
        """按 chunk_id 幂等写入，立即持久化。"""
        self._connection().execute(_UPSERT_SQL.format(table=self.table), self._row(vector))

    # -- 增量替换（W6） ----------------------------------------------------- #

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        """读文档版本。没有记录返回 None（老数据没进注册表时也是 None）。"""
        cursor = self._connection().execute(
            f"SELECT document_id, content_hash, version, model_snapshot, chunk_count "
            f"FROM {self.table}_documents WHERE document_id = %s",
            (document_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {
            "document_id": row[0],
            "content_hash": row[1],
            "version": int(row[2]),
            "model_snapshot": row[3] or "",
            "chunk_count": int(row[4]),
        }

    def replace_document(
        self,
        document_id: str,
        records: list[VectorRecord],
        *,
        content_hash: str,
        version: int,
        model_snapshot: str,
    ) -> tuple[int, int]:
        """**同一事务**内删旧写新，返回 `(写入条数, 删除条数)`。

        为什么必须同事务：中途失败若留下半新半旧，引用会指向已删除的 chunk，
        而检索本身一切正常——这类故障只能靠翻库发现。psycopg 的
        `conn.transaction()` 在 autocommit 连接上同样有效（它会显式 BEGIN）。

        顺带把 `document_id` / `content_hash` / `model_snapshot` / `doc_version`
        写进每个 chunk 的 metadata：检索侧读不到注册表，只能靠 chunk 自带的信息
        判断"这段属于哪个版本的文档"。
        """
        if not document_id:
            raise ValueError("document_id 不能为空：空值会删掉整表")

        staged: list[tuple[VectorRecord, list[Any]]] = []
        for record in records:
            metadata = dict(record.metadata)
            metadata["document_id"] = document_id
            metadata["content_hash"] = content_hash
            metadata["model_snapshot"] = model_snapshot
            metadata["doc_version"] = version
            copy = VectorRecord(
                id=record.id,
                text=record.text,
                metadata=metadata,
                embedding=list(record.embedding),
            )
            staged.append((copy, self._row(copy)))

        upsert_sql = _UPSERT_SQL.format(table=self.table)
        doc_sql = _DOC_UPSERT_SQL.format(table=self.table)
        conn = self._connection()

        with conn.transaction():
            cursor = conn.execute(
                f"DELETE FROM {self.table} WHERE document_id = %s", (document_id,)
            )
            deleted = int(getattr(cursor, "rowcount", 0) or 0)
            for _record, row in staged:
                conn.execute(upsert_sql, row)
            conn.execute(
                doc_sql,
                (document_id, content_hash, version, model_snapshot, len(staged)),
            )

        return len(staged), deleted

    def drop(self) -> None:
        """清空表（保留结构）。索引与扩展不动，省掉重建成本。"""
        self._buffer.clear()
        self._connection().execute(
            f"TRUNCATE TABLE {self.table}; TRUNCATE TABLE {self.table}_documents"
        )

    def delete_document(self, document_id: str) -> int:
        """按文档成组删除，返回删除条数。

        不在 `VectorWriter` 契约里，但 W6 的增量索引必须有它：没有按文档删除，
        "更新一个文档"就只能全量重建，而全量重建的耗时随语料线性增长。
        """
        if not document_id:
            raise ValueError("document_id 不能为空：空值会删掉全表")
        conn = self._connection()
        with conn.transaction():
            cursor = conn.execute(
                f"DELETE FROM {self.table} WHERE document_id = %s", (document_id,)
            )
            deleted = int(getattr(cursor, "rowcount", 0) or 0)
            # 注册表一起删：留着会让"删了再重新入库"被误判成"内容没变"，
            # 于是新文档永远进不了索引（版本号还停在旧值）。
            conn.execute(
                f"DELETE FROM {self.table}_documents WHERE document_id = %s",
                (document_id,),
            )
        return deleted

    # -- 读侧 --------------------------------------------------------------- #

    def count(self) -> int:
        cursor = self._connection().execute(f"SELECT count(*) FROM {self.table}")
        row = cursor.fetchone()
        return int(row[0]) if row else 0

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        *,
        filters: Mapping[str, Any] | None = None,
    ) -> list[tuple[VectorRecord, float]]:
        """余弦相似度 Top-K。

        `score = 1 - cosine_distance`，与本地后端的点积分保持同一量纲
        （两者都是 0–1，越大越相关）。不统一量纲的话，切换后端会让
        "分数阈值"这类下游逻辑悄悄失效。

        `filters` 是**存储侧**的粗过滤（jsonb 包含），目前没有检索器转发它：
        `DenseRetriever` 只调 `search(vec, top_k)`。ACL 判定留在应用层做，
        理由见 `_filter_clause`。这个参数留给 W6 的治理/脚本按需使用。
        """
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if not query_embedding:
            return []

        where, params = self._filter_clause(filters)
        literal = to_vector_literal(list(query_embedding))
        sql = (
            f"SELECT chunk_id, text, metadata, embedding, "
            f"1 - (embedding <=> %s::vector) AS score "
            f"FROM {self.table}{where} "
            f"ORDER BY embedding <=> %s::vector LIMIT %s"
        )
        cursor = self._connection().execute(sql, [literal, *params, literal, top_k])
        return [(self._to_record(row), float(row[4])) for row in cursor.fetchall()]

    # -- 内部 --------------------------------------------------------------- #

    def _row(self, record: VectorRecord) -> list[Any]:
        if not record.embedding:
            raise ValueError(
                f"{record.id} 没有向量：pgvector 后端不能存空 embedding，"
                "BM25 索引请用 local/sparse 路径"
            )
        metadata = dict(record.metadata)
        # document_id 单独成列（删除传播与索引都要它），
        # 同时留在 jsonb 里，保证"读回来的 metadata 与写入时一致"。
        document_id = str(metadata.get("document_id") or "")
        if not document_id:
            raise ValueError(f"{record.id} 缺少 metadata['document_id']：增量删除会找不到它")
        return [
            record.id,
            document_id,
            record.text,
            json.dumps(metadata, ensure_ascii=False),
            to_vector_literal(list(record.embedding)),
            metadata.get("content_hash"),
            metadata.get("model_snapshot"),
        ]

    def _filter_clause(self, filters: Mapping[str, Any] | None) -> tuple[str, list[Any]]:
        """filters 走 jsonb 包含语义（`metadata @> {...}`）。

        只支持"包含"，不支持表达式。理由是这里承载的是**数据选择**
        （文档类型、租户、时间段），不是授权策略；权限判定需要"组集合求交"
        这类语义，放在应用层做（W6 的 ACLFilter），下推成 SQL 会让策略
        散落在两处，审计时看不清全貌。
        """
        if not filters:
            return "", []
        return " WHERE metadata @> %s::jsonb", [json.dumps(filters, ensure_ascii=False)]

    @staticmethod
    def _to_record(row: Any) -> VectorRecord:
        chunk_id, text, metadata, embedding, _score = row
        return VectorRecord(
            id=chunk_id,
            text=text,
            metadata=_as_dict(metadata),
            embedding=_as_vector(embedding),
        )


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return dict(value or {})


def _as_vector(value: Any) -> list[float]:
    """psycopg 没注册 vector 类型时，读到的是字符串 `[1,2,3]`。

    不调用 `pgvector.psycopg.register_vector` 的原因：那需要额外依赖，
    而在这里解析一行的解析成本可以忽略（每次检索只回几十行）。
    """
    if isinstance(value, str):
        return [float(item) for item in value.strip("[]").split(",") if item]
    return [float(item) for item in value]
