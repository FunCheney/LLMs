"""pgvector 后端的契约测试（离线，靠假连接）。

没有真库也要测的部分，恰好是最容易写错的部分：

- `save()` 失败时**不能**清空缓冲（否则一次网络抖动丢整批数据）；
- `count()` 数的是持久化行，不是缓冲；
- 向量必须以字面量 + `::vector` 传入（用 repr 会传 `np.float32(...)`）；
- 表名要校验（它会被拼进 SQL）；
- 排序用余弦距离、分数要转成同类量纲。

真实连通性由 `test_pgvector_integration.py` 负责，没配 PGVECTOR_DSN 时跳过。
"""

from __future__ import annotations

import pytest

from rag_mvp.vector_store.pgvector import PgVectorStore, to_vector_literal
from rag_mvp.vector_store.vector_store_types import VectorRecord


class FakeCursor:
    def __init__(self, rows: list[tuple] | None = None, rowcount: int = 0) -> None:
        self._rows = rows or []
        self.rowcount = rowcount

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


class FakeConnection:
    """记录执行过的 SQL 与参数，按顺序返回预置结果。"""

    closed = False

    def __init__(self, results: list[FakeCursor] | None = None) -> None:
        self.statements: list[tuple[str, object]] = []
        self._results = list(results or [])
        self.close_called = False

    def execute(self, sql: str, params=None) -> FakeCursor:
        self.statements.append((sql, params))
        if self._results:
            return self._results.pop(0)
        return FakeCursor()

    def close(self) -> None:
        self.close_called = True
        self.closed = True


def _store(conn: FakeConnection, **kwargs) -> PgVectorStore:
    # ensure_schema=False：绝大多数用例关心的是读写语句，不是 DDL。
    return PgVectorStore("postgresql://x/y", connect=lambda: conn, ensure_schema=False, **kwargs)


def _record(chunk_id: str = "c1", *, dim: int = 3, **metadata) -> VectorRecord:
    meta = {"document_id": "doc1", "section_title": "调性", **metadata}
    return VectorRecord(
        id=chunk_id,
        text="品牌语气应克制",
        metadata=meta,
        embedding=[0.1] * dim,
    )


# --- 向量字面量 -------------------------------------------------------------


def test_vector_literal_is_plain_numbers() -> None:
    """必须是 `[0.1,0.2]` 这种字面量，不能出现 numpy 的 `np.float32(...)`。"""
    assert to_vector_literal([0.1, 0.2, 0.3]) == "[0.1,0.2,0.3]"


def test_vector_literal_keeps_precision_for_small_values() -> None:
    assert to_vector_literal([1e-8, 0.1234567]) == "[1e-08,0.1234567]"


# --- 构造与 schema ----------------------------------------------------------


def test_missing_dsn_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PGVECTOR_DSN", raising=False)
    with pytest.raises(ValueError, match="PGVECTOR_DSN"):
        PgVectorStore()


def test_illegal_table_name_is_rejected() -> None:
    """表名要拼进 SQL，不校验等于把 DDL/DML 交给调用方拼字符串。"""
    conn = FakeConnection()
    with pytest.raises(ValueError, match="非法表名"):
        PgVectorStore(connect=lambda: conn, table="chunks; DROP TABLE x", ensure_schema=False)


def test_ensure_schema_creates_extension_and_hnsw_index() -> None:
    conn = FakeConnection()
    PgVectorStore(connect=lambda: conn, dimension=1024)

    sql = conn.statements[0][0]
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    assert "vector(1024)" in sql
    assert "hnsw" in sql
    assert "vector_cosine_ops" in sql


# --- 写侧语义 ---------------------------------------------------------------


def test_add_only_buffers() -> None:
    """批量导入不能逐条往返：add 之后数据库里应当一条语句都没有。"""
    conn = FakeConnection()
    store = _store(conn)

    store.add(_record("c1"))
    store.add(_record("c2"))

    assert conn.statements == []


def test_save_writes_buffer_with_upsert() -> None:
    conn = FakeConnection()
    store = _store(conn)
    store.add(_record("c1"))
    store.add(_record("c2"))

    store.save()

    assert len(conn.statements) == 2
    assert all("ON CONFLICT (chunk_id) DO UPDATE" in sql for sql, _ in conn.statements)
    assert store._buffer == []


def test_save_keeps_buffer_when_write_fails() -> None:
    """失败必须保住缓冲：清空后重试就再也不知道该写什么，只能重跑整批。"""

    class BrokenConnection(FakeConnection):
        def execute(self, sql: str, params=None):
            if sql.strip().startswith("INSERT"):
                raise RuntimeError("connection reset")
            return super().execute(sql, params)

    conn = BrokenConnection()
    store = _store(conn)
    store.add(_record("c1"))

    with pytest.raises(RuntimeError):
        store.save()

    assert len(store._buffer) == 1


def test_upsert_passes_metadata_as_jsonb() -> None:
    conn = FakeConnection()
    store = _store(conn)

    store.upsert(_record("c1", content_hash="abc", model_snapshot="bge-m3@1"))

    _sql, params = conn.statements[-1]
    assert params[0] == "c1"
    assert params[1] == "doc1"
    assert '"content_hash": "abc"' in params[3]
    # embedding 与 content_hash 分列：前者给检索，后者给增量索引
    assert params[4].startswith("[")
    assert params[5] == "abc"
    assert params[6] == "bge-m3@1"


def test_record_without_document_id_is_rejected() -> None:
    """document_id 是删除传播的键，缺了它更新文档就删不掉旧 chunk。"""
    conn = FakeConnection()
    store = _store(conn)
    record = VectorRecord(id="c1", text="x", metadata={}, embedding=[0.1])

    with pytest.raises(ValueError, match="document_id"):
        store.upsert(record)


def test_record_without_embedding_is_rejected() -> None:
    conn = FakeConnection()
    store = _store(conn)
    record = VectorRecord(id="c1", text="x", metadata={"document_id": "d"}, embedding=[])

    with pytest.raises(ValueError, match="没有向量"):
        store.upsert(record)


def test_delete_document_refuses_empty_id() -> None:
    """空 document_id 会 `DELETE WHERE document_id = ''`，看起来安全，
    但一旦某次导入写入了空 document_id，就会顺手删掉它们——宁可报错。"""
    conn = FakeConnection()
    store = _store(conn)

    with pytest.raises(ValueError, match="document_id"):
        store.delete_document("")


def test_delete_document_returns_rowcount() -> None:
    conn = FakeConnection([FakeCursor(rowcount=3)])
    store = _store(conn)

    assert store.delete_document("doc1") == 3


def test_drop_truncates_instead_of_dropping_table() -> None:
    """TRUNCATE 保留结构与索引：重建 HNSW 索引比清行贵得多。"""
    conn = FakeConnection()
    store = _store(conn)

    store.drop()

    assert "TRUNCATE TABLE" in conn.statements[-1][0]


# --- 读侧语义 ---------------------------------------------------------------


def test_count_reads_from_table_not_buffer() -> None:
    conn = FakeConnection([FakeCursor([(7,)])])
    store = _store(conn)
    store.add(_record("c1"))

    assert store.count() == 7


def test_search_orders_by_cosine_distance_and_normalizes_score() -> None:
    """分数必须是 1-距离（0–1，越大越相关），与 local 后端的点积分同量纲。"""
    row = ("c1", "正文", {"document_id": "doc1"}, "[0.1,0.2,0.3]", 0.87)
    conn = FakeConnection([FakeCursor([row])])
    store = _store(conn)

    hits = store.search([0.1, 0.2, 0.3], top_k=5)

    sql, params = conn.statements[-1]
    assert "<=> %s::vector" in sql
    assert "1 - (embedding <=> %s::vector) AS score" in sql
    assert params[0] == "[0.1,0.2,0.3]"
    assert params[-1] == 5
    record, score = hits[0]
    assert record.id == "c1"
    assert record.embedding == [0.1, 0.2, 0.3]
    assert score == pytest.approx(0.87)


def test_search_with_filters_uses_jsonb_containment() -> None:
    conn = FakeConnection([FakeCursor([])])
    store = _store(conn)

    store.search([0.1], top_k=3, filters={"source_type": "pdf"})

    sql, params = conn.statements[-1]
    assert "metadata @> %s::jsonb" in sql
    assert '"source_type": "pdf"' in params[1]


def test_search_without_filters_has_no_where() -> None:
    conn = FakeConnection([FakeCursor([])])
    store = _store(conn)

    store.search([0.1], top_k=3)

    assert "WHERE" not in conn.statements[-1][0]


def test_search_rejects_empty_vector() -> None:
    """空向量去查库会得到全 0 距离，等于随机返回——必须拦住而不是查。"""
    conn = FakeConnection()
    store = _store(conn)

    assert store.search([], top_k=3) == []
    assert conn.statements == []


def test_search_rejects_non_positive_top_k() -> None:
    conn = FakeConnection()
    store = _store(conn)

    with pytest.raises(ValueError, match="top_k"):
        store.search([0.1], top_k=0)


def test_connection_is_reused_between_calls() -> None:
    """每次调用都新建连接的话，检索延迟会被握手成本主导，且连接数随 QPS 涨。"""
    conn = FakeConnection()
    calls = {"n": 0}

    def connect():
        calls["n"] += 1
        return conn

    store = PgVectorStore("postgresql://x/y", connect=connect, ensure_schema=False)
    store.count()
    store.count()

    assert calls["n"] == 1


def test_close_releases_connection() -> None:
    conn = FakeConnection()
    store = _store(conn)

    store.count()  # 懒连接：先真正用一次，才有连接可关
    store.close()

    assert conn.close_called is True


def test_close_is_idempotent_and_safe_before_connect() -> None:
    """重复 close 与"从未连过就 close"都不能抛异常。

    这两个行为在服务退出路径上很关键：`__exit__` 与 atexit 可能都会调一次，
    而 most 的请求路径根本不会打开连接（比如只调了 upsert 之前就失败了）。
    """
    conn = FakeConnection()
    store = _store(conn)

    store.close()  # 从未连接

    store.count()
    store.close()
    store.close()

    assert conn.close_called is True
