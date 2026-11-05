"""pgvector 后端的真库集成测试。

默认跳过：没有 `PGVECTOR_DSN` 时不跑。这不是"偷懒"，是因为单元测试用假连接
只能证明 SQL 拼得对，证明不了**语义**对——`<=>` 的排序方向、`ON CONFLICT`
是否真的幂等、`@>` 能否命中 jsonb，这三件事只有真库能验。

跑法：:

    docker compose -f agent_framework/ops/docker-compose.yml up -d pgvector
    PGVECTOR_DSN=postgresql://rag:rag@localhost:5433/rag \\
        uv run pytest test/test_pgvector_integration.py -v

每个用例用独立的表名（`rag_chunks_test`），跑完 DROP，不污染共享库。
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest

from rag_mvp.vector_store.pgvector import PgVectorStore
from rag_mvp.vector_store.vector_store_types import VectorRecord

DSN = os.getenv("PGVECTOR_DSN", "")
pytestmark = pytest.mark.skipif(not DSN, reason="未配置 PGVECTOR_DSN，跳过真库测试")

TABLE = "rag_chunks_itest"


@pytest.fixture()
def store() -> Iterator[PgVectorStore]:
    """建表 → 用 → 删表。异常时也要删，否则残留会干扰下一次跑。"""
    s = PgVectorStore(DSN, table=TABLE, dimension=4, ensure_schema=False)
    s.ensure_schema()
    s.drop()
    try:
        yield s
    finally:
        s._connection().execute(f"DROP TABLE IF EXISTS {TABLE}")
        s.close()


def _vec(*values: float) -> VectorRecord:
    return VectorRecord(
        id=str(uuid.uuid4()),
        text="品牌语气应克制，避免夸张的形容词堆叠。",
        metadata={"document_id": "doc-b", "tenant": "t1", "section_title": "调性"},
        embedding=list(values),
    )


def test_upsert_is_idempotent(store: PgVectorStore) -> None:
    """同一个 chunk_id 写两次，库里只能有一行——这是索引重建的前提。"""
    record = _vec(1.0, 0.0, 0.0, 0.0)

    store.upsert(record)
    store.upsert(record)

    assert store.count() == 1


def test_upsert_overwrites_content(store: PgVectorStore) -> None:
    """重写同 id 必须覆盖正文，而不是留下旧版本。"""
    record = _vec(1.0, 0.0, 0.0, 0.0)
    store.upsert(record)
    record.text = "改过的正文"
    store.upsert(record)

    hits = store.search([1.0, 0.0, 0.0, 0.0], top_k=1)

    assert hits[0][0].text == "改过的正文"


def test_search_orders_by_similarity_desc(store: PgVectorStore) -> None:
    """`<=>` 是余弦**距离**，必须转成 `1 - 距离` 才会"越大越相关"。

    写反了的现象是：检索结果永远是最不相关的几条，而且分数还是正数，
    看起来像"模型不准"，很难归因。
    """
    near = _vec(1.0, 0.0, 0.0, 0.0)
    far = _vec(0.0, 0.0, 0.0, 1.0)
    store.upsert(far)
    store.upsert(near)

    hits = store.search([1.0, 0.0, 0.0, 0.0], top_k=2)

    assert [h[0].id for h in hits] == [near.id, far.id]
    assert hits[0][1] > hits[1][1]
    assert hits[0][1] == pytest.approx(1.0, abs=1e-6)


def test_add_then_save_is_buffered(store: PgVectorStore) -> None:
    """add 不落库，save 才落库——否则批量导入会退化成 N 次网络往返。"""
    store.add(_vec(1.0, 0.0, 0.0, 0.0))
    store.add(_vec(0.0, 1.0, 0.0, 0.0))

    assert store.count() == 0
    store.save()
    assert store.count() == 2


def test_save_failure_keeps_buffer() -> None:
    """落库失败时缓冲必须留着，否则一次网络抖动丢掉整批数据。

    构造手法：第一次 execute 抛错，验证缓冲未清空。
    """
    store = PgVectorStore(DSN, table=TABLE, dimension=4, ensure_schema=False)
    store.add(_vec(1.0, 0.0, 0.0, 0.0))

    class Boom(Exception):
        pass

    original = store._connection

    def broken() -> object:
        class C:
            def execute(self, *args, **kwargs):
                raise Boom("模拟写库失败")

        return C()

    store._connection = broken  # type: ignore[method-assign]
    with pytest.raises(Boom):
        store.save()
    store._connection = original  # type: ignore[method-assign]

    assert len(store._buffer) == 1, "失败后缓冲被清空了：这批数据会丢"


def test_delete_document_removes_only_that_doc(store: PgVectorStore) -> None:
    """按文档删除是增量索引的底层能力，删错范围会波及其它文档。"""
    a = _vec(1.0, 0.0, 0.0, 0.0)
    a.metadata["document_id"] = "doc-a"
    b = _vec(0.0, 1.0, 0.0, 0.0)
    b.metadata["document_id"] = "doc-b"
    store.upsert(a)
    store.upsert(b)

    deleted = store.delete_document("doc-a")

    assert deleted == 1
    assert store.count() == 1
    remaining = store.search([0.0, 1.0, 0.0, 0.0], top_k=5)
    assert [h[0].id for h in remaining] == [b.id]


def test_delete_document_rejects_empty_id(store: PgVectorStore) -> None:
    """空 document_id 会拼出 `WHERE document_id = ''`——那删的是"所有没写 id 的行"。
    这里选择直接报错，因为静默删除是更坏的默认值。"""
    store.upsert(_vec(1.0, 0.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="document_id"):
        store.delete_document("")


def test_metadata_filter_uses_jsonb_containment(store: PgVectorStore) -> None:
    """`@>` 是"包含"语义：传多字段要全部命中，不是任一命中。"""
    t1 = _vec(1.0, 0.0, 0.0, 0.0)
    t2 = _vec(1.0, 0.0, 0.0, 0.0)
    t2.metadata["tenant"] = "t2"
    store.upsert(t1)
    store.upsert(t2)

    hits = store.search([1.0, 0.0, 0.0, 0.0], top_k=10, filters={"tenant": "t1"})

    assert len(hits) == 1
    assert hits[0][0].metadata["tenant"] == "t1"


def test_metadata_roundtrip_preserves_extra_fields(store: PgVectorStore) -> None:
    """读回来的 metadata 必须与写入时一致，否则 W6 的治理字段（quality /
    valid_until / acl）在检索后就丢了。"""
    record = _vec(1.0, 0.0, 0.0, 0.0)
    record.metadata["quality"] = 0.8
    record.metadata["acl"] = ["group:brand"]
    store.upsert(record)

    hits = store.search([1.0, 0.0, 0.0, 0.0], top_k=1)

    assert hits[0][0].metadata["quality"] == 0.8
    assert hits[0][0].metadata["acl"] == ["group:brand"]
    assert hits[0][0].metadata["section_title"] == "调性"


def test_empty_embedding_is_rejected(store: PgVectorStore) -> None:
    """pgvector 列不接受空向量；静默跳过会让"导入了 100 条实际只有 60 条"。"""
    record = _vec()
    record.embedding = []

    with pytest.raises(ValueError, match="embedding"):
        store.upsert(record)


def test_drop_clears_data_but_keeps_table(store: PgVectorStore) -> None:
    store.upsert(_vec(1.0, 0.0, 0.0, 0.0))
    store.drop()

    assert store.count() == 0
    store.upsert(_vec(0.0, 1.0, 0.0, 0.0))  # 表结构还在，能继续写
    assert store.count() == 1
