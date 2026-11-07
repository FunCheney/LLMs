"""增量索引（W6 D2）：新增/更新/删除文档**不触发全量重建**。

全量重建的问题不是"慢一点"：语料翻倍，重建时间翻倍，于是每次文档更新都要挑
凌晨跑，跑完还要重建 BM25 与向量索引，而其中 99% 的块一个字都没变。增量索引的
全部收益来自两条判断：

1. **内容哈希没变就跳过。** 不算 embedding、不写库。跳过是这里最值钱的分支：
   embedding 是整条入库链路里唯一按字符数计费的步骤。
2. **变了就"删旧 + 写新"在同一事务里。** 中途失败留下半新半旧，是最难查的一类
   故障：引用指向已被删除的 chunk，抽样时表现为"引用了不存在的段"，
   而检索本身完全正常。

三条纪律：

- **`model_snapshot` 变了，即使内容没变也要重建。** 换 embedding 模型后旧向量
  与新查询不在同一空间，检索会静默变差（不是报错）。ADR-009 要求把这件事标记出来。
- **`chunker` 由调用方注入。** 页感知切分依赖 PDF 解析，属于检索实现侧；
  框架只要求"给定文本返回 Chunk 序列"。否则这个模块会被拖进 pdfplumber/torch。
- **删干净，返回条数。** 按 `document_id` 成组删（不是按 chunk_id 逐条猜），
  返回条数供核对——`0` 条通常意味着 id 拼错，而不是"本来就没有"。
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from .base import Chunk, EmbeddingProvider

# 文档注册表在 pgvector 侧的默认表名后缀（见 PgVectorStore）。
UNKNOWN_MODEL_SNAPSHOT = "unknown"


class SyncAction(StrEnum):
    """一次同步实际做了什么。报告里按这个分组计数。"""

    CREATED = "created"
    UPDATED = "updated"
    SKIPPED = "skipped"
    DELETED = "deleted"


@dataclass(frozen=True)
class DocVersion:
    """库中已有的文档版本。`content_hash` 与 `model_snapshot` 共同决定"要不要重建"。"""

    document_id: str
    content_hash: str
    version: int = 1
    model_snapshot: str = ""


@dataclass(frozen=True)
class SyncResult:
    """同步结果。字段都能直接进报告，不需要再拼。"""

    action: SyncAction
    document_id: str
    version: int = 0
    chunks_written: int = 0
    chunks_deleted: int = 0
    elapsed_ms: float = 0.0
    # 为什么是 skipped / 为什么明明没改却重建了。空串表示"没什么可解释的"。
    note: str = ""

    @property
    def changed(self) -> bool:
        return self.action in {SyncAction.CREATED, SyncAction.UPDATED, SyncAction.DELETED}


@runtime_checkable
class DocumentStore(Protocol):
    """增量同步需要的存储能力。

    比 `VectorStore` 多两件事，都是有意的：

    - `get_version`：不做"先查有没有"就只能每次盲写，跳过分支无从实现；
    - `replace_document`：**必须在同一事务里**删旧写新。拆成 `delete` + `upsert`
      两次调用，中间失败就会留下半新半旧，而调用方拿不到回滚点。
    """

    async def get_version(self, document_id: str) -> DocVersion | None: ...

    async def replace_document(
        self,
        document_id: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        *,
        version: int,
        content_hash: str,
        model_snapshot: str,
    ) -> tuple[int, int]:
        """原子替换该文档的全部块，返回 `(写入条数, 删除条数)`。"""

    async def delete_document(self, document_id: str) -> int:
        """删除该文档的全部块，返回删除条数。"""


# 切分器契约：给定全文与文档身份，返回块序列。
# 用 Callable 而不是 Protocol：实现通常就是一个函数，包一层类只是多一个名字。
Chunker = Callable[..., Sequence[Chunk]]


def normalize_content(content: str) -> str:
    """哈希前的归一化：只统一换行与首尾空白。

    **不做 NFKC、不去标点、不折叠空格。** 归一化得越多，真实修改越容易被判成
    "没变"（例如把全角括号改成半角，在检索里是有意义的变化）。这里的目标是
    "换行符不同不算改"，仅此而已。
    """

    return content.replace("\r\n", "\n").replace("\r", "\n").strip()


def content_hash(content: str) -> str:
    return hashlib.sha256(normalize_content(content).encode("utf-8")).hexdigest()


def model_snapshot_of(embedder: EmbeddingProvider) -> str:
    """`名字:维度`。维度必须进快照：同名模型换维度会让向量库直接写不进去。"""

    return f"{embedder.name}:{embedder.dimension}"


class IncrementalIndexer:
    """按文档版本同步。

    典型用法::

        indexer = IncrementalIndexer(store, embedder)
        result = await indexer.sync("doc-1", text, chunker=chunk_pdf)
        if result.action is SyncAction.SKIPPED:
            ...  # 一个字符都没变，没有 embedding 成本
    """

    def __init__(
        self,
        store: DocumentStore,
        embedder: EmbeddingProvider,
        *,
        model_snapshot: str | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._model_snapshot = model_snapshot or model_snapshot_of(embedder)

    @property
    def model_snapshot(self) -> str:
        return self._model_snapshot

    async def sync(
        self,
        document_id: str,
        content: str,
        *,
        chunker: Chunker,
        metadata: Mapping[str, object] | None = None,
        dry_run: bool = False,
    ) -> SyncResult:
        """同步一份文档。

        `dry_run=True` 只算哈希、不读不写：验收时用它证明"重复同步一次不会重建"，
        而不用真的跑一遍 embedding（那是分钟级）。
        """

        if not document_id.strip():
            raise ValueError("document_id 不能为空：空值会让删除按整表匹配")
        if not normalize_content(content):
            # 空文档不是"更新成 0 块"，那会把已有索引清空且看起来像成功。
            raise ValueError(f"{document_id} 内容为空：空文档请显式调用 delete()")

        started = time.perf_counter()
        digest = content_hash(content)
        current = None if dry_run else await self._store.get_version(document_id)

        if current is not None and current.content_hash == digest:
            if current.model_snapshot and current.model_snapshot != self._model_snapshot:
                note = (
                    f"内容未变但 embedding 模型变了"
                    f"（{current.model_snapshot} → {self._model_snapshot}），必须重建"
                )
            else:
                return SyncResult(
                    action=SyncAction.SKIPPED,
                    document_id=document_id,
                    version=current.version,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                    note="content_hash 未变，跳过 embedding",
                )
        else:
            note = ""

        chunks = list(
            chunker(content, document_id=document_id, metadata=dict(metadata or {}))
        )
        if not chunks:
            raise ValueError(
                f"{document_id} 切分出 0 个块：拒绝写入——"
                "这会静默清空该文档已有的索引"
            )

        wrong = [chunk.chunk_id for chunk in chunks if chunk.document_id != document_id]
        if wrong:
            # 切分器把 document_id 写错时，删除传播会找不到它们：
            # 表现为"删了文档但 chunk 还在，检索照样召回"。
            raise ValueError(
                f"切分结果的 document_id 与入参不一致（前几个：{wrong[:3]}）；"
                "增量删除按 document_id 成组，写错就会漏删"
            )

        version = (current.version + 1) if current is not None else 1
        if dry_run:
            return SyncResult(
                action=SyncAction.UPDATED if current else SyncAction.CREATED,
                document_id=document_id,
                version=version,
                chunks_written=len(chunks),
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                note="dry_run：未写库",
            )

        vectors = await self._embedder.embed([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            # 条数不齐时按序配对会静默错位：第 3 个块的向量可能是第 4 段的。
            raise ValueError(
                f"embedding 条数不匹配：{len(vectors)} 个向量 / {len(chunks)} 个块"
            )

        written, removed = await self._store.replace_document(
            document_id,
            chunks,
            vectors,
            version=version,
            content_hash=digest,
            model_snapshot=self._model_snapshot,
        )

        return SyncResult(
            action=SyncAction.UPDATED if current is not None else SyncAction.CREATED,
            document_id=document_id,
            version=version,
            chunks_written=written,
            chunks_deleted=removed,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            note=note,
        )

    async def delete(self, document_id: str) -> SyncResult:
        """删除一份文档。幂等：再删一次返回 0 条，不报错。"""

        if not document_id.strip():
            raise ValueError("document_id 不能为空：空值会删掉整表")

        started = time.perf_counter()
        removed = await self._store.delete_document(document_id)
        return SyncResult(
            action=SyncAction.DELETED,
            document_id=document_id,
            chunks_deleted=removed,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            note="" if removed else "库里本来就没有这份文档（幂等）",
        )
