"""把 rag_mvp 的同步检索接到 agent_framework 的异步协议上。

三个必须解决的问题，没有一个是"加个 async def"能解决的：

1. **阻塞**：bge-m3 编码与 cross-encoder 精排都是 CPU/GPU 密集计算。在 async
   函数里直接调用，会占住事件循环——两路并发就能把 p95 拖得比同步还差。
   所以走**专用线程池**（不是默认 executor），并限流。
2. **字段同名反义**：rag_mvp 的 `RetrievalResult.source` 是检索器名，而
   agent_framework 的 `Chunk.source_uri` 是文档出处。映射时必须改名，
   否则引用会指向 `"[hybrid]"` 这种检索器名。
3. **过滤条件**：rag_mvp 不支持 metadata filter。这一条尤其不能"忽略"——
   W6 的 ACL 走的就是 filters，静默忽略等于越权。

典型用法::

    from rag_mvp.contrib.agent_framework import build_async_retriever
    from rag_mvp.retrieval.factory import RetrieverConfig

    # 启动阶段调用一次（会加载模型，数秒）
    retriever = build_async_retriever(RetrieverConfig(mode="hybrid"))

    hits = await retriever.retrieve("品牌调性怎么定", top_k=5)
    for hit in hits:
        print(hit.rank, hit.retriever, hit.chunk.source_uri, hit.score)

依赖方向：本模块依赖 agent_framework，**反向不成立**。没装 agent_framework
时，rag_mvp 的切分、索引、CLI、评测必须照常可用，因此导入是延迟的。
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    from agent_framework.knowledge import ScoredChunk

    from rag_mvp.retrieval.base import Retriever as RagRetriever
    from rag_mvp.retrieval.types import RetrievalResult

# 线程池大小。2 是刻意的：bge-m3 与 cross-encoder 都吃满 CPU，
# 线程再多也只会争 GIL。限到 2 让排队发生在池里（可观测、可告警），
# 而不是发生在 GIL 上（不可观测、表现为"莫名其妙变慢"）。
DEFAULT_MAX_WORKERS = 2

# 整链路超时。精排 20 条候选在 CPU 上通常 1–3 秒；15 秒是"明显异常"的边界。
DEFAULT_TIMEOUT_S = 15.0


class RetrievalTimeoutError(RuntimeError):
    """检索在超时前没返回。

    **不降级成空列表**：空列表在 W4 的归因里表示"链路正常但没召回"（内容问题），
    超时表示"链路坏了"（系统问题）。把超时伪装成没召回，会让人反复去补语料、
    改查询，而真正的原因是精排卡住——这类误判每次都要烧掉半天。
    """


@lru_cache(maxsize=1)
def _core_types() -> tuple[Any, Any]:
    """延迟导入 agent_framework 的契约类型。

    缓存一次是因为本函数在每条结果上都会调用；虽然 import 有内部缓存，
    多一层函数调用在批量映射里仍然可见。
    """
    try:
        from agent_framework.knowledge import Chunk, ScoredChunk
    except ImportError as exc:  # pragma: no cover - 取决于安装环境
        raise ImportError(
            "rag_mvp.contrib 需要 agent_framework。请在仓库根执行 `uv sync`，"
            "或先安装 agent-framework 再使用本模块。"
        ) from exc
    return Chunk, ScoredChunk


def _clean_page(value: int | None) -> int | None:
    """rag_mvp 用 0 表示"页码未知"（`metadata_to_document_chunk` 的兜底值）。

    core 侧用 None 表达同一件事。0 会被下游当成有效页码，写出"第 0 页"这种
    无法定位的引用，也可能参与排序/过滤，所以这里统一转成 None。
    """
    if value is None or value < 1:
        return None
    return value


def to_scored_chunk(hit: RetrievalResult, *, rank: int | None = None) -> ScoredChunk:
    """`RetrievalResult` → `ScoredChunk`。

    两处必须小心：

    - `hit.source` 是**检索器名**（dense/sparse/hybrid），映射到
      `ScoredChunk.retriever`；文档出处来自 `chunk.source_uri`。两者在
      rag_mvp 里共用一个字段名，搞反了引用就会指向检索器名。
    - `rank` 优先用调用方给的位置（列表顺序就是实际名次），其次用
      `hit.rank`，最后兜底 1。不塞 0——RRF 里 0 会让第一名拿到 1/0 的权重。
    """
    Chunk, ScoredChunk = _core_types()
    chunk = hit.chunk

    metadata = dict(hit.metadata)
    # 结构化字段优先：它们由切分层写入，比检索器附加的 metadata 更可信。
    # 用 setdefault 是为了不覆盖检索器自己写进去的信息（如 prerank_score）。
    metadata.setdefault("document_title", chunk.document_title)
    metadata.setdefault("section_title", chunk.section_title)
    metadata.setdefault("source_type", chunk.source_type)

    source_uri = (chunk.source_uri or "").strip()
    if not source_uri:
        # 不抛错：一条脏数据不该让整次检索失败。但也不能编一个看似合法的 URI，
        # 所以用显式标记 + metadata 标记，让 W6 的引用准确率把它算作失败，
        # 而不是悄悄混进"引用正确"的分子里。
        metadata["source_uri_missing"] = True
        source_uri = f"missing-source:{chunk.document_id}"

    return ScoredChunk(
        chunk=Chunk(
            chunk_id=hit.chunk_id,
            document_id=chunk.document_id,
            # hit.text 是检索时实际比对的文本，chunk.content 是落盘时的正文。
            # 二者同源（都来自 VectorRecord.text），取其一即可；优先 hit.text
            # 是因为它反映了"这次检索实际看到的内容"。
            text=hit.text or chunk.content,
            source_uri=source_uri,
            page_start=_clean_page(chunk.page_start),
            page_end=_clean_page(chunk.page_end),
            section_path=tuple(chunk.section_path),
            metadata=metadata,
        ),
        score=float(hit.score),
        retriever=hit.source,
        rank=rank if rank is not None else (hit.rank or 1),
    )


def to_scored_chunks(hits: Sequence[RetrievalResult]) -> list[ScoredChunk]:
    """批量映射，名次按列表位置重编（列表顺序即实际排序）。"""
    return [to_scored_chunk(hit, rank=i) for i, hit in enumerate(hits, start=1)]


class AsyncRetriever:
    """同步 `Retriever` 的异步门面，满足 agent_framework 的 `Retriever` 协议。

    线程池是**专用**的：asyncio 默认 executor 有 min(32, cpu+4) 个线程，
    CPU 密集任务塞进去会互抢 GIL，2 路并发就能比同步还慢。这里限到 2 个线程。

    超时的语义见 `RetrievalTimeoutError`：**抛错，不返回空列表**。
    """

    def __init__(
        self,
        inner: RagRetriever,
        *,
        name: str | None = None,
        max_workers: int = DEFAULT_MAX_WORKERS,
        timeout_s: float | None = DEFAULT_TIMEOUT_S,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self._inner = inner
        self._name = name or getattr(inner, "SOURCE", None) or type(inner).__name__
        self._timeout_s = timeout_s
        self._owns_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"rag-{self._name}",
        )

    @property
    def name(self) -> str:
        return self._name

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        # 不支持就报错，不能默默忽略：W6 的 ACL 走 filters，
        # 静默忽略等于把"只检索他有权限的文档"变成"检索全部"。
        if filters:
            raise NotImplementedError(
                "检索器层尚未转发 filters：`DenseRetriever.retrieve` 只调 "
                "`store.search(vec, top_k)`，不会把 filters 传给向量库。"
                "注意后端能力不等同于链路能力——PgVectorStore.search 已经支持 "
                "jsonb 包含过滤，但缺的是 DenseRetriever → store 这段接线。"
                "W6 实现 ACL 时先补这段接线；在此之前请用 RetrievalPipeline 的 "
                "acl_check 做应用层过滤，不要传 filters——"
                "静默忽略会让越权检索看起来像正常工作。"
            )
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        query = query.strip()
        if not query:
            # 空查询不算故障：返回空列表，与其他实现的语义一致。
            return []

        loop = asyncio.get_running_loop()
        call = functools.partial(self._inner.retrieve, query, top_k=top_k)
        try:
            raw = await asyncio.wait_for(
                loop.run_in_executor(self._executor, call),
                timeout=self._timeout_s,
            )
        except TimeoutError:
            raise RetrievalTimeoutError(
                f"{self._name} 在 {self._timeout_s}s 内未返回（query={query[:40]!r}）。"
                "注意：ThreadPoolExecutor 无法取消已提交的任务，该线程仍会跑完，"
                "所以超时只保证调用方脱身，不保证 CPU 立刻释放。"
            ) from None

        return to_scored_chunks(raw)

    def close(self) -> None:
        """释放自有线程池。外部传入的 executor 不归本对象管。"""
        if self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> AsyncRetriever:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def build_async_retriever(
    config: Any,
    *,
    max_workers: int = DEFAULT_MAX_WORKERS,
    timeout_s: float | None = DEFAULT_TIMEOUT_S,
) -> AsyncRetriever:
    """按 `RetrieverConfig` 构造异步检索器。

    **这是同步函数**：`create_retriever` 会加载 bge-m3（数秒）。请在启动阶段
    调用一次并复用，不要在请求路径里调用——否则第一个请求会吃掉整个模型加载
    时间，p95 被一次冷启动污染，之后所有延迟结论都不可信。
    """
    from rag_mvp.retrieval.factory import create_retriever

    inner = create_retriever(config)
    return AsyncRetriever(inner, max_workers=max_workers, timeout_s=timeout_s)


async def build_async_retriever_lazy(
    config: Any,
    *,
    max_workers: int = DEFAULT_MAX_WORKERS,
    timeout_s: float | None = DEFAULT_TIMEOUT_S,
) -> AsyncRetriever:
    """同 `build_async_retriever`，但把模型加载丢进线程池。

    适用于"第一个请求触发加载"的场景（如 CLI）。注意它**不会**减少总耗时，
    只是让事件循环在加载期间不被堵住。
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        functools.partial(
            build_async_retriever,
            config,
            max_workers=max_workers,
            timeout_s=timeout_s,
        ),
    )


def configure_local_threads(num_threads: int = 1) -> bool:
    """把 torch 限制为单线程，返回是否设置成功（未装 torch 时 False）。

    为什么必须做：每个工作线程内部还会开 OMP 线程。2 个工作线程 × 8 核
    = 16 个计算线程互抢缓存，吞吐反而比单线程更差。这是本地模型跑在
    异步服务里最常见的性能陷阱，且现象是"并发越高越慢"，很难归因。
    """
    try:
        import torch
    except ImportError:
        return False
    torch.set_num_threads(num_threads)
    return True


class AsyncReranker:
    """同步 `Reranker` 的异步门面，满足 agent_framework 的 `Reranker` 协议。

    为什么不复用 `RerankedRetriever`：那个包装器把「召回放大 + 精排」焊成一个
    对象，超时后调用方**拿不到**精排前的顺序，只能整个失败。生产链路要的是
    "精排超时就回退到召回顺序"，所以精排必须能被单独持有、单独放弃。

    **本类不设超时**：超时由调用方（`agent_framework.retrieval.pipeline`）按
    剩余预算施加。在这里再设一层，会出现"内层 3s、外层 800ms"的双重预算，
    日志里看到的是内层超时，而真实约束是外层——归因会跑偏。
    """

    def __init__(
        self,
        inner: Any,
        *,
        name: str | None = None,
        max_workers: int = 1,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self._inner = inner
        self._name = name or f"rerank:{type(inner).__name__}"
        self._owns_executor = executor is None
        # 精排串行即可：cross-encoder 一次前向就吃满单核，多线程只会争 GIL。
        # 并发靠"多进程/多副本"解决，不靠在这里排队。
        self._executor = executor or ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="rag-rerank",
        )

    @property
    def name(self) -> str:
        return self._name

    async def rerank(
        self,
        query: str,
        chunks: Sequence[ScoredChunk],
        *,
        top_k: int = 10,
    ) -> list[ScoredChunk]:
        if not chunks:
            return []
        loop = asyncio.get_running_loop()
        call = functools.partial(self._rerank_sync, query, list(chunks), top_k)
        return await loop.run_in_executor(self._executor, call)

    def _rerank_sync(
        self,
        query: str,
        chunks: list[ScoredChunk],
        top_k: int,
    ) -> list[ScoredChunk]:
        """一次完整的往返：ScoredChunk → RetrievalResult → 精排 → ScoredChunk。

        往返看起来绕，但它是唯一能同时满足两件事的写法：
        1. rag_mvp 的 reranker 只认自己的 `RetrievalResult`；
        2. 精排后的 `chunk` 必须原样保留（metadata/页码/来源），
           重新构造会丢掉检索器写进去的 `prerank_score`。

        所以按 `chunk_id` 回查原始 `ScoredChunk`，只替换分数与名次。
        """
        _, ScoredChunkType = _core_types()
        by_id = {scored.chunk.chunk_id: scored for scored in chunks}
        hits = [_to_retrieval_result(scored) for scored in chunks]

        ranked = self._inner.rerank(query, hits, top_k=top_k)

        out: list[ScoredChunk] = []
        for position, hit in enumerate(ranked, start=1):
            base = by_id.get(hit.chunk_id)
            if base is None:  # pragma: no cover - 精排不该凭空造出新的 chunk
                continue
            out.append(
                ScoredChunkType(
                    chunk=base.chunk,
                    score=float(hit.score),
                    # 精排改变了顺序，就不能再叫原来的 hybrid/dense：
                    # 归因时要能看出"这一条是精排抬上来的"。
                    retriever="rerank",
                    rank=position,
                )
            )
        return out

    def close(self) -> None:
        if self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> AsyncReranker:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _to_retrieval_result(scored: ScoredChunk) -> RetrievalResult:
    """`ScoredChunk` → `RetrievalResult`（精排的唯一输入形态）。

    页码是唯一需要"倒着映射"的字段：core 用 `None` 表示未知，rag_mvp 用 `0`。
    这里必须转回 0，否则 `DocumentChunk(page_start=None)` 会在下游被当成
    合法值参与 `f"{page_start}-{page_end}"` 的格式化，写出 "None-None"。
    """
    from rag_mvp.schemas import DocumentChunk

    chunk = scored.chunk
    metadata = dict(chunk.metadata)
    document = DocumentChunk(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        document_title=str(metadata.get("document_title", "")),
        section_title=str(metadata.get("section_title", "")),
        section_path=list(chunk.section_path),
        page_start=chunk.page_start or 0,
        page_end=chunk.page_end or 0,
        content=chunk.text,
        source_uri=chunk.source_uri,
        source_type=str(metadata.get("source_type", "")),
    )
    from rag_mvp.retrieval.types import RetrievalResult as Result

    return Result(
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        metadata=metadata,
        score=scored.score,
        source=scored.retriever,
        chunk=document,
        rank=scored.rank,
    )

def to_vector_record(chunk: Any, embedding: Sequence[float]) -> Any:
    """框架 `Chunk` + 向量 → rag_mvp `VectorRecord`。

    结构化字段写回 metadata 是必须的：`PgVectorStore._row` 从 metadata 里取
    `document_id`，而 `_to_record` 读回来时也只还原 metadata。少写一个字段，
    表现为"检索结果里 document_id 是空串"，删除传播会立刻失效。
    """
    from rag_mvp.vector_store.vector_store_types import VectorRecord

    metadata = dict(chunk.metadata)
    metadata.update(
        {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "source_uri": chunk.source_uri,
            "page_start": chunk.page_start or 0,
            "page_end": chunk.page_end or 0,
            "section_path": list(chunk.section_path),
        }
    )
    return VectorRecord(
        id=chunk.chunk_id,
        text=chunk.text,
        metadata=metadata,
        embedding=list(embedding),
    )


class AsyncDocumentStore:
    """`PgVectorStore` 的增量同步门面，满足 `agent_framework.knowledge.DocumentStore`。

    三件事都在这里做完，不让框架知道 pgvector 的存在：

    - **换线程**：建索引 / 写库是阻塞调用，直接在事件循环里跑会卡住所有请求；
    - **字段映射**：`Chunk` ↔ `VectorRecord`；
    - **事务**：`replace_document` 在 store 侧已经用 `conn.transaction()` 包住，
      这里只调用一次，**不拆成 delete + upsert**——拆开就失去了原子性。

    单线程执行器（`max_workers=1`）是刻意的：写库本身是串行的（同一张表、
    同一批索引），多线程只会让"第 3 个文档先写完、第 5 个后写"这种顺序问题
    变得不可复现。要并行靠多进程，不靠这里。
    """

    def __init__(
        self,
        store: Any,
        *,
        max_workers: int = 1,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self._store = store
        self._owns_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="rag-doc-store"
        )

    @property
    def name(self) -> str:
        return f"docstore:{type(self._store).__name__}"

    async def get_version(self, document_id: str) -> Any:
        """返回 `DocVersion | None`。None = 库里没有这份文档（首次入库或已删）。"""
        from agent_framework.knowledge import DocVersion

        row = await self._run(self._store.get_document, document_id)
        if row is None:
            return None
        return DocVersion(
            document_id=row["document_id"],
            content_hash=row["content_hash"],
            version=row["version"],
            model_snapshot=row["model_snapshot"],
        )

    async def replace_document(
        self,
        document_id: str,
        chunks: Sequence[Any],
        vectors: Sequence[Sequence[float]],
        *,
        version: int,
        content_hash: str,
        model_snapshot: str,
    ) -> tuple[int, int]:
        """同事务删旧写新，返回 `(写入条数, 删除条数)`。"""
        records = [
            to_vector_record(chunk, vector)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        return await self._run(
            functools.partial(
                self._store.replace_document,
                document_id,
                records,
                content_hash=content_hash,
                version=version,
                model_snapshot=model_snapshot,
            )
        )

    async def delete_document(self, document_id: str) -> int:
        return await self._run(self._store.delete_document, document_id)

    async def count(self) -> int:
        return await self._run(self._store.count)

    async def _run(self, func: Any, *args: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, functools.partial(func, *args)
        )

    def close(self) -> None:
        if self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
        close = getattr(self._store, "close", None)
        if callable(close):
            close()


def build_async_document_store(
    dsn: str | None = None,
    *,
    table: str = "rag_chunks",
    dimension: int = 1024,
) -> AsyncDocumentStore:
    """按连接串建一个增量同步用的 store。

    `ensure_schema=True` 会在启动时建表（全部 IF NOT EXISTS），所以**不需要**
    先跑 migration——但也意味着连库失败会在这里直接抛出，而不是等到第一次入库。
    这是刻意的：增量索引连不上库时，"静默跳过"会让抽取管线以为文档已经同步。
    """
    from rag_mvp.vector_store.pgvector import PgVectorStore

    store = PgVectorStore(dsn, table=table, dimension=dimension)
    return AsyncDocumentStore(store)
