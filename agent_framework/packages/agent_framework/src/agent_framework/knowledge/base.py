"""知识层的数据契约与抽象（W5 D1）。

三条边界，都是"踩过一次才知道"的：

- **Chunk 是唯一的知识单元**：切分、索引、检索、引用都围绕它。任何"检索结果"
  必须是 Chunk + 分数，而不是裸字符串——否则 W6 的引用准确率无法回溯到具体
  文档与页码，只能得到"看起来有引用"的假指标。
- **`source_uri` 与 `retriever` 必须分开**：前者是文档出处，后者是谁召回的。
  rag_mvp 里两者共用了 `source` 一个名字，直接搬过来会让引用指向 `"[hybrid]"`。
- **本模块零第三方依赖**（只有 pydantic 与标准库）。embedding 模型、向量库
  客户端一律以 Protocol 出现，实现由 apps 层注入——framework 包不能因为要接
  RAG 就背上 torch（那会让任何引用框架的项目都被迫装 2GB）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Chunk(BaseModel):
    """一个可检索、可引用的知识单元。"""

    model_config = ConfigDict(frozen=True)

    # 全局唯一，形如 `<doc_id>-p001-p002-c003`。把定位信息编进 id 是刻意的：
    # 排序错乱时能一眼看出它属于哪一页，不用回查索引。
    chunk_id: str

    # 所属文档。增量索引的删除传播按它成组操作（W6），不按 chunk_id 逐条删。
    document_id: str

    text: str

    # 文档出处：URL 或仓库内相对路径。注意**不是**检索器名（见 ScoredChunk.retriever）。
    source_uri: str

    # 起止页（1-based）。跨页块的 page_start != page_end，
    # 引用才能写成"第 3–4 页"而不是只给一个页码。
    page_start: int | None = None
    page_end: int | None = None

    # 章节路径，如 ("第三章", "3.2 检索")。给引用补语境，避免"某文档第 5 页"
    # 这种读者无法定位的引用。
    section_path: tuple[str, ...] = ()

    # 后端与治理字段：quality / valid_until / conflicts_with / acl（W6 填充）。
    # 用自由 dict 而不是固定字段，是因为治理维度会随语料变化，
    # 每加一个维度就改数据类会让迁移成本失控。
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("chunk_id", "document_id", "source_uri")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        # 空值不是"暂时没填"，是"引用会指向空字符串"：删除传播找不到目标，
        # 抽样核对时也无法定位。宁可在这里炸，也不要在 W6 的指标里发现。
        if not value.strip():
            raise ValueError("不能为空")
        return value

    @field_validator("page_end")
    @classmethod
    def _page_order(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("页码从 1 开始")
        return value


class ScoredChunk(BaseModel):
    """检索结果：Chunk + 分数 + 是谁召回的。

    **分数不可跨检索器比较。** dense 是余弦相似度（0–1），sparse 是 BM25
    原始分（无上界，且随语料规模与平均文档长度浮动），rerank 是
    cross-encoder logit。把三者混在一起排序，等于让语料规模决定排序结果。

    这正是 RRF 只用名次、不用分数的原因：名次在所有检索器里同义，分数不是。
    需要跨检索器融合时，先按 `retriever` 分组，再交给融合层。
    """

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float

    # 谁召回的：dense / sparse / hybrid / rerank 或自定义实现的名字。
    # 必填而不是默认值——同一个 chunk 被两个检索器召回时，
    # 只有分开记录才能看出"混合召回是否真的贡献了新命中"。
    retriever: str

    # 1-based 名次。RRF 直接用它，不用 score。
    rank: int = Field(ge=1)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """文本 → 向量。

    `async` 是刻意的：本地模型（bge-m3）是 CPU/GPU 阻塞计算。协议只保证
    "可以 await"，把计算丢进哪个线程池由实现决定——调用方不该知道实现是
    本地模型还是远程 API，否则换 embedding 供应商要改整条调用链。
    """

    @property
    def name(self) -> str: ...

    @property
    def dimension(self) -> int:
        """向量维度。建索引和校验时必须用它，不能靠 len(vectors[0]) 猜——
        空列表时那样会直接 IndexError，而错误信息指向一个无关的地方。"""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@runtime_checkable
class VectorStore(Protocol):
    """向量库抽象。三个方法就够，多一个都别加。

    `delete` 必须存在：W6 的增量索引是"同事务删旧写新"，没有 delete 就只能
    全量重建，而全量重建的耗时随语料线性增长，等于把增量索引的意义抹掉。
    """

    @property
    def name(self) -> str: ...

    async def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """写入或覆盖，返回实际写入条数。"""

    async def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """按向量召回。filters 承载 ACL 与治理条件（W6），现在可以先忽略。"""

    async def delete(self, *, document_id: str) -> int:
        """按文档成组删除，返回删除条数。"""
