from dataclasses import dataclass, field
from typing import Literal



RetrievalMode = Literal["dense", "sparse", "hybrid"]

@dataclass
class ProcessedQuery:
    original_query: str
    rewritten_query: str
    retrieval_mode: RetrievalMode = "hybrid"

    # 可选的元数据过滤条件。
    filters: dict[str, str] = field(
        default_factory=dict
    )
    # 复杂查询拆分或多查询检索时使用
    sub_queries: list[str] = field(
        default_factory=list
    )
