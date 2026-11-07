"""知识层（W5–W6）：文档 → 块 → 可引用、可治理、可增量更新的证据。

- `base.py`         数据契约 Chunk / ScoredChunk + EmbeddingProvider / VectorStore 抽象
- `citation.py`     引用溯源：编号与映射同源生成、答案 `[n]` 解析与体检（W6 D1）
- `incremental.py`  增量索引：按文档哈希跳过、同事务删旧写新（W6 D2）
- `governance.py`   知识治理：质量降权 / 过期降权 / 冲突标注（W6 D5）

纪律：本包**不 import 任何模型推理库**。embedding 实现由 apps 层注入，
否则框架包会被 torch 绑死。切分也不在这里（页感知切分依赖 PDF 解析），
由调用方以 `chunker` 回调注入 —— 见 `incremental.IncrementalIndexer.sync`。
"""

from .base import Chunk, EmbeddingProvider, ScoredChunk, VectorStore
from .citation import (
    CITATION_RULES,
    Citation,
    CitationAudit,
    CitationContext,
    CitationExpectation,
    ParsedCitations,
    audit_citations,
    build_context,
    citation_accuracy,
    citation_payload,
    group_by_document,
    missing_source_chunks,
    parse_citations,
    render_prompt,
)
from .governance import (
    DEFAULT_EXPIRY_DECAY,
    DEFAULT_TRUSTED_DOMAINS,
    GovernancePolicy,
    conflict_components,
    conflict_note,
    source_domain,
)
from .incremental import (
    DocumentStore,
    DocVersion,
    IncrementalIndexer,
    SyncAction,
    SyncResult,
    content_hash,
    model_snapshot_of,
)

__all__ = [
    "CITATION_RULES",
    "DEFAULT_EXPIRY_DECAY",
    "DEFAULT_TRUSTED_DOMAINS",
    "Chunk",
    "Citation",
    "CitationAudit",
    "CitationContext",
    "CitationExpectation",
    "DocVersion",
    "DocumentStore",
    "EmbeddingProvider",
    "GovernancePolicy",
    "IncrementalIndexer",
    "ParsedCitations",
    "ScoredChunk",
    "SyncAction",
    "SyncResult",
    "VectorStore",
    "audit_citations",
    "build_context",
    "citation_accuracy",
    "citation_payload",
    "conflict_components",
    "conflict_note",
    "content_hash",
    "group_by_document",
    "missing_source_chunks",
    "model_snapshot_of",
    "parse_citations",
    "render_prompt",
    "source_domain",
]
