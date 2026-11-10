"""记忆层（W7）：工作记忆（会话内）+ 长期记忆（跨会话）。

- `working.py`     WorkingMemory：滚动摘要 + tool result 截断（20+ 轮不爆）
- `long_term.py`   LongTermMemory 四阶段 + 写入管线（幂等/冲突/污染校验/原子提交）
- `extract.py`     MemoryExtractor：会话结束时的结构化提取（cheap 档）
- `memory_demo.py` 教学参考实现（sqlite，独立于框架，勿在生产引用）

与知识库（knowledge/）的边界：记忆是"会话中产生、随用户演化的偏好"，
知识库是"外部导入、相对静态的事实"。两者共用 pgvector 底座但**必须**
不同表（namespace 隔离）——判据表见 docs/w7/memory-vs-knowledge.md。

本包零 ML 依赖：embedding 与 LLM 提取器都以协议/回调注入，
实现由 apps 层装配（与 knowledge/ 的同一纪律）。
"""

from .extract import (
    EXTRACT_MAX_ITEMS,
    KNOWN_KEYS,
    MemoryExtractor,
    parse_extraction,
)
from .long_term import (
    DEFAULT_TTL_S,
    MIN_CONFIDENCE,
    MemoryItem,
    MemoryRecord,
    MemoryStore,
    WriteAction,
    WriteOutcome,
    dedupe_hash_of,
    normalize_value,
)
from .working import (
    SUMMARY_MAX_TOKENS,
    CompressAction,
    CompressResult,
    SummarizerUsage,
    WorkingMemory,
    make_gateway_summarizer,
)

__all__ = [
    "DEFAULT_TTL_S",
    "EXTRACT_MAX_ITEMS",
    "KNOWN_KEYS",
    "MIN_CONFIDENCE",
    "SUMMARY_MAX_TOKENS",
    "CompressAction",
    "CompressResult",
    "SummarizerUsage",
    "MemoryExtractor",
    "MemoryItem",
    "MemoryRecord",
    "MemoryStore",
    "WorkingMemory",
    "WriteAction",
    "WriteOutcome",
    "dedupe_hash_of",
    "make_gateway_summarizer",
    "normalize_value",
    "parse_extraction",
]
