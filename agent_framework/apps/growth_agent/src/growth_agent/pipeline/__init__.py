"""内容生成流水线（W10）：调研产物 → 结构化文章 → 多平台文案。

为什么单独成包而不是塞进 `routes_*.py`：

W10 之前产品层已经有两个端点（`/v1/draft` 写文案、`/v1/answer` 答事实），
它们各自只调一次模型。本章的 SEO 链路是**四次串行调用 + 一次检索**，
如果写在路由里，路由会同时负责"编排 / 计量 / 校验 / 序列化"四件事——
W1 的 `routes_draft.py` 之所以只有 80 行，正是靠这条纪律。

`seo.py` 与 `style.py` 的分工也刻意划在"事实"与"风格"之间：
SEO 稿要给事实标出处（错了是可信度问题），平台文案只管语域与长度
（错了是观感问题）。混在一起写，就会出现"小红书文案被要求标 [3]"这种荒诞结果。
"""

from __future__ import annotations

from .seo import (
    ArticleDraft,
    BrandProfile,
    CitationCoverage,
    EvidenceProvider,
    KeywordPlan,
    RetrievalEvidence,
    SEOError,
    SEOPipeline,
    StageTrace,
    unpublished_claims,
)
from .style import (
    PLATFORMS,
    PlatformFailure,
    PlatformSpec,
    RewriteResult,
    StyleError,
    StyleRewriter,
    as_payload,
    failure_payload,
)

__all__ = [
    "PLATFORMS",
    "ArticleDraft",
    "BrandProfile",
    "CitationCoverage",
    "EvidenceProvider",
    "KeywordPlan",
    "PlatformFailure",
    "PlatformSpec",
    "RetrievalEvidence",
    "RewriteResult",
    "SEOError",
    "SEOPipeline",
    "StageTrace",
    "StyleError",
    "StyleRewriter",
    "as_payload",
    "failure_payload",
    "unpublished_claims",
]
