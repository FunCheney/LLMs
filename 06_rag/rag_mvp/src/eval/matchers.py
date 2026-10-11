"""
命中判定：判断一条检索结果是否满足评测问题的匹配规则。

匹配是「宽松」的：
- 章节匹配会同时看 section_title 与 section_path（任一命中即可），并做归一化
  （压缩空白、转小写），避免因大小写/空格差异误判；
- 内容匹配则是子串包含（也转小写）。
"""

from retrieval.types import RetrievalResult
from eval.smoke_schema import SmokeMatchRules


def _normalize_section_title(title: str) -> str:
    """归一化标题：压缩连续空白并转小写，便于宽松比较。"""
    return " ".join(title.split()).strip().lower()


def _section_matches(hit: RetrievalResult, expected_titles: list[str]) -> bool:
    """命中结果的章节标题或路径是否命中期望标题集合。"""
    if not expected_titles:
        return False

    normalized_expected = {
        _normalize_section_title(title) for title in expected_titles
    }
    # 同时考虑当前章节标题与完整章节路径中的每一段
    hit_titles = [
        hit.chunk.section_title,
        *hit.chunk.section_path,
    ]

    for title in hit_titles:
        if _normalize_section_title(title) in normalized_expected:
            return True
    return False


def _content_matches(hit: RetrievalResult, snippets: list[str]) -> bool:
    """命中正文是否包含任一期望关键词片段。"""
    if not snippets:
        return False
    content = hit.chunk.content.lower()
    return any(snippet.lower() in content for snippet in snippets)


def hit_matches_rules(hit: RetrievalResult, rules: SmokeMatchRules) -> bool:
    """章节匹配或内容匹配任一成立即视为命中。"""
    if _section_matches(hit, rules.any_section_title):
        return True
    if _content_matches(hit, rules.any_content_contains):
        return True
    return False


def find_first_matching_rank(
    hits: list[RetrievalResult],
    rules: SmokeMatchRules,
) -> int | None:
    """返回第一个命中结果的名次（用于观察命中位置）；无命中返回 None。"""
    for hit in hits:
        if hit_matches_rules(hit, rules):
            return hit.rank
    return None
