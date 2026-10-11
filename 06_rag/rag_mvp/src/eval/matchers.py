from retrieval.types import RetrievalResult
from eval.smoke_schema import SmokeMatchRules


def _normalize_section_title(title: str) -> str:
    return " ".join(title.split()).strip().lower()


def _section_matches(hit: RetrievalResult, expected_titles: list[str]) -> bool:
    if not expected_titles:
        return False

    normalized_expected = {
        _normalize_section_title(title) for title in expected_titles
    }
    hit_titles = [
        hit.chunk.section_title,
        *hit.chunk.section_path,
    ]

    for title in hit_titles:
        if _normalize_section_title(title) in normalized_expected:
            return True
    return False


def _content_matches(hit: RetrievalResult, snippets: list[str]) -> bool:
    if not snippets:
        return False
    content = hit.chunk.content.lower()
    return any(snippet.lower() in content for snippet in snippets)


def hit_matches_rules(hit: RetrievalResult, rules: SmokeMatchRules) -> bool:
    if _section_matches(hit, rules.any_section_title):
        return True
    if _content_matches(hit, rules.any_content_contains):
        return True
    return False


def find_first_matching_rank(
    hits: list[RetrievalResult],
    rules: SmokeMatchRules,
) -> int | None:
    for hit in hits:
        if hit_matches_rules(hit, rules):
            return hit.rank
    return None
