"""
Smoke 评测数据集的结构定义与加载（Phase 4 的早期形式）。

数据集是一个 YAML 文件（data/eval/smoke_queries.yaml），包含：
- corpus：语料路径与期望 chunk 数（用于校验索引是否与语料一致）；
- defaults：默认 top_k 与要评测的检索模式；
- queries：若干「问题 + 匹配规则」条目。

匹配规则（SmokeMatchRules）用两种方式判定「是否命中」：
- any_section_title：命中结果的章节标题/路径是否落在期望集合里；
- any_content_contains：命中正文是否包含指定的关键词片段。

expect_hit=False 表示「期望检索不到」，用于测试负例（不应误召回）。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SmokeMatchRules:
    """命中判定规则：命中章节标题 / 命中正文关键词，二者任一满足即可。"""
    any_section_title: list[str] = field(default_factory=list)
    any_content_contains: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        """规则是否为空（两条都未配置）。"""
        return not self.any_section_title and not self.any_content_contains


@dataclass
class SmokeQuery:
    """一条评测问题及其期望。"""
    id: str
    question: str
    match: SmokeMatchRules
    expect_hit: bool = True
    tags: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class SmokeCorpus:
    """评测语料信息（用于一致性校验）。"""
    chunks_path: Path
    expected_chunk_count: int | None = None


@dataclass
class SmokeDefaults:
    """评测默认参数。"""
    top_k: int = 5
    modes: list[str] = field(default_factory=lambda: ["dense", "sparse", "hybrid"])


@dataclass
class SmokeDataset:
    """完整评测数据集。"""
    version: int
    dataset_id: str
    description: str
    corpus: SmokeCorpus | None
    defaults: SmokeDefaults
    queries: list[SmokeQuery]


def _parse_match_rules(raw: dict[str, Any] | None) -> SmokeMatchRules:
    """把 YAML 中的 match 段落解析成 SmokeMatchRules。"""
    raw = raw or {}
    return SmokeMatchRules(
        any_section_title=list(raw.get("any_section_title") or []),
        any_content_contains=list(raw.get("any_content_contains") or []),
    )


def load_smoke_dataset(path: str | Path) -> SmokeDataset:
    """
    读取并解析 smoke 数据集 YAML，做基本校验后返回 SmokeDataset。

    校验：查询必须有 id 与 question；expect_hit=True 的条目必须给出非空匹配规则。
    """
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    corpus_raw = data.get("corpus")
    corpus = None
    if corpus_raw:
        corpus = SmokeCorpus(
            chunks_path=Path(corpus_raw["chunks_path"]),
            expected_chunk_count=corpus_raw.get("expected_chunk_count"),
        )

    defaults_raw = data.get("defaults") or {}
    defaults = SmokeDefaults(
        top_k=int(defaults_raw.get("top_k", 5)),
        modes=list(defaults_raw.get("modes") or ["dense", "sparse", "hybrid"]),
    )

    queries: list[SmokeQuery] = []
    for item in data.get("queries") or []:
        if not item.get("id") or not item.get("question"):
            raise ValueError(f"Invalid smoke query entry: {item}")

        match = _parse_match_rules(item.get("match"))
        expect_hit = bool(item.get("expect_hit", True))
        # 期望命中的条目必须配置匹配规则，否则无从判定
        if expect_hit and match.is_empty():
            raise ValueError(
                f"Smoke query {item['id']} expects a hit but has empty match rules"
            )

        queries.append(
            SmokeQuery(
                id=item["id"],
                question=item["question"],
                match=match,
                expect_hit=expect_hit,
                tags=list(item.get("tags") or []),
                notes=str(item.get("notes") or ""),
            )
        )

    return SmokeDataset(
        version=int(data.get("version", 1)),
        dataset_id=str(data.get("dataset_id", "smoke")),
        description=str(data.get("description") or ""),
        corpus=corpus,
        defaults=defaults,
        queries=queries,
    )
