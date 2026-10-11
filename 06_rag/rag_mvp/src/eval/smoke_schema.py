from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SmokeMatchRules:
    any_section_title: list[str] = field(default_factory=list)
    any_content_contains: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.any_section_title and not self.any_content_contains


@dataclass
class SmokeQuery:
    id: str
    question: str
    match: SmokeMatchRules
    expect_hit: bool = True
    tags: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class SmokeCorpus:
    chunks_path: Path
    expected_chunk_count: int | None = None


@dataclass
class SmokeDefaults:
    top_k: int = 5
    modes: list[str] = field(default_factory=lambda: ["dense", "sparse", "hybrid"])


@dataclass
class SmokeDataset:
    version: int
    dataset_id: str
    description: str
    corpus: SmokeCorpus | None
    defaults: SmokeDefaults
    queries: list[SmokeQuery]


def _parse_match_rules(raw: dict[str, Any] | None) -> SmokeMatchRules:
    raw = raw or {}
    return SmokeMatchRules(
        any_section_title=list(raw.get("any_section_title") or []),
        any_content_contains=list(raw.get("any_content_contains") or []),
    )


def load_smoke_dataset(path: str | Path) -> SmokeDataset:
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
