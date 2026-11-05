"""
Golden Dataset 的结构定义与加载（Phase 4 正式形式）。

与 Smoke 的区别在于「gold 标注粒度」：
- Smoke 用 any_section_title / any_content_contains 做粗粒度（章节级）命中判定；
- Golden 用 gold_chunk_ids 精确到 chunk_id（可多 gold），据此能算出
  Recall@K、MRR、nDCG 等排序质量指标。

数据集是一个 YAML 文件（data/eval/golden_queries.yaml），包含：
- corpus：语料路径与期望 chunk 数（用于校验索引与语料一致）；
- defaults：默认 top_k 与要评测的检索模式；
- queries：若干「问题 + gold chunk_id 列表」条目。

expect_insufficient=True 表示库外/无证据题（gold_chunk_ids 为空），
这类题不参与 Recall@K 等检索指标，留给生成层的拒答评测。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GoldenQuery:
    """一条 Golden 评测问题及其 gold 标注。"""
    id: str
    question: str
    # 精确到 chunk_id 的 gold 标注；一道题可对应多个 chunk（如跨页/跨 chunk 的答案）
    gold_chunk_ids: list[str] = field(default_factory=list)
    # 期望答案的核心要点（供 Phase 4 后续的答案正确率/忠实度判定使用，检索评测暂不消费）
    expected_points: list[str] = field(default_factory=list)
    # True 表示库外/无证据题：期望检索不到，gold_chunk_ids 应为空
    expect_insufficient: bool = False
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def is_negative(self) -> bool:
        """是否为负例题（无 gold、期望拒答）。"""
        return self.expect_insufficient or not self.gold_chunk_ids


@dataclass
class GoldenCorpus:
    """评测语料信息（用于一致性校验）。"""
    chunks_path: Path
    expected_chunk_count: int | None = None


@dataclass
class GoldenDefaults:
    """评测默认参数。"""
    # Recall@10 需要至少检索 10 条，默认取 10
    top_k: int = 10
    modes: list[str] = field(default_factory=lambda: ["dense", "sparse", "hybrid"])


@dataclass
class GoldenDataset:
    """完整 Golden 评测数据集。"""
    version: int
    dataset_id: str
    description: str
    corpus: GoldenCorpus | None
    defaults: GoldenDefaults
    queries: list[GoldenQuery]

    @property
    def positive_queries(self) -> list[GoldenQuery]:
        """只返回有 gold 标注的正例题（用于检索指标）。"""
        return [q for q in self.queries if not q.is_negative]


def load_golden_dataset(path: str | Path) -> GoldenDataset:
    """
    读取并解析 Golden 数据集 YAML，做基本校验后返回 GoldenDataset。

    校验：
    - 查询必须有 id 与 question；
    - 正例题（expect_insufficient=False）必须给出至少一个 gold_chunk_id；
    - 负例题（expect_insufficient=True）的 gold_chunk_ids 应留空。
    """
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    corpus_raw = data.get("corpus")
    corpus = None
    if corpus_raw:
        corpus = GoldenCorpus(
            chunks_path=Path(corpus_raw["chunks_path"]),
            expected_chunk_count=corpus_raw.get("expected_chunk_count"),
        )

    defaults_raw = data.get("defaults") or {}
    defaults = GoldenDefaults(
        top_k=int(defaults_raw.get("top_k", 10)),
        modes=list(defaults_raw.get("modes") or ["dense", "sparse", "hybrid"]),
    )

    queries: list[GoldenQuery] = []
    for item in data.get("queries") or []:
        if not item.get("id") or not item.get("question"):
            raise ValueError(f"Invalid golden query entry: {item}")

        gold_ids = [str(x) for x in (item.get("gold_chunk_ids") or [])]
        expect_insufficient = bool(item.get("expect_insufficient", False))

        # 正例题必须标注 gold，负例题不应标注 gold
        if not expect_insufficient and not gold_ids:
            raise ValueError(
                f"Golden query {item['id']} expects a hit but has empty gold_chunk_ids"
            )
        if expect_insufficient and gold_ids:
            raise ValueError(
                f"Golden query {item['id']} is negative but has gold_chunk_ids"
            )

        queries.append(
            GoldenQuery(
                id=item["id"],
                question=item["question"],
                gold_chunk_ids=gold_ids,
                expected_points=list(item.get("expected_points") or []),
                expect_insufficient=expect_insufficient,
                tags=list(item.get("tags") or []),
                notes=str(item.get("notes") or ""),
            )
        )

    return GoldenDataset(
        version=int(data.get("version", 1)),
        dataset_id=str(data.get("dataset_id", "golden")),
        description=str(data.get("description") or ""),
        corpus=corpus,
        defaults=defaults,
        queries=queries,
    )
