"""知识资产治理（W6 D5）：质量降权 / 过期降权 / 冲突标注。

知识库真正的腐烂方式不是"召回不准"，而是**库里躺着过期和互相矛盾的内容，
而检索照样把它们排到第一**。三种腐烂各有最小对策，全部挂在 chunk 元数据上，
不新增存储、不引入二次模型打分：

| 问题 | 手段 | 关键判据 |
|---|---|---|
| 来源质量参差 | 白名单权重（官方 1.0 / 未知 0.6）+ 解析失败惩罚 | 低质量源不得排在高质量源之前 |
| 内容过期 | `valid_until < now` **降权**不丢弃 | 问"当前规定"必须新的排在前 |
| 同主题冲突 | `conflicts_with` 标记 + 进 context 时显式提示 | 答案必须指出矛盾，不自行裁决 |

三条共同的取舍：**降权而不是过滤，标注而不是裁决。**

- 丢弃过期文档会让"最新政策是什么"直接无答案，而用户看到的是一句"资料未覆盖"——
  比给一条旧政策更糟，因为它连"我有一条旧的"这个信息都丢了。降权后旧文档仍可能
  进 context，答案就能说"截至 2024 年的规定是 X"。
- 冲突裁决需要业务权威（哪份是正式版），模型猜错比暴露冲突贵得多。所以这里的
  目标是让答案**诚实标注不确定**，与 W4 "解析失败给 None 不给补分"是同一条原则。

分数语义提醒：这里的权重乘在**本次检索的分数**上。精排会把 score 换成 logit，
所以治理必须在精排之后应用（`RetrievalPipeline` 的钩子位置就是按这个排的）。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from urllib.parse import urlparse

from ..knowledge.base import Chunk, ScoredChunk

# 可信来源。白名单而不是黑名单：新出现的垃圾站点永远比我们更新黑名单快。
DEFAULT_TRUSTED_DOMAINS: tuple[str, ...] = (
    "anthropic.com",
    "modelcontextprotocol.io",
    "openai.com",
    "docs.python.org",
    "github.com",
    "developer.mozilla.org",
)

TRUSTED_WEIGHT = 1.0
UNKNOWN_WEIGHT = 0.6

# 解析失败惩罚是**乘法**，不是切换到 0：一个表格丢了的 PDF 段仍然有正文，
# 它的价值低于同源的干净段，但不等于无关。乘 0.8 让它退一名，而不是消失。
PARSE_FAILURE_PENALTY = 0.8

# 过期衰减。0.5 是"同等相关内容下，新的排前面"的量级：
# 乘 0.5 足以让一条分数接近的旧文档掉到新文档之后，又不至于让它彻底出局。
DEFAULT_EXPIRY_DECAY = 0.5

# 元数据键名集中在这里，避免各处硬编码字符串拼错（拼错不会报错，只会静默不生效）。
QUALITY_KEY = "quality"
VALID_UNTIL_KEY = "valid_until"
CONFLICTS_KEY = "conflicts_with"
PARSE_FAILURES_KEY = "parse_failures"
WEIGHT_KEY = "governance_weight"
EXPIRED_KEY = "expired"

# 负分缩放时的权重下限。权重可以算出 0（显式标 quality=0），
# 直接拿它做除数会 ZeroDivisionError，而这个块本身只是"不该排前面"，不是错误。
_MIN_WEIGHT = 0.01


def source_domain(source_uri: str) -> str:
    """从 `source_uri` 取域名；仓库内路径返回空串。

    空串代表"本地文件"，与"未知站点"分开：本地自有稿件不该按未知来源降权。
    """

    uri = (source_uri or "").strip()
    if not uri:
        return ""
    if uri.startswith("missing-source:"):
        return ""
    parsed = urlparse(uri)
    if parsed.scheme in {"http", "https"}:
        return (parsed.hostname or "").lower()
    return ""


def _domain_trusted(domain: str, trusted: Sequence[str]) -> bool:
    if not domain:
        return False
    return any(domain == item or domain.endswith("." + item) for item in trusted)


def parse_valid_until(value: object) -> datetime | None:
    """`valid_until` 解析。接受 `2026-01-31` / `2026-01-31T12:00:00Z` / datetime。

    解析不出来返回 None 而不是抛错：一条脏 metadata 会让整次检索 500，
    而治理是"尽量降权"的模块，不该有这个权力。
    """

    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


@dataclass(frozen=True)
class GovernancePolicy:
    """可调用的治理策略，直接交给 `RetrievalPipeline(governance=...)`。

    做成 dataclass 而不是一堆函数，是为了让"这批语料的治理口径"能被序列化进报告：
    A/B 跑出差异时，先看两边的策略参数是否相同。
    """

    trusted_domains: tuple[str, ...] = DEFAULT_TRUSTED_DOMAINS
    unknown_weight: float = UNKNOWN_WEIGHT
    trusted_weight: float = TRUSTED_WEIGHT
    parse_failure_penalty: float = PARSE_FAILURE_PENALTY
    expiry_decay: float = DEFAULT_EXPIRY_DECAY
    enabled: bool = True

    def quality_weight(self, chunk: Chunk) -> float:
        """来源质量权重。

        优先用入库时算好的 `metadata["quality"]`（业务方可能比白名单更清楚），
        没有才按来源白名单推断。两者都不做"模型二次打分"：
        那会把入库成本乘一个数量级，而且分数无法解释。
        """

        declared = chunk.metadata.get(QUALITY_KEY)
        if isinstance(declared, (int, float)) and not isinstance(declared, bool):
            return min(max(float(declared), 0.0), 1.0)

        domain = source_domain(chunk.source_uri)
        if not domain:
            # 本地文件：自有稿件，按可信处理。
            return self.trusted_weight
        return (
            self.trusted_weight
            if _domain_trusted(domain, self.trusted_domains)
            else self.unknown_weight
        )

    def parse_penalty(self, chunk: Chunk) -> float:
        failures = chunk.metadata.get(PARSE_FAILURES_KEY)
        if isinstance(failures, (int, float)) and not isinstance(failures, bool):
            return self.parse_failure_penalty if failures > 0 else 1.0
        return 1.0

    def is_expired(self, chunk: Chunk, *, now: datetime | None = None) -> bool:
        deadline = parse_valid_until(chunk.metadata.get(VALID_UNTIL_KEY))
        if deadline is None:
            return False
        return deadline < (now or datetime.now(UTC))

    def weight(self, chunk: Chunk, *, now: datetime | None = None) -> float:
        weight = self.quality_weight(chunk) * self.parse_penalty(chunk)
        if self.is_expired(chunk, now=now):
            weight *= self.expiry_decay
        return weight

    def apply(
        self, chunks: Sequence[ScoredChunk], *, now: datetime | None = None
    ) -> list[ScoredChunk]:
        """乘权重并重排。**名次由 pipeline 重编**，这里只保证分数已反映治理口径。"""

        if not self.enabled or not chunks:
            return list(chunks)

        adjusted: list[ScoredChunk] = []
        for scored in chunks:
            weight = self.weight(scored.chunk, now=now)
            if weight == 1.0:
                adjusted.append(scored)
                continue

            metadata = dict(scored.chunk.metadata)
            metadata[WEIGHT_KEY] = round(weight, 4)
            if self.is_expired(scored.chunk, now=now):
                # 标在 chunk 上而不是只记在分数里：答案侧要据此写"截至 X 年的规定"，
                # 而分数在组装后就看不出来了。
                metadata[EXPIRED_KEY] = True

            adjusted.append(
                ScoredChunk(
                    chunk=scored.chunk.model_copy(update={"metadata": metadata}),
                    # 负分（cross-encoder logit）不能直接乘权重：-3 × 0.6 = -1.8 会让
                    # 它**变好**，降权反而把低质量内容抬上去。所以负分走除法：
                    # -3 / 0.6 = -5，方向正确且单调。这是治理与精排同时开启时
                    # 最容易出错的一处，且现象很隐蔽（排序只是"有点怪"）。
                    score=(
                        scored.score * weight
                        if scored.score >= 0
                        else scored.score / max(weight, _MIN_WEIGHT)
                    ),
                    retriever=scored.retriever,
                    rank=scored.rank,
                )
            )

        # 稳定排序：分数相同的块保持原有相对顺序，避免同一查询两次结果不同。
        return sorted(adjusted, key=lambda item: item.score, reverse=True)

    def __call__(self, chunks: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        return self.apply(chunks)


def conflict_components(chunks: Sequence[ScoredChunk]) -> list[tuple[str, ...]]:
    """把 `conflicts_with` 关系折成连通分量（组内互相矛盾）。

    用并查集不用"两两配对列表"：三份文档互相矛盾时会形成三个配对，
    报告里看起来像三个独立问题，而实际上是同一个主题的三种说法。
    """

    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    ids = {scored.chunk.chunk_id for scored in chunks}
    for scored in chunks:
        raw = scored.chunk.metadata.get(CONFLICTS_KEY)
        targets = _as_ids(raw)
        for target in targets:
            # 只连出现在本次结果里的块：连到没进 context 的块，
            # 答案里就没法指出"和哪一段矛盾"。
            if target in ids and target != scored.chunk.chunk_id:
                union(scored.chunk.chunk_id, target)

    groups: dict[str, list[str]] = {}
    for chunk_id in list(parent):
        groups.setdefault(find(chunk_id), []).append(chunk_id)
    return [tuple(sorted(members)) for members in groups.values() if len(members) > 1]


def conflict_note(chunks: Sequence[ScoredChunk], numbers: Mapping[str, int]) -> str | None:
    """生成给模型的冲突提示。没有冲突返回 None，调用方不必拼空字符串。

    `numbers` 是 `chunk_id -> 引用编号`。**必须由调用方传入**：编号只在
    `build_context` 那一刻确定，这里自己算一遍就变成了第二个真相来源，
    提示里的编号会和正文里的编号错开。
    """

    groups = conflict_components(chunks)
    if not groups:
        return None

    lines = ["以下资料互相矛盾（同主题、说法不一致），回答时必须指出不一致，不要自行裁决："]
    for members in groups:
        refs = " ".join(f"[{numbers[member]}]" for member in members if member in numbers)
        if refs:
            lines.append(f"- {refs}")
    return "\n".join(lines) if len(lines) > 1 else None


def _as_ids(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(item) for item in value if str(item).strip()]
    return []
