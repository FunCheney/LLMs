"""输出侧过滤（W10 D4）：产物在交付前过一遍确定性检查。

三件事，全部是**纯函数**（不调模型），所以能零成本跑在每次输出上，
也能在 CI 里稳定复现：

1. **禁用词**：品牌规范 + 平台规范。
2. **引用完整性**：正文里的 `[n]` 必须都能在候选出处里找到（悬空编号），
   且带事实信号的句子不能漏标。
3. **格式与长度**：平台硬上限、必需元素。

## 引用检查为什么放在框架层而不是产品层

它看着像"内容产品的业务规则"，其实是**任何带出处的生成**都要过的一关：
`/v1/answer`（W6）、`/v1/article`（W10）、以后任何一个接检索的端点，
判据都是"编号能不能落到出处上"。放在产品层的结果是每个产品抄一份，
而三份判据必然漂移——W4 的 `RELIABILITY_CODES` 就吃过这个亏（两份码表
十个码里六个判定相反）。

所以规则在这里，产品层只决定**怎么处置**（拦断 / 降级 / 只报告）。

## 判据的取舍：句子级检测为什么带信号词

"每句话都要标号"会把"因此我们需要更重视内容"这种过渡句也判成违规，
运营会直接把这条检查关掉。所以只对**带事实信号**的句子要求标号：
数字/百分比/年份、`据…统计`、`研究显示`、`报告指出`。

这是一条**会漏**的判据（"内容营销的投资回报很高"没有信号词，但也是事实断言），
但漏检的代价是"少拦一条"，误检的代价是"这个功能被关掉"。后者更贵。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "OutputCheck",
    "OutputGuard",
    "OutputPolicy",
    "citation_markers",
    "dangling_markers",
    "split_sentences",
    "unpublished_claims",
]

# 引用标号：`[1]` / `[12]`。不认 `[1,2]` 连写——它让"第 2 个编号"在正则层面
# 无法区分"一个标号指向两段"还是"两个标号"，而 W6 的 `parse_citations` 也只认单编号。
# 两边口径必须一致，否则审计说"没引用"、流水线说"引用了"。
_MARKER = re.compile(r"\[(\d{1,2})\]")

# 事实信号词。命中即认为这句话**陈述了一个事实**，必须有标号。
# 刻意保守：宁可漏，不可误——理由见模块头。
_CLAIM_SIGNALS = re.compile(
    r"(\d+(?:\.\d+)?\s*%|\d{4}\s*年|据[^，。；]{0,16}(?:统计|显示|报告|数据)"
    r"|研究(?:显示|表明|发现)|报告(?:显示|指出|称)|数据(?:显示|表明))"
)

# 句末切分。不处理 "3.5%" 里的点：它后面不是句末标点。
_SENTENCE = re.compile(r"(?<=[。！？!?；;])\s*")


def split_sentences(body: str) -> tuple[str, ...]:
    """按句末标点切句，丢掉空串。"""

    return tuple(part.strip() for part in _SENTENCE.split(body) if part.strip())


def citation_markers(body: str) -> tuple[int, ...]:
    """正文里出现的全部编号，去重升序。

    升序是刻意的：报告里 `[1][3]` 与 `[3][1]` 应该长得一样，
    否则"编号集合不同"会被当成一次内容变化。
    """

    return tuple(sorted({int(match.group(1)) for match in _MARKER.finditer(body)}))


def dangling_markers(body: str, known: set[int]) -> tuple[int, ...]:
    """正文引了、候选出处里没有的编号。**最严重**的一类：读者点开是空的。"""

    return tuple(index for index in citation_markers(body) if index not in known)


def unpublished_claims(body: str) -> tuple[str, ...]:
    """返回"带事实信号但没有标号"的句子。空元组 = 全部有出处。

    不去重：同一句话出现两次就是两个需要修的位置，去重会让人改掉一处就以为改完了。
    """

    missing: list[str] = []
    for sentence in split_sentences(body):
        if _MARKER.search(sentence):
            continue
        if _CLAIM_SIGNALS.search(sentence):
            missing.append(sentence)
    return tuple(missing)


@dataclass(frozen=True)
class OutputPolicy:
    """一次输出要满足的约束。默认值 = "什么都不要求"，
    因为输出侧检查是**按场景加约束**的，默认严格会让所有调用点都要显式放宽。"""

    # 禁用字面量（品牌 + 平台合并后的结果）。
    forbidden: tuple[str, ...] = ()
    # 字符上限。0 = 不限。
    max_chars: int = 0
    # 必须出现的字面量。
    required_elements: tuple[str, ...] = ()
    # True = 正文至少要有一个引用标号。
    require_citations: bool = False
    # True = 带事实信号的句子必须有标号。
    strict_claims: bool = False


@dataclass(frozen=True)
class OutputCheck:
    """检查结果。`violations` 空 = 通过。"""

    violations: tuple[str, ...] = ()
    # 结构化明细，给报告与前端用（不必去正则 violations 字符串）。
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.violations

    def raise_if_blocked(self) -> None:
        if self.violations:
            raise OutputRejected(self.violations)


class OutputRejected(RuntimeError):
    """输出不满足约束。**抛出而不是记日志**：记完继续等于没有这道门。"""

    code = "CONTENT_BLOCKED"

    def __init__(self, violations: tuple[str, ...]) -> None:
        super().__init__("；".join(violations))
        self.violations = violations


class OutputGuard:
    """确定性输出过滤。**无状态**：同一输入永远同一结论，审计可复算。"""

    def check(
        self,
        text: str,
        *,
        policy: OutputPolicy | None = None,
        known_citations: set[int] | None = None,
    ) -> OutputCheck:
        """`known_citations` 为 None 时跳过引用编号的"存在性"检查，
        但仍然检查"有没有引"与"事实句有没有漏标"。"""

        policy = policy or OutputPolicy()
        violations: list[str] = []
        detail: dict[str, Any] = {}

        if not text.strip():
            violations.append("输出为空")

        if policy.max_chars and len(text) > policy.max_chars:
            violations.append(f"超出长度上限：{len(text)} > {policy.max_chars}")

        hits = tuple(word for word in policy.forbidden if word and word in text)
        if hits:
            detail["forbidden_hits"] = list(hits)
            violations.append(f"命中禁用词：{list(hits)}")

        for element in policy.required_elements:
            if element not in text:
                # 元素名里的换行要显示成 \n，否则报错信息里是两个空行，读不出是什么。
                shown = element.replace("\n", "\\n")
                violations.append(f"缺少必须元素：{shown!r}")

        markers = citation_markers(text)
        detail["markers"] = list(markers)

        if policy.require_citations and not markers:
            violations.append("没有任何引用标号")

        if known_citations is not None:
            dangling = dangling_markers(text, known_citations)
            detail["dangling"] = list(dangling)
            if dangling:
                # 悬空编号是**硬错**，与 policy 无关：引了不存在的出处，
                # 无论什么场景都是错的。
                violations.append(f"引用了不存在的编号：{list(dangling)}")

        if policy.strict_claims:
            unpublished = unpublished_claims(text)
            detail["unpublished"] = list(unpublished)
            if unpublished:
                violations.append(f"{len(unpublished)} 句事实陈述没有出处")

        return OutputCheck(violations=tuple(violations), detail=detail)

    @staticmethod
    def strip_forbidden(text: str, forbidden: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
        """删掉禁用词，返回 (新文本, 实际删掉的词)。

        只对**词级**禁用项做删除：它多半出现在导流尾巴上，删掉不影响句子成立。
        "缺话题标签"不能这么处理——补一个假的等于伪造平台元素。
        """

        removed: list[str] = []
        result = text
        for word in forbidden:
            if word and word in result:
                result = result.replace(word, "")
                removed.append(word)
        return result, tuple(removed)
