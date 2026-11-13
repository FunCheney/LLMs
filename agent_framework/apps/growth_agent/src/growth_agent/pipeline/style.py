"""多平台风格化改写（W10 D2）：一稿多投，但每个平台有自己的硬约束。

## 约束为什么是数据结构而不是 prompt 里的一句话

"小红书要口语、X 要短、LinkedIn 要专业"这三句话写在 prompt 里也能用，
但接下来运营一定会提"小红书正文上限是 1000 字、X 是 280、LinkedIn 的外链
不会被降权而 X 会"这类**平台规则**。规则散在 prompt 字符串里，
改一条就得翻代码、且无法单测；写成 `PlatformSpec` 之后，
`validate()` 是纯函数，"这条规则到底有没有生效"变成一次断言。

## 生成之后为什么先跑确定性修复，再判失败

顺序是：`LLM 生成 → repair（确定性） → validate（确定性） → 不过才报错`。

三条权衡：

1. **超长不该让整条链路失败**。截断到句末是确定性的、无损语义的，
   比"再调一次模型让它缩短"便宜两个数量级，也比 500 好用。
2. **禁用词可以删**。"微信"这种词删掉不影响句子成立（它多半出现在
   导流尾巴上）；但"漏了话题标签"不能自动补一个假的——那等于伪造平台元素。
3. **只有修不了的问题才抛错**。抛错意味着这一段没有产物，
   运营要重跑；把它花在"其实能修"的问题上，会让人关掉整条校验。

## 并发为什么放在这一层

三平台改写的输入是同一篇稿子、彼此无依赖，串行跑等于 3 倍延迟。
放在 `rewrite_all` 里并发，同时用 `asyncio.Semaphore` 卡住并发数——
W2 的 `ConcurrencyLimiter` 限的是**入口请求**，这里是同一次请求内部的扇出，
不能被入口限流覆盖（一个请求打三个上游，对上游就是三倍压力）。
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Any

from agent_framework.guardrails.output import OutputGuard
from agent_framework.llm.client import ChatCall, LLMGateway
from agent_framework.llm.types import Message, MessageRole

from .seo import ArticleDraft, BrandProfile, StageTrace, text_of

__all__ = [
    "PLATFORMS",
    "PlatformFailure",
    "PlatformSpec",
    "RewriteResult",
    "StyleError",
    "StyleRewriter",
    "as_payload",
    "failure_payload",
]


class StyleError(RuntimeError):
    """改写产不出合规文案。`violations` 是给人看的违规清单。"""

    def __init__(self, platform: str, message: str, violations: tuple[str, ...] = ()) -> None:
        super().__init__(f"[{platform}] {message}")
        self.platform = platform
        self.message = message
        self.violations = violations


@dataclass(frozen=True)
class PlatformSpec:
    """一个平台的硬约束。**全部可测**，没有"看情况"。"""

    name: str
    # 正文字符上限。是平台硬限制，超了发不出去，所以要截断而不是警告。
    max_chars: int
    # 语域描述。只影响 prompt，不参与校验——"够不够专业"是判断题，不是校验项。
    tone: str
    # 必须出现的字面量（话题标签 `#`、段落分隔 `\n\n`）。
    # 空元组 = 这个平台没有硬性元素要求，不是"忘了填"。
    required_elements: tuple[str, ...] = ()
    # 禁用字面量。与品牌禁用词**叠加**（不是覆盖）：平台规则和品牌规则
    # 是两条独立的线，覆盖会让人以为"设了平台词就不用管品牌词"。
    forbidden: tuple[str, ...] = ()
    # 给 prompt 的结构提示（如"开头一句钩子"）。不参与校验。
    structure_hint: str = ""


# 三档的真实平台规则（2026-11 口径）。数字写死在这里而不是配置里：
# 它是平台属性，不是部署差异；做成配置只会让"线上为什么截断到 800 字"
# 变成一次配置排查。
PLATFORMS: dict[str, PlatformSpec] = {
    "xiaohongshu": PlatformSpec(
        name="xiaohongshu",
        max_chars=1000,
        tone="口语、第一人称、有具体细节，多用短句",
        required_elements=("#",),
        # 导流违规词：小红书的社区规范里，站外导流是被限流的主因。
        forbidden=("微信", "二维码", "私信我", "加V"),
        structure_hint="开头一句钩子（痛点或反常识），正文分 3–5 个短段，结尾一个话题标签",
    ),
    "x": PlatformSpec(
        name="x",
        max_chars=280,
        tone="结论先行、信息密度高、不用形容词堆砌",
        required_elements=(),
        # X 对外链降权，这是投放侧的真实约束（不是技术限制）。
        forbidden=("http://", "https://", "www."),
        structure_hint="一句话结论 + 一句支撑，最多一个话题标签",
    ),
    "linkedin": PlatformSpec(
        name="linkedin",
        max_chars=3000,
        tone="专业、有结构、给同行可复用的判断",
        required_elements=("\n\n",),
        forbidden=(),
        structure_hint="短段（每段 1–3 句），段间空行，结尾一个开放问题",
    ),
}


@dataclass(frozen=True)
class RewriteResult:
    """一个平台的产物 + 体检结果。"""

    platform: str
    text: str
    # 修复阶段做了什么（去掉了哪些禁用词、截断了多少字）。
    # 空元组 = 一次过，没触发修复。留这个字段是因为"三平台都过"
    # 与"三平台都靠截断才过"是两种质量，报告里必须能区分。
    repairs: tuple[str, ...] = ()
    # 最终校验仍不通过的项（非空即抛错，所以正常结果里恒为空）。
    violations: tuple[str, ...] = ()
    chars: int = 0
    trace: StageTrace | None = None


@dataclass(frozen=True)
class PlatformFailure:
    """某个平台改写失败。**是返回值，不是异常**。

    三平台扇出时，一个平台失败不该让另外两个的产物被丢掉——它们已经
    花过 token 了。所以 `rewrite_all` 用 `return_exceptions` 收口，
    把失败也变成一个可序列化的结果，由路由层决定怎么呈现。
    """

    platform: str
    error: str
    violations: tuple[str, ...] = ()


_SENTENCE_END = re.compile(r"[。！？!?；;\n]")


def _truncate_at_sentence(text: str, limit: int) -> str:
    """截到 limit 以内，尽量落在句末。找不到句末就硬截。

    硬截的代价是可能切在词中间，但比"超长被平台拒"好；
    而且 `repairs` 里会记一笔，报告能看出有多少条是硬截的。
    """

    if len(text) <= limit:
        return text

    window = text[:limit]
    cut = -1
    for match in _SENTENCE_END.finditer(window):
        cut = match.end()
    if cut > limit * 0.5:
        return window[:cut].rstrip()

    return window.rstrip()


def _strip_all(text: str, needles: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    """删掉全部禁用字面量，返回 (新文本, 实际删掉的词)。

    实现委托给 `OutputGuard.strip_forbidden`（W10 D4）：平台改写与最终
    输出守卫必须用同一套删词逻辑，否则会出现"改写阶段删了、守卫又报命中"
    这种自相矛盾的结果。
    """

    return OutputGuard.strip_forbidden(text, needles)


class StyleRewriter:
    """把一篇 `ArticleDraft` 改写成各平台文案。

    三平台共用一个 gateway 但**各有独立 prompt**：合成一个"按平台切风格"的
    大 prompt 会让三个平台的约束互相污染（实测最典型的是 LinkedIn 的段落
    结构被带进 X，导致 X 文案里出现空行）。
    """

    def __init__(
        self,
        gateway: LLMGateway,
        *,
        brand: BrandProfile | None = None,
        task: str = "chat",
        temperature: float = 0.5,
        max_concurrency: int = 3,
    ) -> None:
        self._gateway = gateway
        self._brand = brand or BrandProfile()
        self._task = task
        self._temperature = temperature
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))

    # -- 对外 -------------------------------------------------------------- #

    async def rewrite(self, article: ArticleDraft, spec: PlatformSpec) -> RewriteResult:
        prompt = self._prompt(article, spec)
        started = time.perf_counter()
        try:
            outcome = await self._gateway.chat(
                ChatCall(
                    messages=[
                        Message(role=MessageRole.USER, content=prompt),
                    ],
                    task=self._task,
                    opts={"temperature": self._temperature, "max_tokens": 900},
                )
            )
        except Exception as exc:  # noqa: BLE001 - 统一补平台上下文
            raise StyleError(spec.name, f"改写调用失败：{exc}") from exc

        raw = text_of(outcome).strip()
        if not raw:
            raise StyleError(spec.name, "改写结果为空")

        text, repairs = self.repair(raw, spec)
        violations = self.validate(text, spec)
        if violations:
            raise StyleError(spec.name, "改写结果不满足平台约束", violations)

        usage = outcome.response.usage
        return RewriteResult(
            platform=spec.name,
            text=text,
            repairs=repairs,
            chars=len(text),
            trace=StageTrace(
                name=f"style:{spec.name}",
                ms=(time.perf_counter() - started) * 1000,
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                alias=outcome.alias,
                fallback_from=outcome.fallback_from,
            ),
        )

    async def rewrite_all(
        self,
        article: ArticleDraft,
        platforms: tuple[str, ...] = ("xiaohongshu", "x", "linkedin"),
    ) -> tuple[RewriteResult | PlatformFailure, ...]:
        """并发改写。返回顺序与 `platforms` 一致（不按完成顺序，
        否则响应里三段的顺序会随机变，前端没法做 diff）。

        **单平台失败不抛，变成 `PlatformFailure` 返回**：三份产物里坏一份，
        另外两份仍然可用；整体抛错等于把两次成功调用的钱一起扔掉。
        未知平台名仍然抛——那是调用方参数错误，不该混进"生成失败"。
        """

        specs = []
        for name in platforms:
            spec = PLATFORMS.get(name)
            if spec is None:
                raise StyleError(name, f"未知平台；支持：{sorted(PLATFORMS)}")
            specs.append(spec)

        async def run(spec: PlatformSpec) -> RewriteResult | PlatformFailure:
            async with self._semaphore:
                try:
                    return await self.rewrite(article, spec)
                except StyleError as exc:
                    return PlatformFailure(
                        platform=spec.name, error=exc.message, violations=exc.violations
                    )

        return tuple(await asyncio.gather(*(run(spec) for spec in specs)))

    # -- 确定性的两道 ------------------------------------------------------ #

    def validate(self, text: str, spec: PlatformSpec) -> tuple[str, ...]:
        """纯校验，不改文本。返回违规项（空 = 通过）。

        **不调模型**：所有检查都是字面量/长度判断。这意味着它能在 CI 里
        稳定复现，也能在每次输出上零成本跑一遍。
        """

        problems: list[str] = []

        if not text.strip():
            problems.append("正文为空")
        if len(text) > spec.max_chars:
            problems.append(f"超出平台上限：{len(text)} > {spec.max_chars}")

        for element in spec.required_elements:
            if element not in text:
                shown = element.replace("\n", "\\n")
                problems.append(f"缺少必须元素：{shown!r}")

        for word in self._forbidden_for(spec):
            if word in text:
                problems.append(f"命中禁用词：{word!r}")

        return tuple(problems)

    def repair(self, text: str, spec: PlatformSpec) -> tuple[str, tuple[str, ...]]:
        """确定性修复。**只做无损的两件事**：删禁用词、超长截到句末。

        不做"补话题标签"：补一个 `#` 后面接什么都得编，而编出来的标签
        会被平台判成 spam。缺元素就让它报错，让人改 prompt。
        """

        repairs: list[str] = []

        cleaned, removed = _strip_all(text, self._forbidden_for(spec))
        if removed:
            repairs.append(f"删除禁用词：{'、'.join(removed)}")

        before = len(cleaned)
        truncated = _truncate_at_sentence(cleaned, spec.max_chars)
        if len(truncated) < before:
            # 句末截断 vs 硬截断，报告里要能区分（前者无损，后者可能切词）。
            kind = "按句末截断" if truncated and truncated[-1] in "。！？!?；;\n" else "硬截断"
            repairs.append(f"{kind}：{before} → {len(truncated)} 字")

        return truncated, tuple(repairs)

    def _forbidden_for(self, spec: PlatformSpec) -> tuple[str, ...]:
        """平台词 + 品牌词，合并去重。**顺序稳定**：平台词在前，
        因为违规处置时先看平台（平台会直接限流），品牌是内部规范。"""

        merged: list[str] = list(spec.forbidden)
        for word in self._brand.forbidden:
            if word not in merged:
                merged.append(word)
        return tuple(merged)

    def _prompt(self, article: ArticleDraft, spec: PlatformSpec) -> str:
        forbidden = self._forbidden_for(spec)
        return (
            f"把下面这篇 SEO 文章改写成 {spec.name} 平台的帖子。\n\n"
            f"语域要求：{spec.tone}\n"
            f"结构要求：{spec.structure_hint}\n"
            f"硬性上限：{spec.max_chars} 字（超过会被平台拒绝）\n"
            + (
                f"必须包含：{'、'.join(repr(e) for e in spec.required_elements)}\n"
                if spec.required_elements
                else ""
            )
            + (f"绝对不要出现：{'、'.join(forbidden)}\n" if forbidden else "")
            + f"品牌调性：{self._brand.tone}\n\n"
            "改写要求：\n"
            "1. 不要照抄原文，要按平台语域重写；\n"
            "2. 原文里的引用标号 [n] 全部去掉（平台文案不带出处）；\n"
            "3. 不要新增原文没有的事实；\n"
            "4. 只输出文案本体，不要解释、不要加“以下是”。\n\n"
            f"标题：{article.title}\n\n正文：\n{article.body}"
        )


def as_payload(result: RewriteResult) -> dict[str, Any]:
    """响应投影。手写字段而不是 `asdict`：字段一旦出去就是契约。"""

    payload: dict[str, Any] = {
        "platform": result.platform,
        "text": result.text,
        "chars": result.chars,
        "repairs": list(result.repairs),
    }
    if result.trace is not None:
        payload["ms"] = round(result.trace.ms, 1)
        payload["alias"] = result.trace.alias
    return payload


def failure_payload(failure: PlatformFailure) -> dict[str, Any]:
    """失败条目的响应投影。**必须带 violations**：只说"失败了"，
    运营没法改 prompt，只能重试——而重试同一个 prompt 会得到同一个结果。"""

    return {
        "platform": failure.platform,
        "error": failure.error,
        "violations": list(failure.violations),
    }
