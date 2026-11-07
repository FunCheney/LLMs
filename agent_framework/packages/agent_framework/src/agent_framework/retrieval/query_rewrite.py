"""查询改写（W5 D4）：multi-query 与 HyDE。

**改写不是"让 query 更漂亮"，是"让召回多一条独立证据"。** 所以两条实现都
遵守同一个取舍：宁可不改，也不能把召回改差。落实方式有三处：

1. 协议要求返回的第一条永远是**原查询**（`retrieval/base.py`），
   `RetrievalPipeline._rewrite` 还会再补一次——"改写变差"不该比"不改写"更糟；
2. 任何失败（超时、429、返回格式坏掉）都**不在这里兜**，让它冒到 pipeline：
   降级逻辑只有一份，记 `rewrite_degraded` 也只有一处。在这里 try/except 吞掉，
   日志里就永远看不到"改写一直在失败"；
3. 成本按调用累加并可读（`usage`），因为改写是**每次请求都花钱**的那一类，
   D5 的 A/B 必须能回答"这点召回提升值不值这个钱"。

模型选择：取 `catalog.routes["cheap"].primary`，与 judge 取 strong 的方式对称
（`evals/judge/llm_judge.py`）。**不在这里钉模型名**——换 cheap 档时改写要跟着换，
而硬编码的模型名不会。

温度 0.3：0 会让三次调用产出同一批 query（多查询退化成单查询），
1.0 会开始产出跑题的句子。0.3 是本机试出来的可用区间，不是理论值。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from ..llm.client import ChatCall, LLMGateway
from ..llm.types import Message, MessageRole
from ..observability import compute_cost

logger = logging.getLogger(__name__)

# 展开几条（含原查询）。3–5 是收益/噪声的平衡点：每多一条，融合后的候选池就
# 大一截，而 RRF 只认名次——候选池越大，"噪声挤掉正确答案"的机会越多。
# 本语料的全库只有 425 条 chunk，3 条已经能覆盖同义表述，再多是给 RRF 添乱。
DEFAULT_N = 3

REWRITE_TEMPERATURE = 0.3
REWRITE_MAX_TOKENS = 160

MULTI_QUERY_SYSTEM = """你是检索查询改写器。给定用户问题，产出若干条**等价的检索查询**，\
用于在向量库与关键词库里召回同一批文档。

硬性要求：
1. 只输出查询本身，每行一条，不要编号、不要解释、不要引号；
2. 每条必须能独立成立（不出现"它""这个"这类指代）；
3. 保留原问题里的专有名词、文件名、缩写（`CLAUDE.md`、`MCP` 这类不要翻译、不要改大小写）；
4. 不要输出与原问题完全相同的句子。"""

HYDE_SYSTEM = """你是检索用的"假设答案"生成器。给定用户问题，写一段**可能出现在\
权威文档里的答案段落**（80–150 字），用于按语义相似度召回真实文档。

硬性要求：
1. 只输出这段答案，不要标题、不要编号、不要"根据资料"这类套话；
2. 用陈述句写出结论与关键术语，不要分点；
3. 不确定的细节不要编造具体数字或人名——宁可写得笼统。"""

# 模型很爱加的东西：编号、项目符号、引号、前缀。全部在解析里剥掉，
# 不要靠 prompt 求它——prompt 是建议，解析是保证。
_LEADING_NOISE = re.compile(r"^\s*(?:[-*•·]|\d+[.)、]|\(\d+\)|\[\d+\])\s*")
_PREFIX = re.compile(r"^\s*(?:查询|改写|query|rewritten query)\s*[:：]\s*", re.IGNORECASE)
_WRAPPING_QUOTES = "\"'`“”‘’「」"


@dataclass(frozen=True, slots=True)
class RewriteUsage:
    """累计用量。给日志与 A/B 报告的成本列用。

    改写是**每次请求都花钱**的一类，所以成本必须能单独列出来：
    D5 的 A/B 要回答"这点召回提升值不值这个钱"，而不是只报一个总延迟。
    """

    # 实际完成的 LLM 调用次数。0 次与"调用很便宜"在报告里靠它区分，
    # 不靠 cost_per_call（后者在 0 次时也返回 0.0）。
    calls: int = 0
    # 累计输入 token（system prompt + 用户问题），按响应 usage 累加，不估算。
    input_tokens: int = 0
    # 累计输出 token（改写出的候选查询 / 假设答案）。
    output_tokens: int = 0
    # 按 catalog 价格表算出的累计成本，保留 6 位小数。
    cost: float = 0.0
    # 币种来自所选 cheap 模型的价格表；取不到模型时兜底 CNY。
    # 不同币种不能直接相加，报告里必须带着它。
    currency: str = "CNY"

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_per_call(self) -> float:
        # 0 次调用时返回 0.0 而不是除零；"没调用"与"调用很便宜"在报告里
        # 用 calls 区分，不靠这个比值区分。
        return self.cost / self.calls if self.calls else 0.0


def parse_queries(text: str, *, limit: int) -> list[str]:
    """把模型输出解析成查询列表：剥编号/引号/前缀，去空行，去重保序。

    单独一个函数是为了可单测：**改写的 bug 九成在解析上**（模型返回了编号、
    或者把三条查询写成一行），而这类 bug 在端到端测试里表现为"效果一般"，
    很难归因。
    """

    seen: set[str] = set()
    queries: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = _PREFIX.sub("", line)
        # 循环剥：模型会写 "2) - 品牌 tone 定义" 这种"编号 + 项目符号"叠加的形态，
        # 只替换一次会剩下 "- 品牌 tone 定义"，那个连字符会被当成查询的一部分。
        while True:
            stripped_line = _LEADING_NOISE.sub("", line.strip())
            if stripped_line == line:
                break
            line = stripped_line
        line = line.strip().strip(_WRAPPING_QUOTES).strip()
        if not line:
            continue

        key = line.casefold()
        if key in seen:
            continue
        seen.add(key)
        queries.append(line)

        if len(queries) >= limit:
            break

    return queries


class _GatewayRewriter:
    """改写器的公共部分：一次 LLM 调用 + 用量累加。

    子类只决定"prompt 怎么问"和"输出怎么解释"，不重复计费与错误处理。
    """

    def __init__(
        self,
        gateway: LLMGateway,
        *,
        alias: str | None = None,
        temperature: float = REWRITE_TEMPERATURE,
        max_tokens: int = REWRITE_MAX_TOKENS,
    ) -> None:
        # 生成用的网关，复用重试/fallback/计费；改写不打直连厂商 HTTP。
        self._gateway = gateway
        # 从路由表取而不是 `catalog.models[...]` 里挑一个 tier == cheap：
        # 路由表才是"谁来干便宜活"的唯一真相。
        self._alias = alias or gateway.catalog.routes["cheap"].primary
        # 采样温度 0.3：0 会让三次调用产出同一批 query（多查询退化成单查询），
        # 1.0 开始跑题。这是本机试出来的可用区间，不是理论值。
        self._temperature = temperature
        # 单次改写输出上限（默认 160 token），防止模型写成长篇解释。
        self._max_tokens = max_tokens
        # 以下四个是累计用量，随每次 _complete 递增，由 usage 属性汇总给 A/B 报告。
        self._calls = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._cost = 0.0

    @property
    def alias(self) -> str:
        return self._alias

    @property
    def usage(self) -> RewriteUsage:
        model = self._gateway.catalog.models.get(self._alias)
        return RewriteUsage(
            calls=self._calls,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            cost=round(self._cost, 6),
            currency=model.pricing.currency if model else "CNY",
        )

    async def _complete(self, system_prompt: str, user_content: str) -> str:
        """打一次模型，返回纯文本。**异常不吞**：降级由 pipeline 统一处理。"""

        call = ChatCall(
            messages=[
                Message(role=MessageRole.SYSTEM, content=system_prompt),
                Message(role=MessageRole.USER, content=user_content),
            ],
            # task 用 chat（路由到 cheap 档）；model 显式钉住解析出来的 cheap alias，
            # 避免"路由改了 task 的映射，改写悄悄跑到 strong 上烧钱"。
            task="chat",
            model=self._alias,
            opts={"temperature": self._temperature, "max_tokens": self._max_tokens},
        )
        outcome = await self._gateway.chat(call)

        self._calls += 1
        usage = outcome.response.usage
        self._input_tokens += usage.input_tokens
        self._output_tokens += usage.output_tokens
        model = self._gateway.catalog.models.get(self._alias)
        if model is not None:
            self._cost += compute_cost(model.pricing, usage)

        logger.debug(
            "rewrite.call alias=%s retry=%d in=%d out=%d estimated=%s",
            outcome.alias,
            outcome.retry_count,
            usage.input_tokens,
            usage.output_tokens,
            usage.estimated,
        )
        return _response_text(outcome.response)


class MultiQueryRewriter(_GatewayRewriter):
    """一次生成 N 条等价查询，各自召回后融合。

    **这是 W5 唯一开着的改写方式**：HyDE 同时开会让"召回变好"无法归因到哪一种。
    """

    @property
    def name(self) -> str:
        return "multi_query"

    async def rewrite(self, query: str, *, n: int = DEFAULT_N) -> list[str]:
        stripped = query.strip()
        if not stripped:
            # 空查询不该花一次调用。返回 [query] 让 pipeline 照常走（召回会返回空，
            # 那是"内容层"的正常结果，不是改写失败）。
            return [query]

        text = await self._complete(
            MULTI_QUERY_SYSTEM,
            f"用户问题：{stripped}\n\n请给出 {max(n - 1, 1)} 条等价查询。",
        )
        generated = parse_queries(text, limit=max(n - 1, 1))
        return _with_original(query, generated, limit=n)


class HydeRewriter(_GatewayRewriter):
    """HyDE：先生成"假设答案"，再用它去检索。

    注意返回的第一条仍是**原查询**——HyDE 的假设答案只作为额外的一路召回。
    只返回假设答案的话，一旦假设答案跑偏，召回会整体跑偏，而且无法和原查询
    做 RRF 兜底。这一条是协议要求，不是风格选择。
    """

    @property
    def name(self) -> str:
        return "hyde"

    async def rewrite(self, query: str, *, n: int = 2) -> list[str]:
        stripped = query.strip()
        if not stripped:
            return [query]

        text = await self._complete(HYDE_SYSTEM, f"用户问题：{stripped}")
        hypothetical = _first_paragraph(text)
        return _with_original(query, [hypothetical] if hypothetical else [], limit=max(n, 2))


def _with_original(query: str, generated: list[str], *, limit: int) -> list[str]:
    """把原查询固定在第一位，其余去重后追加，最多 `limit` 条。"""

    ordered = [query]
    seen = {query.strip().casefold()}
    for item in generated:
        key = item.strip().casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(item.strip())
        if len(ordered) >= limit:
            break
    return ordered


def _response_text(response: object) -> str:
    """从 `LLMResponse` 取正文。

    用 `getattr` 而不是直接属性访问：`LLMResponse.text` 是个 property，
    内容可能是纯字符串、内容块列表或 None（模型只返回了 tool_call）。
    这里只要正文；取不到就返回空串——空串会让解析返回空列表，
    改写退化成"只有原查询"，这是可接受且可观测的降级。
    """

    text = getattr(response, "text", None)
    return text if isinstance(text, str) else ""


def _first_paragraph(text: str) -> str:
    """取第一段非空文本，并压成一行（HyDE 的假设答案要能被 embedding 整段吃下）。"""

    for block in re.split(r"\n\s*\n", text.strip()):
        joined = " ".join(line.strip() for line in block.splitlines() if line.strip())
        if joined:
            return joined
    return ""
