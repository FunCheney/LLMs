"""查询改写的边界用例（W5 D4 / Gate「成本可查 + 失败降级」）。

改写器最容易出的错不是"调用失败"，而是：

1. **解析错**——模型返回了编号或把三条写成一行，于是三条查询变成一条带着
   "1." 的怪字符串，检索结果变差但没人报错；
2. **忘了原查询**——只返回改写结果，于是"改写比不改写差"的情况下全盘倒退；
3. **成本算不出来**——改写是每次请求都花钱的环节，D5 的 A/B 要回答
   "这点召回提升值不值这个钱"，没有 token/成本就没法回答。

这一组用**真 catalog + 夹具 client**：路由/取价走真实 `models.yaml`
（所以 `routes["cheap"].primary` 解析错会立刻红），网络请求走离线夹具。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

# `offline_fixtures` 与 `evals` 都靠仓库根入 path
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "evals" / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "evals" / "scripts"))

from agent_framework.llm.client import ChatCall  # noqa: E402
from agent_framework.retrieval.query_rewrite import (  # noqa: E402
    DEFAULT_N,
    HydeRewriter,
    MultiQueryRewriter,
    parse_queries,
)
from offline_fixtures import build_offline_gateway  # noqa: E402


@pytest.fixture()
def gateway():
    # provider=openai 复用 OpenAICompat 的解析逻辑；case=text_basic 是固定答复。
    return build_offline_gateway(provider="openai", fixture_case="text_basic")


# --------------------------------------------------------------------------- #
# 解析：改写的 bug 九成在这里
# --------------------------------------------------------------------------- #


def test_parse_strips_numbering_and_bullets() -> None:
    text = '1. "品牌调性怎么定"\n2) - 品牌 tone 定义\n- brand voice 是什么'
    assert parse_queries(text, limit=10) == [
        "品牌调性怎么定",
        "品牌 tone 定义",
        "brand voice 是什么",
    ]


def test_parse_strips_label_prefix() -> None:
    # 模型很爱写「查询：xxx」。前缀留在字符串里会变成一个语义完全不同的 query。
    assert parse_queries("查询：MCP 是什么\nQuery: MCP 协议", limit=5) == [
        "MCP 是什么",
        "MCP 协议",
    ]


def test_parse_dedupes_case_insensitively() -> None:
    text = "MCP 是什么\nmcp 是什么\nMCP是什么"
    # 前两条只差大小写 → 后一条留下（去空白后确实不同），但不该出现两条"MCP 是什么"
    parsed = parse_queries(text, limit=5)
    assert parsed[0] == "MCP 是什么"
    assert len(parsed) == 2


def test_parse_respects_limit() -> None:
    text = "\n".join(f"查询{i}" for i in range(10))
    assert len(parse_queries(text, limit=3)) == 3


def test_parse_empty_text_is_empty_list() -> None:
    # 空输出不是错误：调用方会退化成"只有原查询"，那是可用的。
    assert parse_queries("   \n\n  ", limit=3) == []


def test_parse_keeps_identifier_case_and_punctuation() -> None:
    # 专有名词被"美化"是隐性 bug：`CLAUDE.md` 变成 `claude.md` 后 BM25 直接不命中。
    assert parse_queries("CLAUDE.md 和 Memory 有什么区别？", limit=3) == [
        "CLAUDE.md 和 Memory 有什么区别？"
    ]


# --------------------------------------------------------------------------- #
# multi-query：原查询必须排第一
# --------------------------------------------------------------------------- #


async def test_multi_query_always_puts_original_first(gateway) -> None:
    rewriter = MultiQueryRewriter(gateway)
    result = await rewriter.rewrite("MCP 是什么", n=3)

    assert result[0] == "MCP 是什么"
    assert len(result) <= 3


async def test_multi_query_returns_original_on_empty_query_without_calling_model(
    gateway,
) -> None:
    rewriter = MultiQueryRewriter(gateway)
    result = await rewriter.rewrite("   ", n=3)

    assert result == ["   "]
    # 空查询花一次调用是纯浪费，而且会让成本均值被无意义的调用拉偏
    assert rewriter.usage.calls == 0


async def test_multi_query_counts_usage_and_cost(gateway) -> None:
    rewriter = MultiQueryRewriter(gateway)
    await rewriter.rewrite("MCP 是什么", n=3)

    usage = rewriter.usage
    assert usage.calls == 1
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0
    assert usage.tokens == usage.input_tokens + usage.output_tokens
    # 夹具里 qwen3.7-plus 是 CNY 计价；能算出来说明取价链路通了
    assert usage.currency == "CNY"
    assert usage.cost > 0
    assert usage.cost_per_call == pytest.approx(usage.cost)


async def test_usage_accumulates_across_calls(gateway) -> None:
    rewriter = MultiQueryRewriter(gateway)
    await rewriter.rewrite("第一问", n=3)
    await rewriter.rewrite("第二问", n=3)

    assert rewriter.usage.calls == 2
    assert rewriter.usage.cost > 0


def test_rewriter_uses_cheap_alias_from_catalog(gateway) -> None:
    """别名取 `routes["cheap"].primary`，不在代码里钉模型名。

    钉死模型名的话，换 cheap 档时改写会继续跑旧模型；而"改写跑在 strong 上"
    这件事在报告里只表现为成本涨了 10 倍，看不出原因。
    """

    rewriter = MultiQueryRewriter(gateway)
    assert rewriter.alias == gateway.catalog.routes["cheap"].primary


def test_rewriter_accepts_explicit_alias(gateway) -> None:
    rewriter = MultiQueryRewriter(gateway, alias="gpt-4o-mini")
    assert rewriter.alias == "gpt-4o-mini"
    # 显式别名也要能从 catalog 取到价，否则成本会静默变 0
    assert gateway.catalog.models["gpt-4o-mini"].pricing.currency == "USD"


async def test_model_failure_propagates_to_pipeline(gateway) -> None:
    """改写器**不吞异常**：降级只有 pipeline 一处会做。

    在这里 try/except 兜住的话，`rewrite_degraded` 永远记不上，
    "改写一直在失败"会彻底不可见。
    """

    class _Boom:
        catalog = gateway.catalog

        async def chat(self, call: ChatCall) -> None:
            raise RuntimeError("upstream 500")

    rewriter = MultiQueryRewriter(_Boom())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        await rewriter.rewrite("MCP 是什么")


# --------------------------------------------------------------------------- #
# HyDE
# --------------------------------------------------------------------------- #


async def test_hyde_keeps_original_as_first_query(gateway) -> None:
    """HyDE 也必须把原查询放第一位。

    只返回"假设答案"的话，一旦假设答案跑偏，召回整体跑偏且没有兜底——
    这和 multi-query 是同一个协议要求，不是风格选择。
    """

    rewriter = HydeRewriter(gateway)
    result = await rewriter.rewrite("MCP 是什么")

    assert result[0] == "MCP 是什么"
    assert rewriter.name == "hyde"


async def test_hyde_returns_at_most_two_entries(gateway) -> None:
    """原查询 + 一段假设答案。假设答案再拆成多段就变成 multi-query 了。"""

    rewriter = HydeRewriter(gateway)
    assert len(await rewriter.rewrite("MCP 是什么")) <= 2


def test_default_n_is_small() -> None:
    """展开条数不能随手加大：每多一条，RRF 的候选池就大一截。

    RRF 只认名次，池子越大，"噪声把正确答案挤出 top-k"的机会越多——本周
    混合检索的负收益就是这么来的。
    """

    assert 2 <= DEFAULT_N <= 5
