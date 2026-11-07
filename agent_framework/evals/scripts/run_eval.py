"""W3 D3 · golden 批量评测 runner（`make eval` 的执行体）。

它和 W2 的 `run_smoke.py` 的分工
------------------------------
- `run_smoke`：10 条，回答"**没坏吗**"，进每次提交的 CI，必须 3 秒内跑完；
- `run_eval`：54 条，回答"**多好**"，产出分层指标 + 报告，进周度合流。

六条纪律（每条都对应一次真实踩坑，见 `docs/w3/adr-w3-eval-scope.md`）
------------------------------------------------------------------
1. **钉 alias + 温度 0**：`ChatCall.model` 恒取 case 里的 alias，`opts["temperature"]` 强制 0。
   不钉就是不可复现，两次跑出两个数，趋势图失去意义。
2. **指标只从 `RequestLog` 读**（`LogCapture`）：自己 `time.monotonic()` 会造出第二套
   `first_token_ms`，W11 的曲线会在某天无故断层。
3. **单条失败不中断整批**：`LLMError` → `interrupted`，其余异常 → `failed`，
   继续下一条。一条 502 不该毁掉整晚评测。
4. **离线档只断言"执行管道"**：用 case 的 `offline.expect`，不用 `expect` 的语义判据。
   两套判据混用，要么假绿要么红一片（ADR-W3-01 D1）。
5. **逐条串行**：指标从共享的日志 handler 读，并发跑会让 `latest()` 串到别人的记录上。
   要测并发请用 `locustfile.py`，它是另一个问题。
6. **离线结果不进 `quality` 块**：只写 `pipeline`。离线跑通的 100% 不等于"系统很好"。
7. **judge 默认关闭**（W4 D1）：一次 judge 调用比被测模型本身还贵。开 `--judge` 才打分，
   且报告里单列 `judge_alias` 与调用数——分数由谁打的，不能被省略。
8. **RAGAS 只对带 references 的 case 算**（W4 D2）：没引用的 case 返回 `None`，
   不进均值。把"算不了"当 0 会让 faithfulness 随数据集构成漂移。

用法
----
    uv run python evals/scripts/run_eval.py                    # 离线档（默认，零网络）
    uv run python evals/scripts/run_eval.py --live             # 在线档（需 .env.local）
    uv run python evals/scripts/run_eval.py --only gold-c01 -v
    uv run python evals/scripts/run_eval.py --break-case gold-c03   # 反向验证"不中断"
    uv run python evals/scripts/run_eval.py --judge            # 顺带跑 judge（离线档=规则基线）
    uv run python evals/scripts/run_eval.py --no-write         # 只看输出，不落盘

退出码：0 全过 / 1 有失败或中断 / 2 环境不满足。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_framework.config import ConfigError, Settings, load_catalog  # noqa: E402
from agent_framework.llm.client import ChatCall, LLMGateway  # noqa: E402
from agent_framework.llm.errors import (  # noqa: E402
    LLMError,
    error_code_for,
    http_status_for,
)
from agent_framework.llm.types import (  # noqa: E402
    Message,
    MessageRole,
    StreamEventType,
    ToolSpec,
)
from agent_framework.observability import get_request_logger  # noqa: E402
from bench_baseline import LogCapture  # noqa: E402
from classify_failures import (  # noqa: E402
    FailureCategory,
    classify,
    is_quality_failure,
    is_reliability_code,
    summarize,
)
from offline_fixtures import OFFLINE_ALIAS, build_offline_gateway  # noqa: E402

from evals.judge.llm_judge import (  # noqa: E402
    Judge,
    JudgeCase,
    build_judge,
    mean_overall,
    summarize_tokens,
)
from evals.judge.ragas_metrics import score_case  # noqa: E402
from evals.judge.rubric import DEFAULT_RUBRIC, JudgeResult  # noqa: E402
from evals.metrics import (  # noqa: E402
    CaseMetrics,
    CaseStatus,
    EvalResult,
    RagasScores,
    TaskMetrics,
    aggregate,
    by_difficulty,
    by_kind,
    by_tag,
)

GOLDEN_PATH = REPO_ROOT / "evals" / "golden" / "golden_v1.jsonl"
HISTORY_DIR = REPO_ROOT / "evals" / "history"

# 工具符号名 → 真实 ToolSpec。`validate_golden.py` 用同一张表校验 tool_name 是否在声明内。
TOOL_SPECS: dict[str, ToolSpec] = {
    "search_web": ToolSpec(
        name="search_web",
        description="检索网页，返回标题与摘要",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "检索词"}},
            "required": ["query"],
        },
    ),
    "fetch_doc": ToolSpec(
        name="fetch_doc",
        description="按文档 id 取仓库内文档正文",
        parameters={
            "type": "object",
            "properties": {"doc_id": {"type": "string", "description": "文档 id"}},
            "required": ["doc_id"],
        },
    ),
}

TOOL_SYMBOLS: dict[str, tuple[str, ...]] = {
    "search_or_fetch": ("search_web", "fetch_doc"),
}

# 哪些错误码算"链路没跑通"（→ interrupted，不进成功率分母）由分类器单点决定，
# 这里不再维护第二份表（W4 D5 案例 3：两份表曾让 6/10 个码判定不一致）。


# --------------------------------------------------------------------------- #
# 用例
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class GoldenCase:
    """一条 golden 用例，从 jsonl 的一行还原而来。字段名与 `evals/golden/schema.md` 对齐。"""

    # 用例标识，全局唯一。趋势对比、失败回查、judge 标注都靠它对上同一条。
    id: str

    # 调用形态：chat / stream / tool_call / error。延迟与成功率按它分组看。
    kind: str

    # 难度 1/2/3（1=单轮直答，2=需一次检索或工具，3=多步或多源引用）。
    difficulty: int

    # 自由标签（research / citation / adversarial ...），用于切片统计。
    tags: tuple[str, ...]

    # 路由任务名（对应 `models.yaml` 的 `tasks`），决定走哪条路由档位。
    task: str

    # 离线档使用的 alias（通常是夹具模型）。**必须钉死**，否则两次跑不可比。
    model: str

    # 在线档使用的 alias。离线与在线分开是因为同一 task 在两档下模型不同，
    # 混成一个字段就必然有一档是错的。
    online_model: str

    # 送进 gateway 的完整消息序列。角色与内容一起还原，不在 runner 里拼 prompt。
    messages: list[Message]

    # 断言集合（text_contains / must_not_include / finish_reason / tool_calls ...）。
    # 原样保留 dict 而不建强类型：判据种类会随周增加，建类型等于每周改一次结构。
    expect: dict[str, Any]

    # 离线档的夹具段落（`fixture` + `expect`）。为 None 表示这条只能在线跑。
    offline: dict[str, Any] | None = None

    # 工具声明。为 None 表示这条 case 不挂工具；空列表与 None 含义不同（后者是"没声明"）。
    tools: list[ToolSpec] | None = None

    # 标注来源（`{kind, target}` 列表），供 RAGAS 与 judge 的 grounding 使用。
    # 空元组 = 这条没有可核的出处，`faithfulness` 会返回 None 而不是 0。
    references: tuple[dict[str, Any], ...] = ()

    @property
    def has_references(self) -> bool:
        return bool(self.references)

    @property
    def offline_fixture(self) -> dict[str, str] | None:
        if not isinstance(self.offline, dict):
            return None

        fixture = self.offline.get("fixture")

        return dict(fixture) if isinstance(fixture, dict) else None

    @property
    def offline_expect(self) -> dict[str, Any] | None:
        if not isinstance(self.offline, dict):
            return None

        expect = self.offline.get("expect")

        return dict(expect) if isinstance(expect, dict) else None


def load_cases(path: Path) -> list[GoldenCase]:
    cases: list[GoldenCase] = []

    for line_no, raw in enumerate(path.read_text("utf-8").splitlines(), start=1):
        line = raw.strip()

        if not line or line.startswith("//"):
            continue

        try:
            payload = json.loads(line)
        except ValueError as exc:
            raise SystemExit(f"{path}:{line_no} 不是合法 JSON：{exc}") from exc

        try:
            cases.append(_build_case(payload))
        except (KeyError, ValueError) as exc:
            raise SystemExit(f"{path}:{line_no} 用例结构不对：{exc}") from exc

    if not cases:
        raise SystemExit(f"{path} 里没有任何 case")

    return cases


def _build_case(payload: dict[str, Any]) -> GoldenCase:
    tools: list[ToolSpec] | None = None
    symbol = payload.get("tools")

    if symbol:
        names = TOOL_SYMBOLS.get(str(symbol))

        if not names:
            raise ValueError(f"未知 tools 符号 {symbol!r}")

        tools = [TOOL_SPECS[name] for name in names]

    return GoldenCase(
        id=str(payload["id"]),
        kind=str(payload["kind"]),
        difficulty=int(payload.get("difficulty", 1)),
        tags=tuple(payload.get("tags", ())),
        task=str(payload.get("task", "chat")),
        model=str(payload.get("model", OFFLINE_ALIAS)),
        online_model=str(payload.get("online_model", payload.get("model", OFFLINE_ALIAS))),
        messages=[
            Message(role=MessageRole(item["role"]), content=item.get("content"))
            for item in payload["messages"]
        ],
        expect=dict(payload["expect"]),
        offline=payload.get("offline"),
        tools=tools,
        references=tuple(payload.get("references", ())),
    )


# --------------------------------------------------------------------------- #
# 观测值与断言
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Observation:
    """一次调用能看到的所有东西。断言只读它，不再回头改 runner。"""

    # 模型最终吐出的文本。断言 `text_contains` / `must_not_include` 读它；
    # 流式档是拼帧后的全文，不是首帧。
    text: str = ""

    # 结束原因（STOP / LENGTH / TOOL_CALLS ...）。`LENGTH` 是"被截断"，
    # 不是"答完了"，用它区分 `BUDGET` 与 `FORMAT` 两类失败。
    finish_reason: str | None = None

    # 本次调用用到的工具名（按调用顺序，含重复）。用来核 `tool_calls` 断言。
    tool_names: list[str] = field(default_factory=list)

    # 流式档收到的帧类型序列（delta / tool_call / done ...）。用来核帧结构断言——
    # "内容对但帧顺序乱"在客户端就是渲染错位，`text` 看不出这个问题。
    frame_types: list[str] = field(default_factory=list)

    # 上游是否声明 token 为估算值。`None` = 这条没走到计费路径（夹具/跳过）。
    # 用它而不是直接信 `RequestLog`：断言要能在"没有日志"时也判出来。
    usage_estimated: bool | None = None

    # 归一化错误码与 HTTP 状态。两者分开是因为 `error_code` 会被人为归一
    # （`SERVER_BUSY`），而原始状态码保留现场，排查时先看它。
    error_code: str | None = None
    http_status: int | None = None

    # 工具调用计数：`tool_calls` 含失败，`tool_errors` 是其中报错的子集。
    tool_calls: int = 0
    tool_errors: int = 0


def check(observation: Observation, expect: dict[str, Any]) -> list[str]:
    """按 `expect` 判定。文案会进 `classify_failures`，所以措辞要保持稳定。"""

    failures: list[str] = []

    for needle in expect.get("text_contains", []):
        if needle not in observation.text:
            failures.append(f"文本里没有 {needle!r}（实际 {observation.text[:80]!r}）")

    any_words = expect.get("text_contains_any", [])

    if any_words and not any(word in observation.text for word in any_words):
        failures.append(f"文本里没有任何一个 {any_words!r}（实际 {observation.text[:80]!r}）")

    for needle in expect.get("must_not_include", []):
        if needle in observation.text:
            failures.append(f"文本里出现了本不该有的 {needle!r}")

    expected_reason = expect.get("finish_reason")

    if expected_reason and observation.finish_reason != expected_reason:
        failures.append(
            f"finish_reason 期望 {expected_reason}，实际 {observation.finish_reason}"
        )

    expected_tool = expect.get("tool_name")

    if expected_tool and expected_tool not in observation.tool_names:
        failures.append(f"tool_calls 里没有 {expected_tool!r}（实际 {observation.tool_names}）")

    expected_frames = expect.get("frame_types")

    if expected_frames and observation.frame_types != list(expected_frames):
        failures.append(f"帧序列期望 {expected_frames}，实际 {observation.frame_types}")

    expected_code = expect.get("error_code")

    if expected_code and observation.error_code != expected_code:
        failures.append(f"error_code 期望 {expected_code}，实际 {observation.error_code}")

    expected_status = expect.get("http_status")

    if expected_status and observation.http_status != expected_status:
        failures.append(f"http_status 期望 {expected_status}，实际 {observation.http_status}")

    return failures


def _check_log(case_id: str, expect: dict[str, Any], capture: LogCapture) -> list[str]:
    """`require_log`：证明埋点没断。

    它判的不是模型，是**观测链路**：没有 `trace_id` 就意味着这次调用在 trace 里
    是隐形的，W3 的"DoD 可观测"直接不成立。
    """

    if not expect.get("require_log"):
        return []

    latest = capture.latest()

    if not latest:
        return [f"[{case_id}] 没有 RequestLog：埋点断了（离线档就该在这里红）"]

    if not latest.get("trace_id") or latest.get("trace_id") == "-":
        return [f"[{case_id}] RequestLog 没有 trace_id：中间件没生效"]

    return []


# --------------------------------------------------------------------------- #
# 单条执行
# --------------------------------------------------------------------------- #


async def evaluate(
    case: GoldenCase,
    gateway: LLMGateway,
    capture: LogCapture,
    expect: dict[str, Any],
    *,
    alias: str,
    judge: Judge | None = None,
) -> tuple[EvalResult, JudgeResult | None]:
    capture.records.clear()
    call = ChatCall(
        messages=case.messages,
        task=case.task,
        tools=case.tools,
        model=alias,
        # 温度恒 0（ADR-W3-01 D3）：默认值会被某条 case 悄悄覆盖，而那条恰好最常被引用。
        opts={"temperature": 0},
    )

    observation = Observation()

    try:
        observation = await _invoke(case, gateway, call, observation)
    except LLMError as exc:
        observation.error_code = error_code_for(exc)
        observation.http_status = http_status_for(exc)
    except Exception as exc:  # noqa: BLE001 - 单条异常绝不能中断整批
        return (
            _result(
                case,
                status=CaseStatus.FAILED,
                capture=capture,
                observation=Observation(),
                failures=[f"执行异常 {type(exc).__name__}: {exc}"],
            ),
            None,
        )

    failures = check(observation, expect) + _check_log(case.id, expect, capture)

    # 期望报错的 case：错误码对上就算通过，别再去挑文本断言（它没有文本）。
    if case.kind == "error" and observation.error_code:
        status = CaseStatus.PASSED if not failures else CaseStatus.FAILED
    elif is_reliability_code(observation.error_code):
        # 上游挂了：记 interrupted，不进 success_rate 分母。
        status = CaseStatus.INTERRUPTED
        failures = failures + [f"上游中断 {observation.error_code}"]
    else:
        status = CaseStatus.PASSED if not failures else CaseStatus.FAILED

    ragas = await _ragas(case, observation.text)
    judged = await _judge(case, observation.text, judge)

    return (
        _result(
            case,
            status=status,
            capture=capture,
            observation=observation,
            failures=failures,
            ragas=ragas,
        ),
        judged,
    )


def _contexts(case: GoldenCase) -> list[str]:
    """把 references 解析成**正文**，供 RAGAS 使用（W5 接检索后换成真实 chunk）。

    两种引用要分开处理，否则指标恒为 0：
    - `kind=repo`：target 是仓库内真实文件，读正文；
    - `kind=url`：离线档不联网，没有正文。**不用 URL 字符串冒充 context**——
      `"https://modelcontextprotocol.io/specification"` 的实词是
      `https/modelcontextprotocol/io/specification`，任何答案都不会命中，
      faithfulness 会恒为 0，而读者会以为"模型在编造"。
    """

    contexts: list[str] = []

    for item in case.references:
        text = _reference_text(item)

        if text:
            contexts.append(text)

    return contexts


def _reference_text(reference: dict[str, Any]) -> str | None:
    explicit = reference.get("text")

    if isinstance(explicit, str) and explicit.strip():
        return explicit

    if reference.get("kind") != "repo":
        return None

    target = str(reference.get("target") or "")

    if not target:
        return None

    path = (REPO_ROOT / target).resolve()

    if REPO_ROOT.resolve() not in path.parents or not path.is_file():
        return None

    try:
        body = path.read_text("utf-8", errors="ignore")
    except OSError:
        return None

    # 截断：整篇文档进词面判定会把 faithfulness 顶满（context 越大越容易命中），
    # 6000 字符约等于一次检索返回的前几个 chunk。
    return body[:6000]


async def _ragas(case: GoldenCase, answer: str) -> RagasScores | None:
    """只有带 references 的 case 才算 RAGAS。

    错误 case 的 answer 是空串，`score_case` 会返回一个全 None 的结构——
    那不该计入 `ragas_samples`，否则"分母里塞了一堆算不了的样本"。
    """

    contexts = _contexts(case)

    if not contexts: # 如果没有检索到任何 references，直接返回 NONE
        return None

    scores = await score_case(
        question=_question(case), answer=answer, references=contexts
    )

    return scores if scores is not None and scores.has_any else None


async def _judge(case: GoldenCase, answer: str, judge: Judge | None) -> JudgeResult | None:
    if judge is None:
        return None

    return await judge.score(
        JudgeCase(
            id=case.id,
            question=_question(case),
            answer=answer,
            references=tuple(_contexts(case)),
            tags=case.tags,
        )
    )


def _question(case: GoldenCase) -> str:
    for message in case.messages:
        if str(message.role) == "user":
            return str(message.content or "")

    return ""


async def _invoke(
    case: GoldenCase,
    gateway: LLMGateway,
    call: ChatCall,
    observation: Observation,
) -> Observation:
    if case.kind == "stream":
        chunks: list[str] = []

        async for event in gateway.stream(call):
            observation.frame_types.append(event.type.value)

            if event.type is StreamEventType.TEXT_DELTA:
                chunks.append(event.text_delta or "")

            if event.type is StreamEventType.TOOL_CALL and event.tool_call:
                observation.tool_names.append(event.tool_call.name)
                observation.tool_calls += 1

                if event.tool_call.arguments_error:
                    observation.tool_errors += 1

            if event.type is StreamEventType.MESSAGE_END:
                observation.finish_reason = (
                    str(event.finish_reason) if event.finish_reason else None
                )
                observation.usage_estimated = event.usage.estimated if event.usage else None

        observation.text = "".join(chunks)

        return observation

    outcome = await gateway.chat(call)
    response = outcome.response

    observation.text = response.text or ""
    observation.finish_reason = str(response.finish_reason)
    observation.usage_estimated = response.usage.estimated
    observation.tool_names = [item.name for item in response.tool_calls]
    observation.tool_calls = len(response.tool_calls)
    observation.tool_errors = sum(
        1 for item in response.tool_calls if item.arguments_error
    )

    return observation


def _result(
    case: GoldenCase,
    *,
    status: CaseStatus,
    capture: LogCapture,
    observation: Observation,
    failures: list[str],
    ragas: RagasScores | None = None,
) -> EvalResult:
    metrics = CaseMetrics.from_request_log(capture.latest())
    metrics.tool_calls = observation.tool_calls
    metrics.tool_errors = observation.tool_errors

    if metrics.error_code is None:
        metrics.error_code = observation.error_code

    return EvalResult(
        case_id=case.id,
        status=status,
        kind=case.kind,
        difficulty=case.difficulty,
        tags=case.tags,
        metrics=metrics,
        failures=tuple(failures),
        categories=tuple(
            item.value
            for item in classify(
                error_code=metrics.error_code,
                failures=failures,
                tool_errors=metrics.tool_errors,
                finish_reason=observation.finish_reason,
                cost=metrics.cost,
                has_references=case.has_references,
            )
        )
        if status is CaseStatus.FAILED
        else (),
        detail=None,
        ragas=ragas,
    )


def _skipped(case: GoldenCase, reason: str) -> EvalResult:
    return EvalResult(
        case_id=case.id,
        status=CaseStatus.SKIPPED,
        kind=case.kind,
        difficulty=case.difficulty,
        tags=case.tags,
        detail=reason,
    )


# --------------------------------------------------------------------------- #
# 批量
# --------------------------------------------------------------------------- #


async def run_offline(
    cases: list[GoldenCase], capture: LogCapture, judge: Judge | None = None
) -> tuple[list[EvalResult], list[JudgeResult]]:
    results: list[EvalResult] = []
    judged: list[JudgeResult] = []

    for case in cases:
        expect = case.offline_expect
        fixture = case.offline_fixture

        if expect is None or fixture is None:
            # 对抗样例等语义类 case：离线档没有可执行判据。记 skipped，不算通过。
            results.append(_skipped(case, "离线档无判据（语义类，需在线档）"))
            continue

        gateway = build_offline_gateway(
            provider=str(fixture.get("provider", "openai")),
            fixture_case=str(fixture.get("case", "text_basic")),
            expect_error=case.kind == "error",
        )

        try:
            result, judge_result = await evaluate(
                case, gateway, capture, expect, alias=OFFLINE_ALIAS, judge=judge
            )
        finally:
            await gateway.aclose()

        results.append(result)

        if judge_result is not None:
            judged.append(judge_result)

    return results, judged


async def run_online(
    cases: list[GoldenCase], capture: LogCapture, judge: Judge | None = None
) -> tuple[list[EvalResult], list[JudgeResult]]:
    settings = Settings()
    catalog = load_catalog()

    try:
        gateway = build_gateway_or_raise(settings, catalog)
    except ConfigError as exc:
        raise SystemExit(f"在线档需要可用的 API key（.env.local）：{exc}") from exc

    results: list[EvalResult] = []
    judged: list[JudgeResult] = []

    try:
        for case in cases:
            result, judge_result = await evaluate(
                case,
                gateway,
                capture,
                case.expect,
                alias=case.online_model,
                judge=judge,
            )
            results.append(result)

            if judge_result is not None:
                judged.append(judge_result)
    finally:
        await gateway.aclose()

    return results, judged


def build_gateway_or_raise(settings: Settings, catalog: Any) -> LLMGateway:
    """在线档网关。抽成函数只为让异常类型在调用处显式可见。"""

    from agent_framework.bootstrap import build_gateway

    return build_gateway(settings, catalog=catalog)


# --------------------------------------------------------------------------- #
# 输出：控制台 / JSON / Markdown
# --------------------------------------------------------------------------- #


def print_table(results: list[EvalResult], metrics: TaskMetrics, *, mode: str) -> None:
    header = (
        f"{'id':<12}{'kind':<12}{'d':>2}{'result':<12}"
        f"{'first_tok':>10}{'latency':>9}{'tokens':>8}{'cost':>9}"
    )
    print(header)
    print("-" * len(header))

    for item in results:
        data = item.metrics
        first = data.first_token_ms
        print(
            f"{item.case_id:<12}{item.kind:<12}{item.difficulty:>3} "
            f"{item.status.value:<12}"
            f"{(f'{first:.1f}' if first is not None else '-'):>10}"
            f"{(f'{data.latency_ms:.1f}' if data.latency_ms is not None else '-'):>9}"
            f"{data.total_tokens:>8}"
            f"{(f'{data.cost:.6f}' if data.cost is not None else '-'):>9}"
        )

        for failure in item.failures:
            print(f"    - {failure}")

    rate = "n/a" if metrics.success_rate is None else f"{metrics.success_rate:.1%}"
    print(
        f"\n{metrics.total} 条：passed {metrics.passed} / failed {metrics.failed}"
        f" / interrupted {metrics.interrupted} / skipped {metrics.skipped}"
        f" → success_rate {rate}（mode={mode}）"
    )

    if mode == "offline":
        print(
            "注意：离线档是夹具回放，success_rate 只说明**执行管道**通，"
            "不代表模型质量；quality 块保持 null。"
        )


def _trend() -> str:
    """从 history 里读最近几期的数据点，拼成一行趋势。

    读不到就返回空——趋势图缺一个点比画错一个点好。
    """

    points: list[str] = []

    for path in sorted(HISTORY_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text("utf-8"))
        except (ValueError, OSError):
            continue

        quality = payload.get("quality") or payload.get("metrics", {}).get("quality") or {}
        rate = quality.get("success_rate") if isinstance(quality, dict) else None

        if rate is None:
            continue

        points.append(f"{payload.get('week', path.stem)}: {rate:.1%}")

    return " · ".join(points[-4:])


def render_markdown(
    results: list[EvalResult],
    metrics: TaskMetrics,
    *,
    mode: str,
    week: str,
    golden_digest: str,
) -> str:
    """四块固定结构（D5）：分层指标 / 难度分布 / 趋势 / 失败分类。"""

    lines: list[str] = [
        f"# 评测报告 · {week}（{mode}）",
        "",
        f"- golden：`evals/golden/golden_v1.jsonl` sha256:`{golden_digest}`",
        f"- 生成时间：{datetime.now(UTC).astimezone().isoformat(timespec='seconds')}",
        f"- 总数 {metrics.total}：passed {metrics.passed} / failed {metrics.failed}"
        f" / interrupted {metrics.interrupted} / skipped {metrics.skipped}",
        "",
    ]

    if mode == "offline":
        lines += [
            "> **离线档声明**：本报告来自夹具回放，`success_rate` 只证明执行管道可用。",
            "> 质量结论必须等在线档（`--live`）。未开 `--judge` 时 `judge_score` 为 `null`。",
            "",
        ]

    rate = "n/a" if metrics.success_rate is None else f"{metrics.success_rate:.2%}"
    lines += [
        "## 1. 分层指标",
        "",
        "| 层 | 指标 | 值 | 样本数 |",
        "|---|---|---|---|",
        f"| 任务 | success_rate | {rate} | {metrics.passed + metrics.failed} |",
        f"| 任务 | interrupted_rate | {_fmt_rate(metrics.interrupted_rate)} | {metrics.total} |",
        f"| 任务 | skip_rate | {_fmt_rate(metrics.skip_rate)} | {metrics.total} |",
        f"| 工具 | tool_error_rate | {_fmt_rate(metrics.tool_error_rate)} | {metrics.tool_calls} |",
        f"| 延迟 | latency p50 / p95 | {_fmt_ms(metrics.latency_p50_ms)} /"
        f" {_fmt_ms(metrics.latency_p95_ms)} | {metrics.latency_samples} |",
        f"| 延迟 | first_token p50 / p95 | {_fmt_ms(metrics.first_token_p50_ms)} /"
        f" {_fmt_ms(metrics.first_token_p95_ms)} | {metrics.first_token_samples} |",
        f"| 成本 | cost_per_task | {_fmt_cost(metrics.cost_per_task, metrics.currency)} |"
        f" {metrics.passed + metrics.failed} |",
        f"| 成本 | usage_estimated_ratio | {metrics.usage_estimated_ratio:.2%} |"
        f" {metrics.total} |",
        "| 质量 | judge_score（1–5） | "
        f"{_fmt_score(metrics.judge_score)} | {metrics.judge_samples} |",
        "| 质量 | faithfulness | "
        f"{_fmt_ratio(metrics.faithfulness)} | {metrics.ragas_samples} |",
        "| 质量 | answer_relevancy | "
        f"{_fmt_ratio(metrics.answer_relevancy)} | {metrics.ragas_samples} |",
        "| 质量 | context_precision | "
        f"{_fmt_ratio(metrics.context_precision)} | {metrics.ragas_samples} |",
        "",
    ]

    lines += [
        "## 2. 难度与标签分布",
        "",
        "| 维度 | 值 | 通过率 | 条数 |",
        "|---|---|---|---|",
    ]

    for level, group in by_difficulty(results).items():
        lines.append(f"| difficulty | {level} | {_fmt_rate(group.success_rate)} | {group.total} |")

    for tag, group in by_tag(results).items():
        lines.append(f"| tag | {tag} | {_fmt_rate(group.success_rate)} | {group.total} |")

    lines += [
        "",
        "### 按调用形态拆延迟（chat 与 stream 不是一个分布）",
        "",
        "| kind | 条数 | latency p50 | latency p95 | 样本 |",
        "|---|---|---|---|---|",
    ]

    for kind, group in by_kind(results).items():
        lines.append(
            f"| {kind} | {group.total} | {_fmt_ms(group.latency_p50_ms)} |"
            f" {_fmt_ms(group.latency_p95_ms)} | {group.latency_samples} |"
        )

    trend = _trend()
    lines += [
        "",
        "> 一条 case 有多个 tag 时会同时进各 tag 的分母，各 tag 条数之和大于总数是特性。",
        "",
        "## 3. 趋势",
        "",
        trend or "（`evals/history` 里还没有可比的 `success_rate` 数据点）",
        "",
        "## 4. 失败分类",
        "",
    ]

    failed = [item for item in results if item.status is CaseStatus.FAILED]
    interrupted = [item for item in results if item.status is CaseStatus.INTERRUPTED]

    if not failed and not interrupted:
        lines.append("本期无失败。")
    else:
        counts = summarize(
            [FailureCategory(value) for value in item.categories] for item in failed
        )
        lines += ["| 分类 | 条数 | 归因性质 |", "|---|---|---|"]

        for name, count in counts.items():
            if name in ("upstream", "timeout"):
                nature = "可靠性（不计入模型质量）"
            elif name == "request":
                nature = "我方请求/配置（不计入模型质量）"
            else:
                nature = "质量"
            lines.append(f"| {name} | {count} | {nature} |")

        lines += ["", "| case | 状态 | 分类 | 首条失败原因 |", "|---|---|---|---|"]

        for item in failed + interrupted:
            reason = item.failures[0] if item.failures else "-"
            lines.append(
                f"| {item.case_id} | {item.status.value} |"
                f" {', '.join(item.categories) or '-'} | {reason} |"
            )

    quality_failures = [
        item for item in failed if is_quality_failure([FailureCategory(c) for c in item.categories])
    ]
    lines += [
        "",
        f"质量类失败 {len(quality_failures)} 条，"
        f"可靠性类 {len(failed) - len(quality_failures)} 条。",
        "两者分开看：前者要改 prompt/工具，后者要看厂商状态页。",
        "",
    ]

    return "\n".join(lines)


def _fmt_score(value: float | None) -> str:
    return "null（未开 judge）" if value is None else f"{value:.2f}"


def _fmt_ratio(value: float | None) -> str:
    return "n/a（无 references）" if value is None else f"{value:.2f}"


def _fmt_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def _fmt_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}ms"


def _fmt_cost(value: float | None, currency: str | None) -> str:
    if value is None:
        return "n/a（无精确 usage）"

    return f"{value:.6f} {currency or ''}".strip()


def snapshot(
    results: list[EvalResult],
    metrics: TaskMetrics,
    *,
    mode: str,
    week: str,
    golden_path: Path,
    golden_digest: str,
    judged_alias: str | None = None,
) -> dict[str, Any]:
    """落盘结构：与 W44/W45 的 `metrics` 对齐，离线结果只写 `pipeline`。"""

    payload: dict[str, Any] = {
        "week": week,
        "gate": "W3",
        "mode": mode,
        "generated_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "golden": {
            "path": str(golden_path.relative_to(REPO_ROOT)),
            "sha256": golden_digest,
            "total": len(results),
        },
        "by_difficulty": {
            str(level): group.as_dict() for level, group in by_difficulty(results).items()
        },
        "by_tag": {tag: group.as_dict() for tag, group in by_tag(results).items()},
        "by_kind": {kind: group.as_dict() for kind, group in by_kind(results).items()},
        "failures": summarize(
            [FailureCategory(value) for value in item.categories] for item in results
        ),
        "cases": [
            {
                "id": item.case_id,
                "kind": item.kind,
                "difficulty": item.difficulty,
                "status": item.status.value,
                "categories": list(item.categories),
                "failures": list(item.failures),
                "metrics": asdict(item.metrics),
            }
            for item in results
        ],
    }

    payload["judge"] = {
        "alias": judged_alias,
        "samples": metrics.judge_samples,
        "score": metrics.judge_score,
        "rubric": f"{DEFAULT_RUBRIC.name}@{DEFAULT_RUBRIC.version}",
    }
    payload["ragas"] = {
        "samples": metrics.ragas_samples,
        "faithfulness": metrics.faithfulness,
        "answer_relevancy": metrics.answer_relevancy,
        "context_precision": metrics.context_precision,
    }

    if mode == "offline":
        # 离线跑通 ≠ 系统好（ADR-W3-01 D2）：quality 保持 null，离线数字进 pipeline。
        payload["pipeline"] = metrics.as_dict()
        payload["quality"] = {
            "success_rate": None,
            "judge_score": None,
            "status": "pending-online",
            "note": "离线档是夹具回放，只证明执行管道可用",
        }
    else:
        payload["quality"] = metrics.as_dict()

    return payload


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #


def current_week() -> str:
    today = datetime.now(UTC).astimezone().isocalendar()

    return f"{today.year}-W{today.week:02d}"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


async def _run(args: argparse.Namespace) -> int:
    golden_path = Path(args.golden)
    cases = load_cases(golden_path)

    if args.only:
        wanted = set(args.only)
        cases = [case for case in cases if case.id in wanted]

        if not cases:
            print(f"没有匹配的 case：{sorted(wanted)}", file=sys.stderr)
            return 2

    if args.break_case:
        # 反向验证：把指定 case 的判据改成永远不成立，用来证明"单条失败不中断整批"。
        for case in cases:
            if case.id == args.break_case:
                case.expect = {"text_contains": ["__never_appears__"]}

                if case.offline is not None:
                    case.offline = {
                        **case.offline,
                        "expect": {"text_contains": ["__never_appears__"]},
                    }

        print(f"[break-case] {args.break_case} 的判据已被改坏（预期：它红、其余照跑）")

    judge: Judge | None = None

    if args.judge:
        # 离线档 = 规则基线；在线档 = strong tier。别把两者混为一谈（报告会标 alias）。
        judge = build_judge(live=args.live)

    capture = LogCapture()
    logger = get_request_logger()
    # 直接挂 handler 不改 level：库不该动宿主 logging 配置，runner 自己负责看得见。
    logger.addHandler(capture)
    logger.setLevel(logging.INFO)

    started = time.perf_counter()

    try:
        results, judged = (
            await run_online(cases, capture, judge)
            if args.live
            else await run_offline(cases, capture, judge)
        )
    finally:
        logger.removeHandler(capture)

    metrics = aggregate(results)
    # judge 均分只对"评成了"的样本求均值；judge_samples 落盘，读者能判断覆盖率。
    metrics.judge_score = mean_overall(judged, DEFAULT_RUBRIC) if judged else None
    metrics.judge_samples = len(judged)
    judged_alias = getattr(judge, "alias", judge.name) if judge is not None else None
    mode = "online" if args.live else "offline"
    # 默认按 ISO 周；`--week` 用于"提前一周做下周的活"——本仓库 W2 就是这么排的，
    # 用日历周自动命名会把 W3 的报告写成 W2 那一周，趋势图上两期并成一条。
    week = args.week or current_week()
    digest = _digest(golden_path)

    print_table(results, metrics, mode=mode)

    if judged:
        print(f"judge：{json.dumps(summarize_tokens(judged), ensure_ascii=False)}")
        print(f"judge 均分 {metrics.judge_score}（{metrics.judge_samples} 条）")

    if metrics.ragas_samples:
        print(
            f"RAGAS（{metrics.ragas_samples} 条）：faithfulness {metrics.faithfulness}"
            f" · relevancy {metrics.answer_relevancy}"
            f" · context_precision {metrics.context_precision}"
        )

    if args.verbose:
        for item in results:
            print(
                f"\n[{item.case_id}] "
                f"{json.dumps(asdict(item.metrics), ensure_ascii=False, default=str)}"
            )

    if not args.no_write:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        json_path = HISTORY_DIR / f"{week}-eval.json"
        md_path = HISTORY_DIR / f"{week}-w3-eval.md"

        json_path.write_text(
            json.dumps(
                snapshot(
                    results,
                    metrics,
                    mode=mode,
                    week=week,
                    golden_path=golden_path,
                    golden_digest=digest,
                    judged_alias=judged_alias,
                ),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            "utf-8",
        )
        md_path.write_text(
            render_markdown(
                results, metrics, mode=mode, week=week, golden_digest=digest
            ),
            "utf-8",
        )
        print(f"\n已写入 {json_path.relative_to(REPO_ROOT)} 与 {md_path.relative_to(REPO_ROOT)}")

    elapsed = time.perf_counter() - started
    print(f"耗时 {elapsed:.2f}s")

    bad = metrics.failed + metrics.interrupted

    return 0 if bad == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="golden 批量评测（默认离线档）")
    parser.add_argument("--golden", default=str(GOLDEN_PATH))
    parser.add_argument("--only", nargs="*", help="只跑指定 id")
    parser.add_argument("--live", action="store_true", help="在线档（需 .env.local 的 key）")
    parser.add_argument("--break-case", default=None, help="把指定 case 的判据改坏（反向验证）")
    parser.add_argument("--no-write", action="store_true", help="不落盘（只看输出）")
    parser.add_argument("--week", default=None, help="覆盖报告期号，如 2026-W46")
    parser.add_argument(
        "--judge", action="store_true", help="顺带跑 judge（离线档为规则基线，在线档走 strong）"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
