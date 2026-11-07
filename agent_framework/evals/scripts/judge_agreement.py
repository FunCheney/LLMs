"""Judge 一致性校验（W4 D1）：`make judge-agreement`。

**没有这一步的 Judge 等于随机数。** 这份脚本回答一个必须能被追问的问题：
"你说分数从 3.8 涨到 4.1，凭什么相信这个分数？"

它把 ≥20 条人工标注的答案交给 judge 重打一遍，然后算三个一致性系数
（二次加权 kappa / 三档名义 kappa / Spearman）。低于阈值不是"调高温度"，
是回去改 rubric——温度只会让分数更随机，不会让它更对。

两种模式
--------
- 默认（离线）：`HeuristicJudge`，即关键词规则基线。它给出的是**地板**：
  LLM judge 如果打不过它，这笔钱就是白花的。
- `--live`：真 judge，走 gateway 的 strong tier，需 `.env.local`。

报告里的 `judge_alias` 决定了这行数字能不能被引用：`heuristic-keyword` 只证明
"校验链路通"，不证明"judge 准"。

用法
----
    uv run python evals/scripts/judge_agreement.py              # 规则基线（零网络）
    uv run python evals/scripts/judge_agreement.py --live       # 真 judge
    uv run python evals/scripts/judge_agreement.py --min-kappa 0.5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agent_framework.config import (  # noqa: E402
    ConfigError,
    Settings,
    load_catalog,
)

from evals.judge.llm_judge import (  # noqa: E402
    Judge,
    JudgeCase,
    build_judge,
    summarize_tokens,
)
from evals.judge.rubric import DEFAULT_RUBRIC, JudgeResult  # noqa: E402
from evals.judge.stats import (  # noqa: E402
    agree_exact_rate,
    bucket,
    distribution,
    mean_absolute_error,
    nominal_kappa,
    quadratic_weighted_kappa,
    spearman,
)

GOLDEN_PATH = REPO_ROOT / "evals" / "golden" / "golden_v1.jsonl"
ANNOTATIONS_PATH = REPO_ROOT / "evals" / "judge" / "annotations_v1.jsonl"
HISTORY_DIR = REPO_ROOT / "evals" / "history"

# 0.4 是"中等一致"的下限（Landis & Koch 的 moderate 档）。
# 定得更高会让人为了过线去改标注，定得更低则等于承认分数不可用。
DEFAULT_MIN_KAPPA = 0.40

# 少于这个数量的一致性系数没有统计意义：20 条里差 2 条就能让 kappa 抖动 0.1。
MIN_ANNOTATIONS = 20


@dataclass(slots=True)
class Row:
    """一条配对记录：同一条 case 的"人工分"与"judge 分"。

    一次性存三种形态的分数是刻意的：一致性系数对"用哪个"极其敏感，
    而汇总表里只留一个数字，评审过程就不可复核了。
    """

    # 标注记录自己的 id（不是 case id）。一条 case 可能被标注多次时靠它区分。
    annotation_id: str

    # 指向 golden 的 case id。用它把 golden 里的问题与引用取出来送审。
    case_id: str

    # 人工标注的总体分（1–5 整数）。**这是基准**，judge 的漂移都相对它来判断。
    human: int

    # judge 的加权总分（1–5 的浮点，如 4.25）。给 spearman 用——相关系数
    # 需要连续值，把它先取整会丢掉"4.1 和 4.9 都算 4"这一层信息。
    judge: float | None

    # `judge` 四舍五入到整数。给 kappa 与 MAE 用：人工标注只有整数，
    # 不取整就等于拿两把不同刻度的尺子比。
    judge_rounded: int | None

    # 判定依据（`parse_error` 或 "fact=4; grounding=3; ..."）。人工复核
    # 那些不一致的条目时，先看的就是它。
    rationale: str


def load_annotations(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for line_no, raw in enumerate(path.read_text("utf-8").splitlines(), start=1):
        line = raw.strip()

        if not line or line.startswith("//"):
            continue

        try:
            rows.append(json.loads(line))
        except ValueError as exc:
            raise SystemExit(f"{path}:{line_no} 不是合法 JSON：{exc}") from exc

    if not rows:
        raise SystemExit(f"{path} 里没有标注")

    return rows


def load_golden(path: Path) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}

    for raw in path.read_text("utf-8").splitlines():
        line = raw.strip()

        if not line or line.startswith("//"):
            continue

        payload = json.loads(line)
        cases[str(payload["id"])] = payload

    return cases


def build_cases(
    annotations: list[dict[str, Any]], golden: dict[str, dict[str, Any]]
) -> list[tuple[str, JudgeCase]]:
    """把标注拼成送审样本。

    问题与引用从 golden 取，答案与人工分从标注取：两边各管一段，
    改 golden 的引用不需要重标一遍，改标注也不会污染数据集。
    """

    built: list[tuple[str, JudgeCase]] = []

    for row in annotations:
        case_id = str(row["case_id"])
        source = golden.get(case_id)

        if source is None:
            raise SystemExit(f"标注 {row.get('id')} 指向不存在的 case {case_id}")

        question = next(
            (
                str(message.get("content") or "")
                for message in source.get("messages", ())
                if message.get("role") == "user"
            ),
            "",
        )
        references = tuple(
            str(item.get("target", "")) for item in source.get("references", ())
        )

        built.append(
            (
                str(row.get("id") or case_id),
                JudgeCase(
                    id=case_id,
                    question=question,
                    answer=str(row["answer"]),
                    references=references,
                    tags=tuple(str(tag) for tag in source.get("tags", ())),
                ),
            )
        )

    return built


async def score_all(
    judge: Judge, samples: list[tuple[str, JudgeCase]], annotations: list[dict[str, Any]]
) -> tuple[list[Row], list[JudgeResult]]:
    rows: list[Row] = []
    results: list[JudgeResult] = []

    for (annotation_id, case), annotation in zip(samples, annotations, strict=True):
        result = await judge.score(case)
        results.append(result)
        overall = result.overall(DEFAULT_RUBRIC)
        human = int(annotation["human_score"])

        rows.append(
            Row(
                annotation_id=annotation_id,
                case_id=case.id,
                human=human,
                judge=overall,
                judge_rounded=None if overall is None else int(round(overall)),
                rationale=(
                    result.parse_error
                    or "; ".join(
                        f"{item.key}={item.score}" for item in result.scores
                    )
                ),
            )
        )

    return rows, results


def evaluate_rows(rows: list[Row], *, min_kappa: float) -> dict[str, Any]:
    usable = [row for row in rows if row.judge is not None and row.judge_rounded is not None]
    unavailable = len(rows) - len(usable)

    if len(usable) < 2:
        return {
            "samples": len(usable),
            "unavailable": unavailable,
            "passed": False,
            "reason": "可用的 judge 结果不足 2 条",
        }

    human = [row.human for row in usable]
    judge_int = [row.judge_rounded for row in usable if row.judge_rounded is not None]
    judge_float = [row.judge for row in usable if row.judge is not None]

    weighted = quadratic_weighted_kappa(human, judge_int)
    nominal = nominal_kappa([bucket(item) for item in human], [bucket(item) for item in judge_int])
    rho = spearman(human, judge_float)

    return {
        "samples": len(usable),
        "unavailable": unavailable,
        "quadratic_weighted_kappa": round(weighted, 4),
        "nominal_kappa": round(nominal, 4),
        "spearman": round(rho, 4),
        "exact_match_rate": round(agree_exact_rate(human, judge_int), 4),
        "mae": round(mean_absolute_error(human, judge_int), 4),
        "human_distribution": distribution(human),
        "judge_distribution": distribution(judge_int),
        "min_kappa": min_kappa,
        # 用名义 kappa 当门禁：它比加权 kappa 严，且不容易被"相邻档宽容"救回来。
        "passed": nominal >= min_kappa,
    }


def render_report(
    rows: list[Row], stats: dict[str, Any], *, mode: str, judge_alias: str, week: str
) -> str:
    lines = [
        f"# Judge 一致性报告 · {week}（{mode}）",
        "",
        f"- judge：`{judge_alias}`；rubric：`{DEFAULT_RUBRIC.name}` v{DEFAULT_RUBRIC.version}",
        f"- 标注文件：`evals/judge/annotations_v1.jsonl`（{len(rows)} 条）",
        f"- 生成时间：{datetime.now(UTC).astimezone().isoformat(timespec='seconds')}",
        "",
    ]

    if judge_alias == "heuristic-keyword":
        lines += [
            "> **这份数字不能用来宣称 judge 可信。** `heuristic-keyword` 是关键词规则基线，",
            "> 它给出的是**地板**：真正的 LLM judge 必须明显高于它，否则不如不用。",
            "> 要拿到 LLM judge 的一致性，需要 `--live`（依赖 `.env.local` 的 key）。",
            "",
        ]

    lines += [
        "## 系数",
        "",
        "| 指标 | 值 | 说明 |",
        "|---|---|---|",
        f"| 样本数 | {stats.get('samples', 0)} | 可用 judge 结果 |",
        f"| 不可用 | {stats.get('unavailable', 0)} | judge 报错/解析失败，不计入系数 |",
        f"| 二次加权 kappa | {stats.get('quadratic_weighted_kappa', 'n/a')} | 有序，差 1 分轻罚 |",
        f"| 三档名义 kappa | {stats.get('nominal_kappa', 'n/a')} | 门禁用这个（低/中/高） |",
        f"| Spearman | {stats.get('spearman', 'n/a')} | 只看排序 |",
        f"| 完全一致率 | {stats.get('exact_match_rate', 'n/a')} | 分数完全相同 |",
        f"| MAE | {stats.get('mae', 'n/a')} | 平均差几分 |",
        "",
        "## 分数分布（看有没有全堆在 4 分）",
        "",
        f"- 人工：`{stats.get('human_distribution')}`",
        f"- judge：`{stats.get('judge_distribution')}`",
        "",
        "## 逐条",
        "",
        "| 标注 | case | 人工 | judge | 维度分 |",
        "|---|---|---|---|---|",
    ]

    for row in rows:
        judge = "n/a" if row.judge is None else f"{row.judge:.2f}"
        lines.append(
            f"| {row.annotation_id} | {row.case_id} | {row.human} | {judge} | {row.rationale} |"
        )

    threshold = stats.get("min_kappa")
    verdict = "通过" if stats.get("passed") else "未通过"

    lines += [
        "",
        "## 判定",
        "",
        f"阈值 `min_kappa={threshold}`，名义 kappa = {stats.get('nominal_kappa', 'n/a')}"
        f" → **{verdict}**。",
        "",
    ]

    if not stats.get("passed"):
        lines += [
            "未通过时的处置顺序（**不是**调温度）：",
            "",
            "1. 看分布：若 judge 分布挤在单一分值，说明锚定描述没有区分度，改锚；",
            "2. 看逐条：若差异集中在某一维度，说明该维度问得不清楚，改 `question`；",
            "3. 看标注：若两条人工分本身互相矛盾，先停下来复核标注，别改 rubric。",
            "",
        ]

    return "\n".join(lines)


def build_live_judge(alias: str | None = None) -> Judge:
    settings = Settings()

    try:
        from agent_framework.bootstrap import build_gateway

        gateway = build_gateway(settings, catalog=load_catalog())
    except ConfigError as exc:
        raise SystemExit(f"--live 需要可用的 API key（.env.local）：{exc}") from exc

    return build_judge(live=True, gateway=gateway, alias=alias)


def current_week() -> str:
    today = datetime.now(UTC).astimezone().isocalendar()

    return f"{today.year}-W{today.week:02d}"


async def _run(args: argparse.Namespace) -> int:
    annotations = load_annotations(Path(args.annotations))
    golden = load_golden(Path(args.golden))

    if len(annotations) < MIN_ANNOTATIONS:
        # 少于 20 条时系数没有意义，直接拦住比"跑出一个 0.62"更负责。
        print(
            f"标注只有 {len(annotations)} 条，少于下限 {MIN_ANNOTATIONS}；"
            "请补足后再算一致性（样本不足的 kappa 会误导）",
            file=sys.stderr,
        )
        return 2

    samples = build_cases(annotations, golden)
    judge = build_live_judge(args.alias) if args.live else build_judge(live=False)
    rows, results = await score_all(judge, samples, annotations)
    stats = evaluate_rows(rows, min_kappa=args.min_kappa)
    mode = "online" if args.live else "offline"
    alias = getattr(judge, "alias", judge.name)

    header = f"{'标注':<8}{'case':<10}{'人工':>4}{'judge':>8}  维度分"
    print(header)
    print("-" * len(header))

    for row in rows:
        judge_text = "-" if row.judge is None else f"{row.judge:.2f}"
        print(
            f"{row.annotation_id:<8}{row.case_id:<10}{row.human:>4}{judge_text:>8}  "
            f"{row.rationale[:60]}"
        )

    if args.verbose:
        print(json.dumps(stats, ensure_ascii=False, indent=2, default=str))

    print(
        f"\n{stats.get('samples', 0)} 条可用 / {stats.get('unavailable', 0)} 条不可用 · "
        f"加权 kappa {stats.get('quadratic_weighted_kappa', 'n/a')} · "
        f"名义 kappa {stats.get('nominal_kappa', 'n/a')} · "
        f"spearman {stats.get('spearman', 'n/a')} · "
        f"MAE {stats.get('mae', 'n/a')}"
    )
    print(f"阈值 {args.min_kappa} → {'通过' if stats.get('passed') else '未通过'}（judge={alias}）")

    if judge_alias_is_llm(alias):
        # 成本单独打一行：judge 会把评测成本抬上去，W11 的看板要能按 judge/被测分账。
        print(f"judge 成本摘要：{json.dumps(summarize_tokens(results), ensure_ascii=False)}")

    if not args.no_write:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        path = HISTORY_DIR / f"{args.week or current_week()}-judge-agreement.md"
        path.write_text(
            render_report(
                rows, stats, mode=mode, judge_alias=alias, week=args.week or current_week()
            ),
            "utf-8",
        )
        print(f"已写入 {path.relative_to(REPO_ROOT)}")

    return 0 if stats.get("passed") else 1


def judge_alias_is_llm(alias: str) -> bool:
    """`heuristic-keyword` 是规则基线，不该走成本摘要那套话术。"""

    return alias != "heuristic-keyword"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Judge 与人工标注的一致性校验")
    parser.add_argument("--annotations", default=str(ANNOTATIONS_PATH))
    parser.add_argument("--golden", default=str(GOLDEN_PATH))
    parser.add_argument("--live", action="store_true", help="用真 judge（需 key）")
    parser.add_argument("--alias", default=None, help="覆盖 judge 的 alias（默认 strong tier）")
    parser.add_argument("--min-kappa", type=float, default=DEFAULT_MIN_KAPPA)
    parser.add_argument("--week", default=None)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
