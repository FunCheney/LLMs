"""golden set 的静态校验（W3 D2，进 CI）。

它不跑模型，只回答一个问题：**这份数据集本身有没有坏**。坏有六种，任何一种都会让
后面的评测数字失去意义：

1. **悬空引用** —— 引用指向不存在的文件，W4 的 faithfulness 无从算起；
2. **重复 id** —— 两条 case 覆盖同一行，结果是"总 case 数"虚高；
3. **重复输入** —— 换个 id 抄一遍题干，成功率被同一道题灌水；
4. **难度缺失/分布塌陷** —— 3 档只剩 1 条，按难度切的趋势线全是噪声；
5. **夹具不存在** —— 离线档跑到一半 `FileNotFoundError`，整批中断；
6. **对抗样例偷懒补 offline** —— 用夹具回放"证明"模型会拒绝，这是假绿。

退出码：0 通过 / 1 有问题（CI 门禁靠它）。

用法
----
    uv run python evals/scripts/validate_golden.py
    uv run python evals/scripts/validate_golden.py --golden evals/golden/golden_v1.jsonl -v
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = REPO_ROOT / "evals" / "golden" / "golden_v1.jsonl"
FIXTURES_ROOT = REPO_ROOT / "packages" / "agent_framework" / "tests" / "fixtures"

# 符号名 → 实际声明给模型的工具。runner 用同一张表展开，避免"校验器认、runner 不认"。
TOOL_SYMBOLS: dict[str, tuple[str, ...]] = {
    "search_or_fetch": ("search_web", "fetch_doc"),
}

KINDS = {"chat", "tool_call", "research", "draft", "stream", "error", "adversarial"}
SOURCES = {"handwritten", "recorded"}
ROLES = {"system", "user", "assistant", "tool"}

# 判据键白名单：拼错一个键（text_contain）会让断言静默失效——runner 读到未知键就当没写。
EXPECT_KEYS = {
    "text_contains",
    "text_contains_any",
    "must_not_include",
    "finish_reason",
    "tool_name",
    "error_code",
    "http_status",
    "frame_types",
    "require_log",
}

REQUIRED_KEYS = {
    "id",
    "kind",
    "difficulty",
    "tags",
    "source",
    "task",
    "messages",
    "expect",
}

MIN_TOTAL = 50
MIN_PER_DIFFICULTY = 10
MIN_ADVERSARIAL = 3


@dataclass
class Report:
    """校验结果。分三桶而不是只留一个 bool：`errors` 决定退出码，`warnings` 决定
    "这周要不要补标注"，`cases` 留给下游算难度分布。"""

    # 硬错（重复 id、悬空引用、未知 kind）。任一非空即退出码 1。
    errors: list[str] = field(default_factory=list)

    # 软错（难度分布不足、引用指向占位符）。不阻断合并，但要在 CI 日志里被看见——
    # 它们不修，W4 的 faithfulness 就永远只能在一部分 case 上算。
    warnings: list[str] = field(default_factory=list)

    # 解析成功的原始 case（未校验的 dict）。用来算难度/kind/tag 分布，
    # 也用来回答"这条 id 到底存不存在"。
    cases: list[dict[str, Any]] = field(default_factory=list)

    def error(self, case_id: str, message: str) -> None:
        self.errors.append(f"[{case_id}] {message}")

    def warn(self, case_id: str, message: str) -> None:
        self.warnings.append(f"[{case_id}] {message}")


def load(path: Path) -> list[dict[str, Any]]:
    """读 JSONL。注释行（`//`）与空行跳过——数据集里要能写解释，不然半年后没人看懂为什么这么设计。"""

    cases: list[dict[str, Any]] = []

    for line_no, raw in enumerate(path.read_text("utf-8").splitlines(), start=1):
        line = raw.strip()

        if not line or line.startswith("//"):
            continue

        try:
            payload = json.loads(line)
        except ValueError as exc:
            raise SystemExit(f"{path}:{line_no} 不是合法 JSON：{exc}") from exc

        if not isinstance(payload, dict):
            raise SystemExit(f"{path}:{line_no} 不是 JSON 对象")

        payload["__line__"] = line_no
        cases.append(payload)

    return cases


# --------------------------------------------------------------------------- #
# 逐条校验
# --------------------------------------------------------------------------- #


def check_case(case: dict[str, Any], report: Report) -> None:
    case_id = str(case.get("id", f"line{case.get('__line__')}"))

    missing = REQUIRED_KEYS - set(case)

    if missing:
        report.error(case_id, f"缺字段 {sorted(missing)}")
        return

    kind = case["kind"]

    if kind not in KINDS:
        report.error(case_id, f"kind={kind!r} 不在 {sorted(KINDS)}")

    difficulty = case["difficulty"]

    if (
        not isinstance(difficulty, int)
        or isinstance(difficulty, bool)
        or difficulty not in (1, 2, 3)
    ):
        report.error(case_id, f"difficulty={difficulty!r} 必须是 1/2/3 的整数")

    if not isinstance(case["tags"], list) or not case["tags"]:
        report.error(case_id, "tags 必须是非空列表")

    if case["source"] not in SOURCES:
        report.error(case_id, f"source={case['source']!r} 不在 {sorted(SOURCES)}")

    if not case_id.startswith(f"gold-{kind[0]}"):
        report.warn(case_id, f"id 前缀与 kind={kind} 不一致（不影响执行，影响人读）")

    check_messages(case_id, case["messages"], report)
    check_expect(case_id, case["expect"], report, label="expect")
    check_tools(case_id, case, report)
    check_offline(case_id, case, report)
    check_references(case_id, case.get("references"), report)


def check_messages(case_id: str, messages: Any, report: Report) -> None:
    if not isinstance(messages, list) or not messages:
        report.error(case_id, "messages 必须是非空列表")
        return

    for index, item in enumerate(messages):
        if not isinstance(item, dict):
            report.error(case_id, f"messages[{index}] 不是对象")
            continue

        if item.get("role") not in ROLES:
            report.error(case_id, f"messages[{index}].role={item.get('role')!r} 非法")

        if not isinstance(item.get("content"), str) or not item["content"].strip():
            report.error(case_id, f"messages[{index}].content 必须是非空字符串")


def check_expect(case_id: str, expect: Any, report: Report, *, label: str) -> None:
    if not isinstance(expect, dict) or not expect:
        report.error(case_id, f"{label} 必须是非空对象")
        return

    unknown = set(expect) - EXPECT_KEYS

    if unknown:
        report.error(case_id, f"{label} 含未知判据 {sorted(unknown)}（拼错的键会静默失效）")

    for key in ("text_contains", "text_contains_any", "must_not_include", "frame_types"):
        value = expect.get(key)

        if value is not None and (not isinstance(value, list) or not value):
            report.error(case_id, f"{label}.{key} 必须是非空列表")

    # 两套判据同时出现同一个串，说明有人把在线判据复制到离线判据里凑绿。
    if "must_not_include" in expect and "text_contains" in expect:
        overlap = set(expect["must_not_include"]) & set(expect["text_contains"])

        if overlap:
            report.error(case_id, f"{label} 同时要求出现与不出现 {sorted(overlap)}")


def check_tools(case_id: str, case: dict[str, Any], report: Report) -> None:
    tools = case.get("tools")
    kind = case["kind"]

    if kind == "tool_call" and not tools:
        report.error(case_id, "kind=tool_call 必须声明 tools")
        return

    if tools is None:
        return

    if tools not in TOOL_SYMBOLS:
        report.error(case_id, f"tools={tools!r} 不在符号表 {sorted(TOOL_SYMBOLS)}")
        return

    expected_tool = case["expect"].get("tool_name")

    if kind == "tool_call" and expected_tool and expected_tool not in TOOL_SYMBOLS[tools]:
        report.error(
            case_id, f"tool_name={expected_tool!r} 不在 tools={tools!r} 声明范围内"
        )


def check_offline(case_id: str, case: dict[str, Any], report: Report) -> None:
    offline = case.get("offline")
    kind = case["kind"]

    # 对抗样例的判据是"模型拒绝做某事"，夹具回放证明不了这件事。
    if kind == "adversarial":
        if offline is not None:
            report.error(
                case_id,
                "对抗样例不得填 offline：用夹具回放『证明』模型会拒绝是假绿（见 schema.md）",
            )

        return

    if not isinstance(offline, dict):
        report.error(case_id, "非对抗样例必须填 offline（否则离线档无法执行）")
        return

    fixture = offline.get("fixture")

    if not isinstance(fixture, dict) or not fixture.get("provider") or not fixture.get("case"):
        report.error(case_id, "offline.fixture 必须含 provider 与 case")
        return

    check_expect(case_id, offline.get("expect"), report, label="offline.expect")

    provider, fixture_case = str(fixture["provider"]), str(fixture["case"])
    case_dir = FIXTURES_ROOT / provider / fixture_case

    if not case_dir.is_dir():
        report.error(case_id, f"夹具目录不存在：{case_dir.relative_to(REPO_ROOT)}")
        return

    is_error_fixture = fixture_case.startswith("error_")

    if kind == "error" and not is_error_fixture:
        report.error(case_id, f"kind=error 却用了非错误夹具 {provider}/{fixture_case}")

    if kind != "error" and is_error_fixture:
        report.error(
            case_id, f"kind={kind} 却用了错误夹具 {provider}/{fixture_case}（永远会中断）"
        )


def check_references(case_id: str, references: Any, report: Report) -> None:
    if references is None:
        return

    if not isinstance(references, list) or not references:
        report.error(case_id, "references 必须是非空列表（不填就别写这个字段）")
        return

    for index, item in enumerate(references):
        if not isinstance(item, dict):
            report.error(case_id, f"references[{index}] 不是对象")
            continue

        kind = item.get("kind")
        target = item.get("target")

        if not isinstance(target, str) or not target:
            report.error(case_id, f"references[{index}].target 必须是非空字符串")
            continue

        if kind == "repo":
            path = REPO_ROOT / target

            if not path.is_file():
                report.error(case_id, f"references[{index}] 指向不存在的文件：{target}")
                continue

            anchor = item.get("anchor")

            if anchor and anchor not in path.read_text("utf-8"):
                report.error(
                    case_id, f"references[{index}] 的锚点 {anchor!r} 在 {target} 里找不到"
                )
        elif kind == "url":
            parsed = urlparse(target)

            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                report.error(case_id, f"references[{index}] 不是合法 URL：{target}")
            # 刻意不联网校验：CI 里一次 302/超时就让门禁随机变红，与代码质量无关。
        else:
            report.error(case_id, f"references[{index}].kind={kind!r} 必须是 repo/url")


# --------------------------------------------------------------------------- #
# 整体校验
# --------------------------------------------------------------------------- #


def check_dataset(cases: list[dict[str, Any]], report: Report) -> None:
    if len(cases) < MIN_TOTAL:
        report.errors.append(f"总条数 {len(cases)} < {MIN_TOTAL}")

    ids = Counter(str(case.get("id")) for case in cases)
    duplicated = [key for key, count in ids.items() if count > 1]

    if duplicated:
        report.errors.append(f"重复 id：{sorted(duplicated)}")

    # 重复输入检测：同 kind + 同 messages 视为同一道题换了个 id。
    fingerprints: dict[tuple[str, str], str] = {}

    for case in cases:
        key = (str(case.get("kind")), json.dumps(case.get("messages"), ensure_ascii=False))

        if key in fingerprints:
            report.errors.append(
                f"重复输入：[{case.get('id')}] 与 [{fingerprints[key]}] 题干完全相同"
            )
        else:
            fingerprints[key] = str(case.get("id"))

    difficulties = Counter(case.get("difficulty") for case in cases)

    for level in (1, 2, 3):
        if difficulties.get(level, 0) < MIN_PER_DIFFICULTY:
            report.errors.append(
                f"难度 {level} 只有 {difficulties.get(level, 0)} 条，"
                f"低于 {MIN_PER_DIFFICULTY}（按难度切的趋势线会全是噪声）"
            )

    adversarial = sum(1 for case in cases if case.get("kind") == "adversarial")

    if adversarial < MIN_ADVERSARIAL:
        report.errors.append(f"对抗样例只有 {adversarial} 条，低于 {MIN_ADVERSARIAL}")

    if not any(case.get("kind") == "stream" for case in cases):
        report.warnings.append("没有任何 stream 用例：首 token 指标会一直是 None")


def print_summary(cases: list[dict[str, Any]]) -> None:
    kinds = Counter(str(case.get("kind")) for case in cases)
    difficulties = Counter(case.get("difficulty") for case in cases)
    tags = Counter(tag for case in cases for tag in case.get("tags", []))
    providers = Counter(
        str(case["offline"]["fixture"]["provider"])
        for case in cases
        if isinstance(case.get("offline"), dict)
    )
    offline = sum(1 for case in cases if isinstance(case.get("offline"), dict))

    print(f"总条数 {len(cases)}（离线可执行 {offline}，skipped {len(cases) - offline}）")
    print("难度分布 " + " ".join(f"{level}:{difficulties.get(level, 0)}" for level in (1, 2, 3)))
    print("kind 分布 " + " ".join(f"{key}:{value}" for key, value in sorted(kinds.items())))
    print("供应商分布 " + " ".join(f"{key}:{value}" for key, value in sorted(providers.items())))
    print("tag 数 " + str(len(tags)) + "：" + ", ".join(sorted(tags)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="校验 golden set")
    parser.add_argument("--golden", default=str(GOLDEN_PATH))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    path = Path(args.golden)
    cases = load(path)
    report = Report(cases=cases)

    for case in cases:
        check_case(case, report)

    check_dataset(cases, report)

    print_summary(cases)

    if args.verbose and report.warnings:
        print("\n警告：")

        for item in report.warnings:
            print(f"  - {item}")

    if report.errors:
        print(f"\n{len(report.errors)} 个问题：", file=sys.stderr)

        for item in report.errors:
            print(f"  - {item}", file=sys.stderr)

        return 1

    print("\nOK：字段、引用、夹具、难度分布全部通过")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
