"""golden set 的守门测试（W3 D2）。

正向：当前数据集通过校验，且分布满足 Gate 要求。
反向：**故意弄坏一份拷贝，校验器必须报错**。不变红的校验器等于没有校验器——
W2 的 smoke 已经用同一招证明过门禁有效，这里沿用。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from evals.metrics import CaseStatus  # noqa: F401  (确认 evals 包可导入)

REPO_ROOT = Path(__file__).resolve().parents[3]
VALIDATOR = REPO_ROOT / "evals" / "scripts" / "validate_golden.py"
GOLDEN = REPO_ROOT / "evals" / "golden" / "golden_v1.jsonl"


def _validate(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), "--golden", str(path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _cases() -> list[dict]:
    return [
        json.loads(line)
        for line in GOLDEN.read_text("utf-8").splitlines()
        if line.strip() and not line.strip().startswith("//")
    ]


def test_golden_set_validates_clean() -> None:
    result = _validate(GOLDEN)

    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_golden_meets_size_and_difficulty_gate() -> None:
    """Gate 2：≥50 条，三档每档 ≥10，≥3 条对抗样例。"""

    cases = _cases()
    difficulties = [case["difficulty"] for case in cases]

    assert len(cases) >= 50

    for level in (1, 2, 3):
        assert difficulties.count(level) >= 10, f"难度 {level} 样本不足"

    adversarial = [case for case in cases if case["kind"] == "adversarial"]

    assert len(adversarial) >= 3


def test_adversarial_cases_have_no_offline_expectation() -> None:
    """对抗样例不许用夹具回放"证明"模型会拒绝（假绿）。"""

    for case in _cases():
        if case["kind"] == "adversarial":
            assert "offline" not in case, f"{case['id']} 不该有 offline"


def test_offline_cases_point_to_existing_fixtures() -> None:
    fixtures = REPO_ROOT / "packages" / "agent_framework" / "tests" / "fixtures"

    for case in _cases():
        offline = case.get("offline")

        if not offline:
            continue

        fixture = offline["fixture"]
        assert (fixtures / fixture["provider"] / fixture["case"]).is_dir(), case["id"]


def test_repo_references_resolve() -> None:
    """悬空引用会让 W4 的 faithfulness 无从算起。"""

    for case in _cases():
        for item in case.get("references", []):
            if item["kind"] != "repo":
                continue

            path = REPO_ROOT / item["target"]
            assert path.is_file(), f"{case['id']} 引用不存在：{item['target']}"

            if item.get("anchor"):
                assert item["anchor"] in path.read_text("utf-8"), (
                    f"{case['id']} 锚点找不到：{item['anchor']}"
                )


# --------------------------------------------------------------------------- #
# 反向验证：坏了必须红
# --------------------------------------------------------------------------- #


def _mutate(tmp_path: Path, mutate) -> Path:
    lines = GOLDEN.read_text("utf-8").splitlines()
    target = tmp_path / "golden_broken.jsonl"
    target.write_text("\n".join(mutate(lines)) + "\n", "utf-8")

    return target


def test_validator_catches_dangling_reference(tmp_path: Path) -> None:
    def mutate(lines: list[str]) -> list[str]:
        return [
            line.replace('"docs/w3/metric-spec.md"', '"docs/w3/不存在的文件.md"')
            if "gold-r05" in line
            else line
            for line in lines
        ]

    result = _validate(_mutate(tmp_path, mutate))

    assert result.returncode == 1
    assert "不存在的文件" in result.stderr


def test_validator_catches_duplicate_input(tmp_path: Path) -> None:
    """换个 id 抄一遍题干，成功率会被同一道题灌水。"""

    def mutate(lines: list[str]) -> list[str]:
        original = next(line for line in lines if '"gold-c01"' in line)
        return [*lines, original.replace('"gold-c01"', '"gold-c99"')]

    result = _validate(_mutate(tmp_path, mutate))

    assert result.returncode == 1
    assert "重复" in result.stderr


def test_validator_catches_misspelled_expectation_key(tmp_path: Path) -> None:
    """`text_contain`（少个 s）会让断言静默失效——这是最阴的一类错误。"""

    def mutate(lines: list[str]) -> list[str]:
        return [
            line.replace('"text_contains_any"', '"text_contain_any"', 1)
            if "gold-c01" in line
            else line
            for line in lines
        ]

    result = _validate(_mutate(tmp_path, mutate))

    assert result.returncode == 1
    assert "未知判据" in result.stderr


def test_validator_catches_offline_on_adversarial(tmp_path: Path) -> None:
    def mutate(lines: list[str]) -> list[str]:
        patched = []

        for line in lines:
            if '"gold-a01"' in line:
                payload = json.loads(line)
                payload["offline"] = {
                    "fixture": {"provider": "openai", "case": "text_basic"},
                    "expect": {"finish_reason": "stop"},
                }
                line = json.dumps(payload, ensure_ascii=False)

            patched.append(line)

        return patched

    result = _validate(_mutate(tmp_path, mutate))

    assert result.returncode == 1
    assert "对抗样例不得填 offline" in result.stderr


def test_validator_rejects_unknown_tool_symbol(tmp_path: Path) -> None:
    def mutate(lines: list[str]) -> list[str]:
        return [
            line.replace('"search_or_fetch"', '"search_everything"', 1)
            if "gold-t01" in line
            else line
            for line in lines
        ]

    result = _validate(_mutate(tmp_path, mutate))

    assert result.returncode == 1


@pytest.mark.parametrize("needle", ["难度", "总条数"])
def test_validator_catches_size_regression(tmp_path: Path, needle: str) -> None:
    """砍掉一半用例，校验器必须报"总条数/难度分布"不足。"""

    def mutate(lines: list[str]) -> list[str]:
        kept = [line for line in lines if not line.strip().startswith("{")][:1]
        cases = [line for line in lines if line.strip().startswith("{")]
        return [*kept, *cases[:5]]

    result = _validate(_mutate(tmp_path, mutate))

    assert result.returncode == 1
    assert needle in result.stderr
