"""检索指标脚本的自测（W5 D2 / Gate「指标可定义」）。

这一组只测**纯数学与配置翻译**，不加载任何模型（毫秒级）。理由：真正的评测
要跑 18–20s 的 bge-m3 冷启动，塞进 `make test` 会让全量用例从 20s 变成分钟级，
最后所有人都用 `-k` 跳过它——那等于没有测试。

重点验三件事：

1. 公式与 rag_mvp 的历史报告**同算法**（换算法就等于换口径，两边数字不可比）；
2. `top_k` 小于某档位时**不生成那个档的命中标记**（否则 Recall@10 会引用
   根本没检索的档位）；
3. 评测改配置时**只改 `Settings`**，不绕过 `build_pipeline`——这是"口径一致"
   的机械保证，靠自觉是靠不住的。
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "evals" / "scripts" / "retrieval_metrics.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("retrieval_metrics", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # 必须先登记到 sys.modules：脚本里有 `@dataclass(slots=True)`，
    # slots 的数据类在创建时要回查 `sys.modules[cls.__module__]`，
    # 不登记就会 AttributeError: 'NoneType' object has no attribute '__dict__'。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rm() -> ModuleType:
    return _load_script()


# --------------------------------------------------------------------------- #
# 公式
# --------------------------------------------------------------------------- #


def test_first_gold_rank_is_one_based(rm: ModuleType) -> None:
    assert rm.first_gold_rank(["x", "a", "y"], {"a"}) == 2


def test_first_gold_rank_miss_is_none_not_zero(rm: ModuleType) -> None:
    # 用 0 表示未命中会让 `1.0 / rank` 直接 ZeroDivisionError，
    # 或者更糟：被当成"排第 0 位"算成满分。
    assert rm.first_gold_rank(["x", "y"], {"a"}) is None


def test_ndcg_matches_hand_computation(rm: ModuleType) -> None:
    # 单个 gold 排在第一位、只取 @1：DCG=1/log2(2)=1，IDCG 同值 → 1.0
    assert rm.ndcg_at_k(["a", "x"], {"a"}, 1) == pytest.approx(1.0)
    # 单个 gold 排在第二位：DCG=1/log2(3)，IDCG=1 → 0.6309...
    assert rm.ndcg_at_k(["x", "a"], {"a"}, 10) == pytest.approx(0.6309297535714574)
    # 两个 gold 都在前两位：DCG=1+1/log2(3)，IDCG 同为 1+1/log2(3) → 1.0
    assert rm.ndcg_at_k(["a", "b"], {"a", "b"}, 10) == pytest.approx(1.0)


def test_ndcg_prefers_more_gold_in_top_positions(rm: ModuleType) -> None:
    gold = {"a", "b"}
    both = rm.ndcg_at_k(["a", "x", "b"], gold, 10)
    one = rm.ndcg_at_k(["a", "x", "y"], gold, 10)
    assert both > one


def test_ndcg_without_gold_is_zero(rm: ModuleType) -> None:
    # 负例题不该进指标；万一被传进来，返回 0.0 而不是抛错（调用方已按正例过滤）
    assert rm.ndcg_at_k(["a"], set(), 10) == 0.0


def test_recall_denominator_is_samples_not_k(rm: ModuleType) -> None:
    hit = rm.make_outcome(
        query_id="q1", question="", chunk_ids=["a"], gold_ids={"a"}, top_k=10
    )
    miss = rm.make_outcome(
        query_id="q2", question="", chunk_ids=["x"], gold_ids={"a"}, top_k=10
    )
    assert rm.recall_at([hit, miss], 1) == pytest.approx(0.5)
    assert rm.recall_at([hit, miss], 10) == pytest.approx(0.5)
    assert rm.mean_reciprocal_rank([hit, miss]) == pytest.approx(0.5)


def test_empty_rows_return_zero_not_none(rm: ModuleType) -> None:
    # 与 TaskMetrics 的约定不同：这里的 0 确实是"0 条命中 0 条"，
    # 而"没有样本"由调用方（CLI）在跑之前就挡住，不会走到这里。
    assert rm.recall_at([], 1) == 0.0
    assert rm.mean_reciprocal_rank([]) == 0.0


def test_hits_at_k_does_not_claim_untested_ranks(rm: ModuleType) -> None:
    outcome = rm.make_outcome(
        query_id="q1", question="", chunk_ids=["a"], gold_ids={"a"}, top_k=5
    )
    assert set(outcome.hits_at_k) == {"1", "3", "5"}


def test_multi_gold_counts_every_gold(rm: ModuleType) -> None:
    outcome = rm.make_outcome(
        query_id="q3",
        question="",
        chunk_ids=["a", "x", "b"],
        gold_ids={"a", "b"},
        top_k=10,
    )
    assert outcome.gold_count == 2
    assert outcome.first_rank == 1
    # gold 在位置 1 和 3：DCG = 1 + 1/log2(4) = 1.5；IDCG = 1 + 1/log2(3) = 1.631
    # → 0.9197，**不是 1.0**。这条用例当初就是照着"以为是 1.0"写错的，
    # 正好证明 nDCG 会惩罚"gold 排得不够前"，而不只看"命中没命中"。
    assert outcome.ndcg_at_10 == pytest.approx(0.9197207891481876)


# --------------------------------------------------------------------------- #
# 配置解析
# --------------------------------------------------------------------------- #


def test_parse_config_label_order_is_canonical(rm: ModuleType) -> None:
    # 标签里的顺序不影响 key：`dense+rerank` 与 `dense+rewrite+rerank`
    # 在报告里必须是同一个键，否则同一档位会出现两行。
    assert rm.parse_config("dense+rerank").key == "dense+rerank"
    assert rm.parse_config("dense+rewrite+rerank").key == "dense+rerank+rewrite"


@pytest.mark.parametrize("label", ["denes", "dense+banana", "", "  "])
def test_parse_config_rejects_typos(rm: ModuleType, label: str) -> None:
    # 拼错的档位被静默忽略，报告里就会出现一行"看起来正常、跑的是别的档位"的数据
    with pytest.raises(ValueError):
        rm.parse_config(label)


def test_settings_override_only_touches_settings(rm: ModuleType) -> None:
    """评测档位必须翻译成 `Settings`，而不是直接构造 pipeline。

    `build_pipeline` 是唯一的装配点。评测若自己拼 `budget_ms=...`，
    改了生产装配忘了这边，报告上的数字就不再是生产口径。
    """

    from agent_framework.config import Settings

    config = rm.parse_config("dense+rerank")
    settings = rm._settings_for(
        config, top_k=10, budget_ms=6000.0, recall_multiplier=3.0, rerank_candidates=20
    )

    assert isinstance(settings, Settings)
    assert settings.retrieval_mode == "dense"
    assert settings.retrieval_rerank is True
    assert settings.retrieval_top_k == 10
    assert settings.retrieval_budget_ms == 6000.0
    # 没指定的字段保持 Settings 默认（生产口径），不在这里另立一套默认值
    assert settings.retrieval_rewrite is False


def test_default_mode_is_dense(rm: ModuleType) -> None:
    """默认档是 dense——由六档实测选出，不是拍脑袋（见 docs/w5/rag-ab-report.md）。"""

    from agent_framework.config import Settings

    assert Settings().retrieval_mode == "dense"


# --------------------------------------------------------------------------- #
# 报告告警：降级过的样本不能被当成"该组件无效"
# --------------------------------------------------------------------------- #


def test_report_warns_when_any_config_degraded(rm: ModuleType) -> None:
    rows = [
        rm.make_outcome(
            query_id="q1",
            question="",
            chunk_ids=["a"],
            gold_ids={"a"},
            top_k=10,
            degraded=("rerank_degraded",),
        )
    ]
    report = rm._summarize(
        rm.parse_config("dense+rerank"),
        rows,
        top_k=10,
        recall_multiplier=3.0,
        rerank_candidates=20,
        budget_ms=800.0,
        negative_count=2,
    )
    assert report.degraded_count == 1

    class _Dataset:
        dataset_id = "fake"
        version = 1
        corpus: Any = None

    payload = rm._report_payload(_Dataset(), [report], top_k=10, warmed=True)
    assert payload["warning"] is not None
    assert "dense+rerank" in payload["warning"]


def test_report_has_no_warning_when_clean(rm: ModuleType) -> None:
    rows = [
        rm.make_outcome(
            query_id="q1", question="", chunk_ids=["a"], gold_ids={"a"}, top_k=10
        )
    ]
    report = rm._summarize(
        rm.parse_config("dense"),
        rows,
        top_k=10,
        recall_multiplier=3.0,
        rerank_candidates=20,
        budget_ms=800.0,
        negative_count=2,
    )

    class _Dataset:
        dataset_id = "fake"
        version = 1
        corpus: Any = None

    assert rm._report_payload(_Dataset(), [report], top_k=10, warmed=True)["warning"] is None


# --------------------------------------------------------------------------- #
# CLI（子进程，测的是退出码——CI 的判据就是它）
# --------------------------------------------------------------------------- #


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_cli_self_test_exits_zero() -> None:
    result = _run("--self-test")
    assert result.returncode == 0, result.stderr
    assert "self-test ok" in result.stdout


def test_cli_missing_dataset_exits_two() -> None:
    # 2 = 环境不满足。用 0 表示"没数据也没关系"会让 CI 在索引缺失时静默变绿。
    result = _run("--dataset", "/nonexistent/golden.yaml", "--configs", "dense")
    assert result.returncode == 2
    assert "找不到 golden 数据集" in result.stderr


def test_cli_bad_config_exits_nonzero_before_loading_models() -> None:
    result = _run("--configs", "denes", "--dataset", "/nonexistent/golden.yaml")
    # 解析配置在加载数据集/模型之前，异常直接冒出来（argparse 之后的第一件事）
    assert result.returncode != 0
    assert "denes" in (result.stderr + result.stdout)
