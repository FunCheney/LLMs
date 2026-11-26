"""W7 D4 Drift 三指标的单测。三指标各一个可触发的构造 + 纯规则（零 LLM）。"""

from __future__ import annotations

from agent_framework.context import check_drift, terms


class TestGoalEcho:
    def test_goal_present_high_echo(self) -> None:
        report = check_drift(
            goal="写一篇小红书品牌种草文案",
            payload_texts=["任务：写小红书品牌种草文案，注意调性"],
        )
        assert report.goal_echo >= 0.3
        assert "goal_lost" not in report.alerts

    def test_goal_compressed_away(self) -> None:
        """目标关键词从 payload 里消失 → goal_lost + 建议重注入。"""
        report = check_drift(
            goal="写一篇小红书品牌种草文案",
            payload_texts=["今天天气不错，我们来聊聊美食吧"],
        )
        assert report.goal_echo < 0.3
        assert "goal_lost" in report.alerts
        assert any("任务目标重注入" in r for r in report.reinject)

    def test_terms_bigrams(self) -> None:
        t = terms("品牌调性 brand_tone")
        assert "品牌" in t and "牌调" in t and "brand_tone" in t


class TestSourceLoss:
    def test_all_sources_kept(self) -> None:
        report = check_drift(
            goal="g",
            payload_texts=["x"],
            cited_chunk_ids=["a", "b", "c", "d"],
            in_window_ids=["a", "b", "c", "d"],
        )
        assert report.source_retention == 1.0
        assert "source_loss" not in report.alerts

    def test_sources_squeezed_out(self) -> None:
        """窗口内只剩 <50% 的旧证据 → source_loss。"""
        report = check_drift(
            goal="g",
            payload_texts=["x"],
            cited_chunk_ids=["a", "b", "c", "d"],
            in_window_ids=["a"],
        )
        assert report.source_retention == 0.25
        assert "source_loss" in report.alerts
        assert any("证据" in r for r in report.reinject)

    def test_no_history_no_alert(self) -> None:
        report = check_drift("g", ["g"])
        assert report.source_retention == 1.0


class TestRepeatAction:
    def test_repeat_detected(self) -> None:
        report = check_drift(
            "g",
            ["g"],
            recent_actions=[("search", "h1"), ("search", "h1"), ("draft", "h2")],
        )
        assert report.max_repeat == 2
        assert "repeat_action" in report.alerts
        # 死循环不重注入（重注入解决不了循环）。
        assert report.reinject == ()

    def test_varied_actions_ok(self) -> None:
        report = check_drift(
            "g",
            ["g"],
            recent_actions=[("search", "h1"), ("draft", "h1"), ("search", "h2")],
        )
        assert "repeat_action" not in report.alerts

    def test_window_is_three(self) -> None:
        """只看最近 3 步：更早的重复不算。"""
        report = check_drift(
            "g",
            ["g"],
            recent_actions=[("search", "h1"), ("search", "h1"), ("draft", "a"), ("draft", "b")],
        )
        assert "repeat_action" not in report.alerts


class TestZeroCost:
    def test_pure_rules_no_llm(self) -> None:
        """三指标都不调模型：任意输入瞬间返回（无网络、无 sleep）。"""
        report = check_drift("任意目标", ["任意内容"] * 100, ["x"], [], [("t", "a")] * 10)
        assert report is not None
