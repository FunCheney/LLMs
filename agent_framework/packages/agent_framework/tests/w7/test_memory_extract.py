"""W7 D3 记忆提取的单测：JSON 解析容错 / 信任级标注 / 显式确认。"""

from __future__ import annotations

from agent_framework.memory import parse_extraction


class TestParseExtraction:
    def test_clean_json(self) -> None:
        raw = (
            '[{"key": "brand_tone", "value": "专业克制", "confidence": 0.9,'
            ' "evidence_role": "user", "evidence": "记住调性要专业"}]'
        )
        items = parse_extraction(raw, source_session="s1")
        assert len(items) == 1
        assert items[0].key == "brand_tone"
        assert items[0].trust == "user"
        # 显式"记住"且证据是用户原话 → confirmed。
        assert items[0].confirmed_by_user is True

    def test_markdown_fenced_json(self) -> None:
        raw = (
            "```json\n"
            '[{"key": "forbidden_words", "value": "最、第一", "confidence": 0.85,'
            ' "evidence_role": "user", "evidence": "以后不要用最字"}]\n```'
        )
        items = parse_extraction(raw, source_session="s1")
        assert len(items) == 1
        assert items[0].confirmed_by_user is True

    def test_chatter_wrapped_json(self) -> None:
        raw = (
            '好的，提取结果如下：[{"key": "past_topics", "value": "露营",'
            ' "confidence": 0.6, "evidence_role": "assistant", "evidence": "聊过露营"}] 以上。'
        )
        items = parse_extraction(raw, source_session="s1")
        assert len(items) == 1
        # 证据来自 assistant → trust=external（写入时默认被拒）。
        assert items[0].trust == "external"
        assert items[0].confirmed_by_user is False

    def test_empty_array(self) -> None:
        assert parse_extraction("[]", source_session="s1") == []

    def test_garbage_returns_empty(self) -> None:
        assert parse_extraction("模型跑偏了没有输出数组", source_session="s1") == []
        assert parse_extraction("[broken json", source_session="s1") == []

    def test_unknown_key_dropped(self) -> None:
        raw = (
            '[{"key": "random_key", "value": "x", "confidence": 0.9,'
            ' "evidence_role": "user", "evidence": "y"}]'
        )
        assert parse_extraction(raw, source_session="s1") == []

    def test_max_items_capped(self) -> None:
        raw = (
            "[" + ",".join(
                f'{{"key": "past_topics", "value": "选题{i}", "confidence": 0.5,'
                f' "evidence_role": "user", "evidence": "e{i}"}}'
                for i in range(10)
            ) + "]"
        )
        items = parse_extraction(raw, source_session="s1")
        assert len(items) == 5  # EXTRACT_MAX_ITEMS

    def test_sorted_by_confidence(self) -> None:
        raw = (
            '[{"key": "brand_tone", "value": "低", "confidence": 0.5,'
            ' "evidence_role": "user", "evidence": "e"},'
            '{"key": "forbidden_words", "value": "高", "confidence": 0.9,'
            ' "evidence_role": "user", "evidence": "记住"}]'
        )
        items = parse_extraction(raw, source_session="s1")
        assert items[0].confidence >= items[1].confidence

    def test_confidence_clamped(self) -> None:
        raw = (
            '[{"key": "brand_tone", "value": "x", "confidence": 2.0,'
            ' "evidence_role": "user", "evidence": "e"}]'
        )
        items = parse_extraction(raw, source_session="s1")
        assert items[0].confidence == 1.0
