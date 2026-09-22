import pytest

from ingestion.paragraph_splitter import (
    split_paragraphs,
    split_text_by_paragraphs,
)


def test_split_paragraphs():
    text = """
    Paragraph A.

    Paragraph B.


    Paragraph C.
    """

    paragraphs = split_paragraphs(text)

    assert paragraphs == [
        "Paragraph A.",
        "Paragraph B.",
        "Paragraph C.",
    ]


def test_merge_short_paragraphs():
    text = (
        "Paragraph A.\n\n"
        "Paragraph B.\n\n"
        "Paragraph C."
    )

    chunks = split_text_by_paragraphs(
        text=text,
        chunk_size=100,
        overlap=10,
    )

    assert len(chunks) == 1
    assert "Paragraph A." in chunks[0]
    assert "Paragraph B." in chunks[0]
    assert "Paragraph C." in chunks[0]


def test_split_when_combined_size_exceeds_limit():
    text = (
        "A" * 40
        + "\n\n"
        + "B" * 40
        + "\n\n"
        + "C" * 40
    )

    # 每个段落长度是 40，两个段落加上中间的 \n\n 后长度是：40 + 2 + 40 = 82
    # 所以前两个段落可以合并；加入第三个段落后会超过 85，因此会形成两个 Chunk。
    chunks = split_text_by_paragraphs(
        text=text,
        chunk_size=85,
        overlap=10,
    )

    assert len(chunks) == 2
    assert all(len(chunk) <= 85 for chunk in chunks)


def test_long_paragraph_uses_character_splitter():
    text = "A" * 100

    chunks = split_text_by_paragraphs(
        text=text,
        chunk_size=40,
        overlap=10,
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= 40 for chunk in chunks)


def test_empty_text_returns_empty_list():
    chunks = split_text_by_paragraphs(
        text="",
        chunk_size=100,
        overlap=10,
    )

    assert chunks == []

# 参数化装饰器
# 让同一个测试函数用多组不同的参数各跑一遍
@pytest.mark.parametrize(
    "chunk_size, overlap", # 参数名（对应测试函数的参数）
    [                      # 组合列表
        (0, 0),
        (-1, 0),
        (100, -1),
        (100, 100),
        (100, 101),
    ],
)
def test_invalid_parameters(
    chunk_size: int,
    overlap: int,
):
    with pytest.raises(ValueError):
        split_text_by_paragraphs(
            text="some text",
            chunk_size=chunk_size,
            overlap=overlap,
        )


if __name__ == "__main__":
    test_merge_short_paragraphs()
    test_invalid_parameters()