import pytest

from ingestion.text_splitter import split_text


def test_split_text_without_overlap():
    text = "ABCDEFGHIJ"

    chunks = split_text(
        text=text,
        chunk_size=5,
        overlap=0,
    )

    assert chunks == [
        "ABCDE",
        "FGHIJ",
    ]


def test_split_text_with_overlap():
    text = "ABCDEFGHIJ"

    chunks = split_text(
        text=text,
        chunk_size=6,
        overlap=2,
    )

    assert chunks == [
        "ABCDEF",
        "EFGHIJ",
    ]


def test_empty_text():
    assert split_text(
        text="",
        chunk_size=10,
        overlap=2,
    ) == []


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        split_text(
            text="hello",
            chunk_size=0,
            overlap=0,
        )


def test_invalid_overlap():
    with pytest.raises(ValueError):
        split_text(
            text="hello",
            chunk_size=5,
            overlap=5,
        )