import pytest

from embedding.fake import FakeEmbeddingProvider


def test_embed_text_should_return_fixed_dimension():
    provider = FakeEmbeddingProvider(
        dimension=4,
    )

    result = provider.embed_text(
        "What is Kafka?"
    )

    assert len(result.vector) == 4
    assert all(
        isinstance(value, float)
        for value in result.vector
    )


def test_embed_documents_should_return_same_number_of_vectors():
    provider = FakeEmbeddingProvider(
        dimension=4,
    )

    texts = [
        "Kafka",
        "Producer",
        "Consumer",
    ]

    results = provider.embed_documents(texts)

    assert len(results) == len(texts)

    assert all(
        len(result.vector) == 4
        for result in results
    )


def test_empty_text_should_raise_error():
    provider = FakeEmbeddingProvider(
        dimension=4,
    )

    with pytest.raises(ValueError):
        provider.embed_text("")