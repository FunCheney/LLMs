import pytest

from embedding.fake import FakeEmbeddingProvider


def test_embed_text_should_return_fixed_dimension():
    provider = FakeEmbeddingProvider(
        dimension=4,
    )

    vector = provider.embed_text(
        "What is Kafka?"
    )

    assert len(vector) == 4
    assert all(
        isinstance(value, float)
        for value in vector
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

    vectors = provider.embed_documents(texts)

    assert len(vectors) == len(texts)

    assert all(
        len(vector) == 4
        for vector in vectors
    )


def test_empty_text_should_raise_error():
    provider = FakeEmbeddingProvider(
        dimension=4,
    )

    with pytest.raises(ValueError):
        provider.embed_text("")