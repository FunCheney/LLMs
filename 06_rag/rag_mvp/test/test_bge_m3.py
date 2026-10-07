from embedding import BGEM3EmbeddingProvider


def main():
    embedding = BGEM3EmbeddingProvider()

    result = embedding.embed_text(
        "Python 是一种高级编程语言"
    )

    print("model:", result.model)
    print("dimension:", result.dimension)
    print("vector length:", len(result.vector))
    print("first 10 values:", result.vector[:10])


if __name__ == "__main__":
    main()