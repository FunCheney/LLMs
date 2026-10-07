from embedding import BGEM3EmbeddingProvider


def cosine_similarity(a: list[float], b: list[float]) -> float:
    # 这里为什么可以直接计算，而不是完整的 计算 cosine similarity
    # 因为：normalize_embeddings=True 两个向量都已经是单位向量。因此： cos(A,B)=A⋅B
    return sum(x * y for x, y in zip(a, b))

def main():
    embedding = BGEM3EmbeddingProvider()

    texts = [
        "Python 是一种高级编程语言",
        "Python 是一种非常流行的编程语言",
        "今天天气非常好",
    ]

    results = embedding.embed_documents(texts)

    for i, result in enumerate(results):
        print(i, result.dimension)

    similarity_01 = cosine_similarity(
        results[0].vector,
        results[1].vector,
    )

    similarity_02 = cosine_similarity(
        results[0].vector,
        results[2].vector,
    )

    print("Python vs Python:", similarity_01)
    print("Python vs 天气:", similarity_02)


if __name__ == "__main__":
    main()