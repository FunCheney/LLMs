from  embedding import BGEM3EmbeddingProvider
from retrieval import MyRetriever
from vector_store import LocalVectorIndex

def main():
    embedding = BGEM3EmbeddingProvider()
    index = LocalVectorIndex.load("data/vector_index.pkl")
    retriever = MyRetriever(embedding=embedding, index=index)

    query = "Python 如何读取 pdf?"

    results = retriever.retrieve(query, top_k=3)

    for i, result in enumerate(results):
        print(f'------result{i}------')
        print(f"score: {result.score:.4f}")
        print(f"fis: {result.chunk_id}")
        print(f"text: {result.text}")
        print(f"metadata: {result.metadata}")
        print()

if __name__ == "__main__":
    main()
