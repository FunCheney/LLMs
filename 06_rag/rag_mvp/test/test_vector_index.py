from embedding import BGEM3EmbeddingProvider
from vector_store import LocalVectorIndex, VectorRecord

def main():
    embedding = BGEM3EmbeddingProvider()
    index = LocalVectorIndex()

    documents = [
        (
            "chunk_001",
            "Python 是一种高级编程语言，广泛用于数据分析和人工智能。",
        ),
        (
            "chunk_002",
            "PDF 是一种常见的电子文档格式，可以保存文字、图片和表格。",
        ),
        (
            "chunk_003",
            "Python 可以使用 PyMuPDF、pypdf 等库读取 PDF 文档。",
        ),
    ]

    texts = [text for _,text in documents]
    embeddings = embedding.embed_documents(texts)

    for (chunk_id, text), embedding in zip(documents, embeddings):
        index.add(VectorRecord(chunk_id, text,  metadata={}, embedding=embedding.vector,))

    print("index size:", index.count())
    query = "Python 如何读取 PDF？"

    query_embedding = embedding.embed_text(query)
    results = index.search(query_embedding.vector,  top_k=2,)
    print("\nQuery:", query)

    print("\nResults:")

    for record, score in results:
        print(
            f"\nscore={score:.4f}"
            f"\nid={record.id}"
            f"\ntext={record.text}"
        )


if __name__ == "__main__":
    main()


