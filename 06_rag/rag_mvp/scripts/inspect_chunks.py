from src.ingestion.storage import load_chunks_jsonl


def main():
    chunks = load_chunks_jsonl(
        "data/processed/chunks.jsonl"
    )

    print(f"Total chunks: {len(chunks)}")

    for chunk in chunks[:5]:
        print("=" * 60)
        print(f"ID: {chunk['chunk_id']}")
        print(
            f"Page: {chunk['page_start']}-"
            f"{chunk['page_end']}"
        )
        print(f"Length: {len(chunk['content'])}")
        print(f"Content: {chunk['content'][:200]}")


if __name__ == "__main__":
    main()