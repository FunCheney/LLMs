from ingestion.storage import load_chunks_jsonl


def main():
    chunks = load_chunks_jsonl(
        "data/processed/chunks.jsonl"
    )

    print(f"Total chunks: {len(chunks)}")

    for chunk in chunks[:3]:
        print("=" * 60)
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Page: {chunk['page_start']}-{chunk['page_end']}")
        print(f"Content: {chunk['content'][:200]}")


if __name__ == "__main__":
    main()