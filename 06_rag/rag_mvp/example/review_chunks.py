from pathlib import Path

from ingestion.storage import load_chunks_jsonl
from validation.chunk_validator import validate_chunks


def _print_chunk(chunk: dict, index: int) -> None:
    print("=" * 60)
    print(f"[{index}] {chunk['chunk_id']}")
    print(f"Page: {chunk['page_start']}-{chunk['page_end']}")
    print(f"Section: {chunk.get('section_title')}")
    print(f"Path: {' > '.join(chunk.get('section_path') or [])}")
    print(f"Source: {chunk.get('source_uri')} ({chunk.get('source_type')})")
    print(f"Length: {len(chunk.get('content') or '')}")
    print((chunk.get("content") or "")[:300])


def main():
    chunks = load_chunks_jsonl(Path("data/processed/chunks.jsonl"))
    report = validate_chunks(chunks)
    print(f"Total chunks: {len(chunks)}")
    print("stats:", report.stats)

    if not chunks:
        return

    step = max(1, len(chunks) // 20)
    sampled = chunks[::step][:20]
    print(f"\nReviewing {len(sampled)} chunks:\n")
    for index, chunk in enumerate(sampled, start=1):
        _print_chunk(chunk, index)


if __name__ == "__main__":
    main()
