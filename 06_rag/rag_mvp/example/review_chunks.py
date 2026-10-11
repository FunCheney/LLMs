"""
示例：人工抽检 chunks.jsonl（Phase 1 交付物验收用）。

先跑一遍质量校验打印统计，再等距抽取约 20 条逐条打印，
便于快速判断「章节识别、页码、正文切分」是否符合预期。
"""

from pathlib import Path

from ingestion.storage import load_chunks_jsonl
from validation.chunk_validator import validate_chunks


def _print_chunk(chunk: dict, index: int) -> None:
    """打印单条 Chunk 的关键字段（截断正文）。"""
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

    # 等距抽样最多 20 条，覆盖整个文档而非只看开头
    step = max(1, len(chunks) // 20)
    sampled = chunks[::step][:20]
    print(f"\nReviewing {len(sampled)} chunks:\n")
    for index, chunk in enumerate(sampled, start=1):
        _print_chunk(chunk, index)


if __name__ == "__main__":
    main()
