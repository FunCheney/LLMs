"""
示例：挑出「跨页 Chunk」（page_start != page_end）并打印，用于检查跨页合并是否合理。
"""

from pathlib import Path

from ingestion.storage import load_chunks_jsonl


def main():
    chunks = load_chunks_jsonl(
        Path("data/processed/chunks.jsonl")
    )

    cross_page_count = 0

    for chunk in chunks:
        if chunk["page_start"] != chunk["page_end"]:
            cross_page_count += 1

            print("=" * 60)
            print(
                f"Pages: "
                f"{chunk['page_start']} - "
                f"{chunk['page_end']}"
            )
            print(chunk["content"][:500])

    print(f"Cross-page chunks: {cross_page_count}")


if __name__ == "__main__":
    main()