"""
Chunk 持久化：以 JSONL 格式读写 chunks.jsonl。

JSONL（每行一个 JSON 对象）的好处：
- 支持流式读写，内存占用低；
- 逐行追加方便，出错能定位到具体行号；
- 与下游的 BM25 / 向量索引构建脚本天然契合（逐行读取即可）。

chunks.jsonl 是 ingestion 阶段的核心交付物，也是检索/索引阶段的输入。
"""

import json
from pathlib import Path

from schemas import DocumentChunk


def save_chunks_jsonl(chunks: list[DocumentChunk], out_path: str | Path):
    """把 Chunk 列表写成 JSONL（自动创建父目录，中文不转义）。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open('w', encoding='utf-8') as f:
        for chunk in chunks:
            line = json.dumps(chunk.to_dict(), ensure_ascii=False)
            f.write(line + "\n")


def load_chunks_jsonl(in_path: str | Path) -> list[dict]:
    """逐行读取 JSONL，返回 dict 列表（跳过空行）。"""
    in_path = Path(in_path)
    chunks = []

    # 用 utf-8-sig 自动处理 BOM
    with in_path.open('r', encoding='utf-8-sig') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as e:
                # 抛出带行号和内容的错误，方便定位
                raise ValueError(
                    f"第 {line_no} 行 JSON 解析失败: {e}\n"
                    f"原始内容: {line[:200]!r}"
                ) from e

    return chunks