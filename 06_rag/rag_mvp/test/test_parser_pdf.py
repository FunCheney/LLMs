"""手工冒烟脚本（不是 pytest 用例，只有 main()）。

pytest 会因为文件名收集它并在 import 阶段执行顶层语句，所以必须先探测
可选依赖：缺 `pdf` extra 时跳过，而不是 collection error。
"""

import pytest

pytest.importorskip("pymupdf", reason="需要 pdf extra：uv sync --extra pdf")

from rag_mvp.ingestion.pdf_parser import parse_pdf, summarize_pages  # noqa: E402


def main():
    pdf_path = "../data/raw/Claude Code设计指南.pdf"

    pages = parse_pdf(pdf_path)

    print(f"Total pages: {len(pages)}")

    summary = summarize_pages(pages)

    print(f"Total summary: {summary}")

if __name__ == "__main__":
    main()

