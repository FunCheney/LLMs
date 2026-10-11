"""
PDF 解析器：把 PDF 文件转成「按页」的文本列表。

输出结构（pages）是整条 ingestion 流水线的统一中间格式，
下游的 heading_detector / structured_parser / chunker 都基于它工作：

    [{"page_number": 1, "text": "第 1 页的正文..."},
     {"page_number": 2, "text": "第 2 页的正文..."}, ...]

注意：这里只负责「抽取文本 + 保留页码」，不做分块，也不做结构识别。
"""

from pathlib import Path
import pymupdf

def parse_pdf(pdf_path: str | Path) -> list[dict]:
    """解析 PDF，返回逐页文本（page_number 从 1 开始计数）。"""
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file {pdf_path} does not exist")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"PDF file {pdf_path} is not a PDF file")

    pages = []
    with pymupdf.open(pdf_path) as doc:
        for page_index, page in enumerate(doc):
            # "text" 模式按阅读顺序抽取纯文本；strip 掉首尾空白及孤立换行
            text = page.get_text("text")
            pages.append({
                "page_number": page_index + 1,
                "text": text.strip(),
            })


    return pages


def summarize_pages(pages: list[dict]) -> dict:
    """汇总解析质量指标：总页数、非空页数、总字符数（用于快速判断解析是否异常）。"""
    non_empty_pages = [
        page for page in pages
        if page["text"]
    ]

    total_chars = sum(
        len(page["text"])
        for page in pages
    )

    return {
        "total_pages": len(pages),
        "non_empty_pages": len(non_empty_pages),
        "total_chars": total_chars,
    }

