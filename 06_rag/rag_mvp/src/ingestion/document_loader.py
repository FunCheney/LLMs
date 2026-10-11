"""
文档加载入口：根据文件后缀选择对应的解析器。

统一返回 (pages, source_type)：
- pages：与 PDF/HTML 解析器一致的逐页文本结构；
- source_type：源文件类型（"pdf" / "html"），会写进 Chunk 元数据用于溯源。
"""

from pathlib import Path

from ingestion.html_parser import parse_html
from ingestion.pdf_parser import parse_pdf


def load_pages(path: str | Path) -> tuple[list[dict], str]:
    """按后缀分派解析器；不支持的格式抛 ValueError。"""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path), "pdf"
    if suffix in {".html", ".htm"}:
        return parse_html(path), "html"
    raise ValueError(f"Unsupported document type: {path.suffix}")
