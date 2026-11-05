"""
HTML 解析器：把 HTML 转成与 PDF 一致的结构化文本。

核心思路（HTMLTextExtractor）：
- 用标准库 HTMLParser 做流式解析，不依赖第三方库；
- 把 h1~h6 标题还原成 Markdown 标题（"## 标题"），
  这样下游 heading_detector 可以用统一的正则识别标题；
- 用 <p>/<div>/<li> 等块级标签还原段落边界（插入空行）；
- 跳过 <script>/<style>/<noscript> 里的内容，避免把代码/样式当正文。

输出同样是 pages 列表，但 HTML 无天然页码，统一记为第 1 页。
"""

from html.parser import HTMLParser
from pathlib import Path


class HTMLTextExtractor(HTMLParser):
    """把 HTML 流式解析为「标题 + 正文」混合的纯文本片段。"""

    # 这些标签内的内容整体丢弃（脚本、样式等非正文内容）
    SKIP_TAGS = {"script", "style", "noscript"}
    # 这些是块级标签，前后插入换行以还原段落边界
    BLOCK_TAGS = {
        "p", "div", "br", "li", "ul", "ol", "tr", "table",
        "section", "article", "header", "footer", "blockquote",
    }
    # h1~h6 视为标题
    HEADING_TAGS = {f"h{level}" for level in range(1, 7)}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0            # 处于跳过标签内部的嵌套深度
        self._heading_level = 0         # 当前正在收集的标题级别（0 表示不在标题内）
        self._heading_buf: list[str] = []  # 标题文本缓冲

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in self.HEADING_TAGS:
            # 进入标题：记录级别并重置缓冲
            self._heading_level = int(tag[1])
            self._heading_buf = []
        elif tag in self.BLOCK_TAGS:
            # 块级元素开始：插入换行作为软分隔
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag in self.HEADING_TAGS:
            # 标题结束：输出为 Markdown 标题行，交给下游 heading_detector 识别
            title = "".join(self._heading_buf).strip()
            if title:
                self.parts.append("\n" + "#" * self._heading_level + " " + title + "\n")
            self._heading_level = 0
            self._heading_buf = []
        elif tag in {"p", "div", "li", "tr", "section", "article", "blockquote"}:
            # 段落级元素结束：插入空行作为硬分段
            self.parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._heading_level:
            self._heading_buf.append(data)
        else:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    """把 HTML 字符串转成规整纯文本：压缩连续空行、去除行首尾空白。"""
    extractor = HTMLTextExtractor()
    extractor.feed(html)
    extractor.close()
    text = "".join(extractor.parts)
    lines = [line.rstrip() for line in text.splitlines()]
    compacted: list[str] = []
    blank_run = 0
    for line in lines:
        if not line.strip():
            # 连续空白行只保留一行，避免过多噪声干扰段落切分
            blank_run += 1
            if blank_run <= 1:
                compacted.append("")
            continue
        blank_run = 0
        compacted.append(line.strip())
    return "\n".join(compacted).strip()


def parse_html(html_path: str | Path) -> list[dict]:
    """解析 HTML 文件，返回单页文本（HTML 无页码，统一记为第 1 页）。"""
    html_path = Path(html_path)
    if not html_path.exists():
        raise FileNotFoundError(f"HTML file {html_path} does not exist")
    if html_path.suffix.lower() not in {".html", ".htm"}:
        raise ValueError(f"{html_path} is not an HTML file")

    text = html_to_text(html_path.read_text(encoding="utf-8"))
    return [{"page_number": 1, "text": text}]
