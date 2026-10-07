from html.parser import HTMLParser
from pathlib import Path


class HTMLTextExtractor(HTMLParser):
    SKIP_TAGS = {"script", "style", "noscript"}
    BLOCK_TAGS = {
        "p", "div", "br", "li", "ul", "ol", "tr", "table",
        "section", "article", "header", "footer", "blockquote",
    }
    HEADING_TAGS = {f"h{level}" for level in range(1, 7)}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0
        self._heading_level = 0
        self._heading_buf: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in self.HEADING_TAGS:
            self._heading_level = int(tag[1])
            self._heading_buf = []
        elif tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag in self.HEADING_TAGS:
            title = "".join(self._heading_buf).strip()
            if title:
                self.parts.append("\n" + "#" * self._heading_level + " " + title + "\n")
            self._heading_level = 0
            self._heading_buf = []
        elif tag in {"p", "div", "li", "tr", "section", "article", "blockquote"}:
            self.parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._heading_level:
            self._heading_buf.append(data)
        else:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    extractor = HTMLTextExtractor()
    extractor.feed(html)
    extractor.close()
    text = "".join(extractor.parts)
    lines = [line.rstrip() for line in text.splitlines()]
    compacted: list[str] = []
    blank_run = 0
    for line in lines:
        if not line.strip():
            blank_run += 1
            if blank_run <= 1:
                compacted.append("")
            continue
        blank_run = 0
        compacted.append(line.strip())
    return "\n".join(compacted).strip()


def parse_html(html_path: str | Path) -> list[dict]:
    html_path = Path(html_path)
    if not html_path.exists():
        raise FileNotFoundError(f"HTML file {html_path} does not exist")
    if html_path.suffix.lower() not in {".html", ".htm"}:
        raise ValueError(f"{html_path} is not an HTML file")

    text = html_to_text(html_path.read_text(encoding="utf-8"))
    return [{"page_number": 1, "text": text}]
