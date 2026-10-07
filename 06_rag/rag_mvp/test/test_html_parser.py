from ingestion.html_parser import html_to_text, parse_html
from ingestion.pipeline import ingest_document


def test_html_to_text_keeps_markdown_headings(tmp_path):
    html = """
    <html>
      <head><script>ignore()</script></head>
      <body>
        <h1>安装说明</h1>
        <p>先安装依赖。</p>
        <h2>环境变量</h2>
        <p>需要配置 API Key。</p>
      </body>
    </html>
    """
    html_path = tmp_path / "guide.html"
    html_path.write_text(html, encoding="utf-8")

    pages = parse_html(html_path)
    assert pages[0]["page_number"] == 1
    text = pages[0]["text"]
    assert "# 安装说明" in text
    assert "## 环境变量" in text
    assert "ignore()" not in text

    chunks, report = ingest_document(
        path=html_path,
        document_id="html-001",
        document_title="安装说明",
        chunk_size=200,
        overlap=20,
    )
    assert report.ok
    assert chunks
    assert chunks[0].source_type == "html"
    assert any(chunk.section_title == "安装说明" for chunk in chunks)


def test_html_to_text_helper():
    text = html_to_text("<h1>Title</h1><p>Hello</p>")
    assert text.splitlines()[0] == "# Title"
    assert "Hello" in text
