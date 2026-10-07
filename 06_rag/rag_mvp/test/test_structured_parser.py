from ingestion.structured_parser import parse_structured_paragraphs


def test_attaches_section_metadata():
    pages = [
        {
            "page_number": 1,
            "text": "1.1 Background\nConsumer Groups are logical groups of consumers.",
        }
    ]

    paragraphs = parse_structured_paragraphs(pages)

    assert len(paragraphs) == 1
    assert paragraphs[0].section_title == "1.1 Background"
    assert paragraphs[0].section_path == ["1.1 Background"]
    assert "Consumer Groups" in paragraphs[0].content
    assert "1.1 Background" not in paragraphs[0].content


def test_paragraph_can_cross_pages_with_section():
    pages = [
        {
            "page_number": 7,
            "text": "1.1 Background\nConsumer Groups are logical groups of consumers.",
        },
        {
            "page_number": 8,
            "text": "Consumers read records from partitions.",
        },
    ]

    paragraphs = parse_structured_paragraphs(pages)

    assert len(paragraphs) == 1
    assert paragraphs[0].page_start == 7
    assert paragraphs[0].page_end == 8
    assert paragraphs[0].section_title == "1.1 Background"


def test_new_heading_starts_new_paragraph():
    pages = [
        {
            "page_number": 1,
            "text": "1.1 A\nShort A.\n1.2 B\nShort B.",
        }
    ]

    paragraphs = parse_structured_paragraphs(pages)

    assert len(paragraphs) == 2
    assert paragraphs[0].section_title == "1.1 A"
    assert paragraphs[1].section_title == "1.2 B"


def test_skips_page_number_noise():
    pages = [
        {
            "page_number": 13,
            "text": "1.1 一句话解释\nClaude Code 是执行工具。\n01",
        }
    ]

    paragraphs = parse_structured_paragraphs(pages)

    assert len(paragraphs) == 1
    assert "01" not in paragraphs[0].content
