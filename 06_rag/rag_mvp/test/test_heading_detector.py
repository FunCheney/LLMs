from ingestion.heading_detector import (
    Heading,
    SectionTracker,
    detect_heading,
)


def test_section_tracker():
    tracker = SectionTracker()

    tracker.update(Heading("1 Introduction", 1))
    tracker.update(Heading("1.1 Background", 2))

    assert tracker.get_path() == [
        "1 Introduction",
        "1.1 Background",
    ]

    tracker.update(Heading("1.2 Motivation", 2))
    assert tracker.get_path() == [
        "1 Introduction",
        "1.2 Motivation",
    ]

    tracker.update(Heading("2 Architecture", 1))
    assert tracker.get_path() == [
        "2 Architecture",
    ]


def test_detect_subsection_heading():
    heading = detect_heading("1.1 一句话解释")
    assert heading is not None
    assert heading.title == "1.1 一句话解释"
    assert heading.level == 3


def test_detect_chinese_chapter_and_part():
    part = detect_heading("第一部分 认识 Claude Code（小白友好）")
    chapter = detect_heading("第 1 章 Claude Code 是什么")
    assert part is not None and part.level == 1
    assert chapter is not None and chapter.level == 2


def test_detect_spaced_pdf_headings():
    assert detect_heading("C H AP T ER 1").title == "Chapter 1"
    assert detect_heading("P R EFAC E").title == "Preface"
    assert detect_heading("PA RT I I I").title == "Part III"


def test_list_item_is_not_heading():
    assert detect_heading("1. 用 GlobTool 找到所有 .js / .ts 文件") is None
    assert detect_heading("4. 修改代码") is None
