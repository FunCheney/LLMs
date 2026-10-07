from dataclasses import dataclass
import re


@dataclass
class Heading:
    title: str
    level: int


MARKDOWN_HEADING_PATTERN = re.compile(
    r"^\s*(#{1,6})\s+(.+?)\s*$"
)

# 至少带一层点号，避免把 "1. 用 GlobTool ..." 这种正文列表当成标题。
SUBSECTION_PATTERN = re.compile(
    r"^\s*(\d+\.\d+(?:\.\d+)*)\s+(\S.*?)\s*$"
)

CN_PART_PATTERN = re.compile(
    r"^\s*(第\s*[一二三四五六七八九十百零〇0-9]+\s*部分(?:\s+\S.*)?)\s*$"
)

CN_CHAPTER_PATTERN = re.compile(
    r"^\s*(第\s*[一二三四五六七八九十百零〇0-9]+\s*章(?:\s+\S.*)?)\s*$"
)

CN_PREFACE_PATTERN = re.compile(
    r"^\s*(前言(?:[:：].*)?)\s*$"
)

CN_TOC_PATTERN = re.compile(
    r"^\s*(目\s*录)\s*$"
)

CHAPTER_SPACED_PATTERN = re.compile(
    r"^\s*C\s*H\s*A\s*P\s*T\s*E\s*R\s+(\d+)\s*$",
    re.IGNORECASE,
)

PART_SPACED_PATTERN = re.compile(
    r"^\s*P\s*A\s*R\s*T\s+(.+?)\s*$",
    re.IGNORECASE,
)

PREFACE_SPACED_PATTERN = re.compile(
    r"^\s*P\s*R\s*E\s*F\s*A\s*C\s*E\s*$",
    re.IGNORECASE,
)


def detect_heading(line: str) -> Heading | None:
    """
    尝试识别一行文本是否为标题。

    当前支持：
    1. Markdown 标题
    2. 带点号的章节编号（1.1 / 2.10）
    3. 中文部分 / 章 / 前言 / 目录
    4. PDF 抽出的分散字母标题（CHAPTER / PART / PREFACE）
    """
    line = line.strip()
    if not line:
        return None

    markdown_match = MARKDOWN_HEADING_PATTERN.match(line)
    if markdown_match:
        hashes, title = markdown_match.groups()
        return Heading(title=title.strip(), level=len(hashes))

    subsection_match = SUBSECTION_PATTERN.match(line)
    if subsection_match:
        number, title = subsection_match.groups()
        level = number.count(".") + 2
        return Heading(title=f"{number} {title.strip()}", level=level)

    cn_part_match = CN_PART_PATTERN.match(line)
    if cn_part_match:
        return Heading(title=re.sub(r"\s+", " ", cn_part_match.group(1)).strip(), level=1)

    cn_chapter_match = CN_CHAPTER_PATTERN.match(line)
    if cn_chapter_match:
        return Heading(title=re.sub(r"\s+", " ", cn_chapter_match.group(1)).strip(), level=2)

    cn_preface_match = CN_PREFACE_PATTERN.match(line)
    if cn_preface_match:
        return Heading(title=cn_preface_match.group(1).strip(), level=1)

    cn_toc_match = CN_TOC_PATTERN.match(line)
    if cn_toc_match:
        return Heading(title="目录", level=1)

    chapter_spaced = CHAPTER_SPACED_PATTERN.match(line)
    if chapter_spaced:
        return Heading(title=f"Chapter {chapter_spaced.group(1)}", level=2)

    preface_spaced = PREFACE_SPACED_PATTERN.match(line)
    if preface_spaced:
        return Heading(title="Preface", level=1)

    part_spaced = PART_SPACED_PATTERN.match(line)
    if part_spaced:
        roman = re.sub(r"\s+", "", part_spaced.group(1))
        return Heading(title=f"Part {roman}", level=1)

    return None


class SectionTracker:
    def __init__(self):
        self._sections: list[Heading] = []

    def update(self, heading: Heading) -> None:
        target_level = heading.level
        self._sections = [
            section
            for section in self._sections
            if section.level < target_level
        ]
        self._sections.append(heading)

    def get_path(self) -> list[str]:
        return [section.title for section in self._sections]

    def get_current_title(self) -> str:
        if not self._sections:
            return "Unknown"
        return self._sections[-1].title
