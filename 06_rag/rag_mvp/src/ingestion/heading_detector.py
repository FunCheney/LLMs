"""
标题识别：判断单行文本是不是「标题」，并维护章节层级路径。

为什么需要它：
- 分块时要尽量沿着文档原有的章节结构切，而不是按字符数硬切；
- 检索结果需要给出「这段内容属于哪一节」，便于展示和引用。

支持识别的标题形式（见 detect_heading）：
1. Markdown 标题（# / ## / ...）
2. 带点号的编号标题（1.1 / 2.10 / 3.2.1）
3. 中文「第 X 部分 / 第 X 章 / 前言 / 目录」
4. PDF 抽取时字符被打散的英文标题（C H A P T E R 1 / P A R T II / P R E F A C E）

title/level 会被 SectionTracker 维护成一棵「章节路径」，
例如 ["第 1 部分 基础", "1.2 安装"]，写进每个 chunk 的 section_path 字段。
"""

from dataclasses import dataclass
import re


@dataclass
class Heading:
    """一个标题：title 为标题文字，level 为层级（数字越小层级越高）。"""
    title: str
    level: int


# Markdown 标题：开头 1~6 个 # 后跟标题文本
MARKDOWN_HEADING_PATTERN = re.compile(
    r"^\s*(#{1,6})\s+(.+?)\s*$"
)

# 至少带一层点号，避免把 "1. 用 GlobTool ..." 这种正文列表当成标题。
SUBSECTION_PATTERN = re.compile(
    r"^\s*(\d+\.\d+(?:\.\d+)*)\s+(\S.*?)\s*$"
)

# 中文「第 X 部分」（level=1，最高层）
CN_PART_PATTERN = re.compile(
    r"^\s*(第\s*[一二三四五六七八九十百零〇0-9]+\s*部分(?:\s+\S.*)?)\s*$"
)

# 中文「第 X 章」（level=2）
CN_CHAPTER_PATTERN = re.compile(
    r"^\s*(第\s*[一二三四五六七八九十百零〇0-9]+\s*章(?:\s+\S.*)?)\s*$"
)

# 中文「前言」
CN_PREFACE_PATTERN = re.compile(
    r"^\s*(前言(?:[:：].*)?)\s*$"
)

# 中文「目录」
CN_TOC_PATTERN = re.compile(
    r"^\s*(目\s*录)\s*$"
)

# PDF 中字母被空格打散的 "C H A P T E R 1" 形式
CHAPTER_SPACED_PATTERN = re.compile(
    r"^\s*C\s*H\s*A\s*P\s*T\s*E\s*R\s+(\d+)\s*$",
    re.IGNORECASE,
)

# PDF 中字母被空格打散的 "P A R T II" 形式
PART_SPACED_PATTERN = re.compile(
    r"^\s*P\s*A\s*R\s*T\s+(.+?)\s*$",
    re.IGNORECASE,
)

# PDF 中字母被空格打散的 "P R E F A C E" 形式
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
        # # 的数量直接作为层级
        hashes, title = markdown_match.groups()
        return Heading(title=title.strip(), level=len(hashes))

    subsection_match = SUBSECTION_PATTERN.match(line)
    if subsection_match:
        # 1.1 → level 3，1.1.1 → level 4，以此类推（点号数量 + 2）
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
        # 去掉原文中字母间的空格，还原成 "Part II"
        roman = re.sub(r"\s+", "", part_spaced.group(1))
        return Heading(title=f"Part {roman}", level=1)

    return None


class SectionTracker:
    """
    维护「当前所处的章节路径」。

    每遇到一个标题就调用 update()：
    - 去掉所有层级 >= 当前标题的旧节点（它们是同级或更深的旧章节）；
    - 再把新标题压入栈。
    这样 get_path() 始终返回一条从顶层到当前的合法章节链。
    """

    def __init__(self):
        self._sections: list[Heading] = []  # 章节栈，按层级递增

    def update(self, heading: Heading) -> None:
        target_level = heading.level
        # 弹出所有层级不低于新标题的节点，保证路径层级单调递增
        self._sections = [
            section
            for section in self._sections
            if section.level < target_level
        ]
        self._sections.append(heading)

    def get_path(self) -> list[str]:
        """返回当前章节路径（标题文字列表）。"""
        return [section.title for section in self._sections]

    def get_current_title(self) -> str:
        """返回当前最深层级的章节标题；无标题时返回 "Unknown"。"""
        if not self._sections:
            return "Unknown"
        return self._sections[-1].title
