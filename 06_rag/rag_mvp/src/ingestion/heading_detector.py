from dataclasses import dataclass
import re

@dataclass
class Heading:
    title: str
    level: int # 章节层级


NUMBERED_HEADING_PATTERN = re.compile(
    r"^\s*((?:\d+\.)*\d+)\s+(.+?)\s*$"
)

MARKDOWN_HEADING_PATTERN = re.compile(
    r"^\s*(#{1,6})\s+(.+?)\s*$"
)


def detect_heading(line: str) -> Heading | None:
    """
    尝试识别一行文本是否为标题。

    当前支持：
    1. Markdown 标题
    2. 数字编号标题
    """
    line = line.strip()

    if not line:
        return None

    # Markdown 标题，例如：
    # ## Architecture
    markdown_match = MARKDOWN_HEADING_PATTERN.match(line)

    if markdown_match:
        hashes, title = markdown_match.groups()

        return Heading(
            title=title.strip(),
            level=len(hashes),
        )

    # 数字编号标题，例如：
    # 2. Architecture
    # 2.1 System Overview
    numbered_match = NUMBERED_HEADING_PATTERN.match(line)

    if numbered_match:
        number, title = numbered_match.groups()
        # 数字标题的 level 是通过编号中的点数量推导的，如 2.1.1 对应的 level 是 3
        level = number.count(".") + 1

        return Heading(
            title=f"{number} {title.strip()}",
            level=level,
        )

    return None

class SectionTracker:
    def __init__(self):
        self._sections: list[Heading] = []

    def update(self, heading: Heading) -> None:
        """
        更新当前章节路径。
        """
        target_level = heading.level

        # 删除当前层级及更深层级的旧章节
        self._sections = [
            section
            for section in self._sections
            if section.level < target_level
        ]

        self._sections.append(heading)

    def get_path(self) -> list[str]:
        """
        返回当前章节路径。
        """
        return [
            section.title
            for section in self._sections
        ]

    def get_current_title(self) -> str:
        """
        返回当前最末级章节标题。
        """
        if not self._sections:
            return "Unknown"

        return self._sections[-1].title