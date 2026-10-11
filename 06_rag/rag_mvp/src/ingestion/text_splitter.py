"""
字符级滑动窗口切分器（最底层的兜底切分）。

当段落本身超过 chunk_size（例如整段代码或超长列表）时，
按固定窗口 + 重叠的方式强制切分，并返回每段在原文中的字符区间，
以便上层把「字符区间」映射回「页码」。

    step = chunk_size - overlap   # 相邻窗口的步进
    窗口1: [0, chunk_size)
    窗口2: [step, step+chunk_size)
    ...

这是「结构优先」策略失败时的最后手段，会牺牲一定语义完整性。
"""


def _validate_split_params(chunk_size: int, overlap: int) -> None:
    """校验切分参数：chunk_size 必须为正，overlap 必须落在 [0, chunk_size)。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")


def split_text_windows(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[tuple[str, int, int]]:
    """返回 (chunk, start_offset, end_offset)，offset 相对原始 text。"""
    _validate_split_params(chunk_size, overlap)

    original = text
    text = text.strip()
    if not text:
        return []

    # strip 会改变字符位置，先记录偏移量，最后把它加回到窗口坐标上
    offset_adjust = original.find(text)
    if offset_adjust < 0:
        offset_adjust = 0

    step = chunk_size - overlap
    windows: list[tuple[str, int, int]] = []

    for start in range(0, len(text), step):
        end = start + chunk_size
        piece = text[start:end]
        chunk = piece.strip()
        if chunk:
            # 窗口首尾可能被 strip 掉空白，需修正 chunk 在原文中的真实起点
            rel = piece.find(chunk)
            abs_start = offset_adjust + start + max(rel, 0)
            abs_end = abs_start + len(chunk)
            windows.append((chunk, abs_start, abs_end))
        if end >= len(text):
            break

    return windows


def split_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """只返回切分后的文本片段（丢弃偏移量），供简单场景使用。"""
    return [
        chunk
        for chunk, _, _ in split_text_windows(text, chunk_size, overlap)
    ]
