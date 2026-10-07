def _validate_split_params(chunk_size: int, overlap: int) -> None:
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
    return [
        chunk
        for chunk, _, _ in split_text_windows(text, chunk_size, overlap)
    ]
