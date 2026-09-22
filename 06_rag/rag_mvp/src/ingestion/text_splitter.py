def split_text(
    text: str,
    chunk_size: int, # 处理字符的长度，不是模型的 token 数量
    overlap: int,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0"
        )

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must satisfy "
            "0 <= overlap < chunk_size"
        )

    text = text.strip()

    if not text:
        return []

    # 实际移动的长度
    step = chunk_size - overlap

    chunks = []
    # 从 0 开始，以 step 为间隔移动
    for start in range(0, len(text), step):
        # 获取本次结束的位置
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

    return chunks