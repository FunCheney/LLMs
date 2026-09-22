import re
from typing import List

from ingestion.text_splitter import split_text

'''
接按照字符位置切分存在几个问题：

1. 可能把一个段落从中间切开。
2. 可能把一句话或一个代码示例拆成两半。
3. 不能很好地保留文档原有的结构。

即使设置了 overlap，也不一定能保证上下文完整。
'''

def split_paragraphs(text: str) -> List[str]:

    """
    将文本按照多行拆分为段落
    """
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    paragraphs = re.split(
        r"\n\s*\n", # 两个换行符之间允许存在空格或其他空白字符。
        text,
    )
    result = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if paragraph:
            result.append(paragraph)

    return result


def split_text_by_paragraphs(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    段落优先的切分策略
    1. 先按段落拆分
    2. 尽可能合并多个短段落
    3. 超长段落使用字符级切分
    4. 普通段落优先保持完整
    """
    if chunk_size < 0:
        raise ValueError("chunk_size must be greater than 0")


    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")

    text = text.strip()
    if not text:
        return []
    paragraphs = split_paragraphs(text)

    chunks = []
    current_chunk = ""
    for paragraph in paragraphs:
        # 情况一：当前段落本省超过 chunk_size
        if len(paragraph) > chunk_size:
            # 先保存之前累计的段落
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            # 超长段落交给字符级切分器
            long_chunk = split_text(paragraph, chunk_size, overlap)
            chunks.extend(long_chunk)
            continue

        # 如果非超长段落
        if not current_chunk:
            current_chunk = paragraph
            continue

        # 尝试将当前段落合并到已有段落
        candidate = (current_chunk +"\n\n"+ paragraph)

        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            # 合并后超出限制，先保存已有 Chunk
            chunks.append(current_chunk)
            # 新段落重新开始一个 Chunk
            current_chunk = paragraph
    # 循环结束后，保存最后一个 chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


