"""
中文分词工具：BM25 索引与查询共用，保证两侧切词口径一致。

用 jieba 做通用分词；结果统一转小写、去首尾空白，过滤空 token。
（注意：BM25 是词袋模型，索引和查询必须用同一套分词，否则召回会严重失真。）
"""

from typing import List

import jieba


def tokenize(text: str) -> List[str]:
    """把文本切成小写 token 列表，供 BM25 使用。"""
    return [
        token.strip().lower()
        for token in jieba.lcut(text)
        if token.strip()
    ]


if __name__ == '__main__':
    # 手动调试：观察分词效果
    text = "Kafka消费者offset 提交失败"
    print(tokenize(text))