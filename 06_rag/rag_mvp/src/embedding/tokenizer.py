from typing import List

import jieba


def tokenize(text:str) -> List[str]:
    return [
        token.strip().lower()
        for token in jieba.lcut(text)
        if token.strip()
    ]

if __name__ == '__main__':
    text = "Kafka消费者offset 提交失败"
    print(tokenize(text))