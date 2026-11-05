"""手工冒烟脚本（不是 pytest 用例，只有 main()）。

pytest 会因为文件名收集它并在 import 阶段执行顶层语句，所以必须先探测
可选依赖：缺 `local` extra 时跳过，而不是 collection error。
"""

import pytest

pytest.importorskip(
    "sentence_transformers",
    reason="需要 local extra：uv sync --extra local",
)

from rag_mvp.embedding import BGEM3EmbeddingProvider  # noqa: E402


def main():
    embedding = BGEM3EmbeddingProvider()

    result = embedding.embed_text(
        "Python 是一种高级编程语言"
    )

    print("model:", result.model)
    print("dimension:", result.dimension)
    print("vector length:", len(result.vector))
    print("first 10 values:", result.vector[:10])


if __name__ == "__main__":
    main()