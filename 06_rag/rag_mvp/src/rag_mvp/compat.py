"""兼容包化前生成的 pickle 索引。

背景：W5 之前 rag_mvp 的源码直接放在 `src/` 下，模块路径是 `embedding.bm25`、
`vector_store.vector_store_types` 这类**顶层名**（靠 editable 安装的 .pth 把
整个 `src/` 塞进 sys.path 才能 import）。包化后要加 `rag_mvp.` 前缀。

问题在于 pickle **不存类本身，存的是 `__module__ + __qualname__`**。于是所有
已经落盘的索引（`data/index/{bm25,local}.pkl`）在包化后一律
`ModuleNotFoundError: No module named 'embedding'`——文件没坏，是里面的路径过期了。

这里用自定义 `Unpickler` 在**加载时**做一次前缀映射，而不是往 `sys.modules`
注册别名：后者会把"顶层包污染"永久固定在进程里，正是包化要消除的东西，
而且会让 `import retrieval` 重新变得可用，掩盖真正的问题。

长期方向：索引不该用 pickle 存对象。它把实现细节（模块路径、类布局）焊进了
数据文件，任何重构都会让历史数据失效。下次换格式时优先选 JSON/parquet
这类只描述数据的格式。
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

# 包化前的顶层模块名。它们全部迁到了 rag_mvp.<同名>。
LEGACY_TOPLEVEL: frozenset[str] = frozenset(
    {
        "embedding",
        "eval",
        "generation",
        "indexing",
        "ingestion",
        "pre_query",
        "reranker",
        "retrieval",
        "schemas",
        "text_2_sql",
        "validation",
        "vector_store",
    }
)


class LegacyUnpickler(pickle.Unpickler):
    """把 `embedding.bm25.BM25Index` 解析成 `rag_mvp.embedding.bm25.BM25Index`。"""

    def find_class(self, module: str, name: str) -> Any:
        top = module.split(".")[0]
        if top in LEGACY_TOPLEVEL:
            module = f"rag_mvp.{module}"
        return super().find_class(module, name)


def load_pickle(path: str | Path) -> Any:
    """加载可能来自包化前的 pickle 文件。

    新写入的文件带 `rag_mvp.` 前缀，映射不会触发，走原路径。
    """
    with Path(path).open("rb") as file:
        return LegacyUnpickler(file).load()
