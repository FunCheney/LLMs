"""换向量库后端后的等价性校验：同一查询在 local 与 pgvector 上必须给出同样的 Top-K。

为什么需要这个脚本：向量库只是"存放"，理论上换后端不该改变检索结果。但对不上
的地方全在细节里——余弦距离的方向、归一化与否、分数是否同量纲、Top-K 的截断
发生在 SQL 还是 Python。这些差异不会报错，只会让 A/B 报告里的数字悄悄变好或变坏，
而你以为是"这次改 prompt 起作用了"。

用法::

    uv run --extra pgvector python scripts/compare_vector_backends.py \\
        --query "MCP 是什么" --top-k 5

退出码 0 = 完全一致；1 = 有差异（差异本身不一定是 bug，但必须解释清楚）。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from rag_mvp.embedding import BGEM3EmbeddingProvider
from rag_mvp.retrieval.retriever import DenseRetriever
from rag_mvp.vector_store.config import VectorStoreConfig
from rag_mvp.vector_store.factory import open_reader

DEFAULT_LOCAL = Path("data/index/local.pkl")
DEFAULT_TABLE = "rag_chunks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare retrieval results across vector backends")
    parser.add_argument("--query", default="MCP 是什么", help="查询文本")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--local-index", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--pgvector-dsn", default=None, help="默认 env PGVECTOR_DSN")
    parser.add_argument("--pgvector-table", default=DEFAULT_TABLE)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dsn = args.pgvector_dsn or os.getenv("PGVECTOR_DSN", "")
    if not dsn:
        print("缺少 PGVECTOR_DSN（或 --pgvector-dsn）", file=sys.stderr)
        return 2

    embedder = BGEM3EmbeddingProvider(device=args.device)

    local_cfg = VectorStoreConfig(store_type="local", index_path=args.local_index)
    pg_cfg = VectorStoreConfig(store_type="pgvector", dsn=dsn, table=args.pgvector_table)

    local = DenseRetriever(store=open_reader(local_cfg), embedder=embedder)
    pg = DenseRetriever(store=open_reader(pg_cfg), embedder=embedder)

    local_hits = local.retrieve(args.query, top_k=args.top_k)
    pg_hits = pg.retrieve(args.query, top_k=args.top_k)

    print(f"query = {args.query!r}  top_k = {args.top_k}")
    print(f"{'rank':<5}{'local chunk':<38}{'score':<10}{'pgvector chunk':<38}{'score':<10}")
    for i in range(max(len(local_hits), len(pg_hits))):
        left = local_hits[i] if i < len(local_hits) else None
        right = pg_hits[i] if i < len(pg_hits) else None
        print(
            f"{i + 1:<5}"
            f"{(left.chunk_id if left else '-'):<38}"
            f"{(f'{left.score:.6f}' if left else '-'):<10}"
            f"{(right.chunk_id if right else '-'):<38}"
            f"{(f'{right.score:.6f}' if right else '-'):<10}"
        )

    local_ids = [hit.chunk_id for hit in local_hits]
    pg_ids = [hit.chunk_id for hit in pg_hits]
    if local_ids == pg_ids:
        print("\n结果一致：换后端不改变排序。")
        return 0

    # 只比集合不比顺序时，说明"候选相同、排序不同"——通常是分数精度或
    # 距离函数实现差异，影响比"漏召回"小一个量级。
    only_local = [cid for cid in local_ids if cid not in pg_ids]
    only_pg = [cid for cid in pg_ids if cid not in local_ids]
    if set(local_ids) == set(pg_ids):
        print("\n候选集合一致，但排序不同（分数精度或距离实现差异）。")
    else:
        print(f"\n候选集合不同：仅 local {only_local}；仅 pgvector {only_pg}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
