"""
BGE CrossEncoder 重排序器（BAAI/bge-reranker-v2-m3）。

与 BGE-M3 检索模型配套：bge-m3 负责召回（双塔，快），
bge-reranker-v2-m3 负责精排（交叉编码，query 与候选一起进模型，更准）。

注意：sentence_transformers / torch 只在实例化时才导入（惰性加载），
避免 import 本模块就拖慢 CLI 启动（几十秒级开销）。
"""

from retrieval.types import RetrievalResult

from .base import Reranker

# 默认模型：多语言，和 BGE-M3 同源，适合中文语料
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


class BGEReranker(Reranker):
    """
    基于 CrossEncoder 的重排序器。

    用法：
        reranker = BGEReranker()                    # 首次会下载模型（约 2GB）
        hits = reranker.rerank(query, hits, top_k=5)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        # 与 embedding/bge_m3.py 保持一致：导入前把 HF 请求重定向到国内镜像
        import os

        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        # 惰性导入：只有真正实例化时才加载 torch / sentence-transformers
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name, device=device)
        self._batch_size = batch_size

    def rerank(
        self,
        query: str,
        hits: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """对候选逐条与 query 联合打分，按分数降序重排。"""
        if not hits:
            return []

        pairs = [(query, hit.text) for hit in hits]
        scores = self._model.predict(pairs, batch_size=self._batch_size)

        # 按重排分数降序；分数相同保持原召回顺序（稳定排序）
        ranked = sorted(
            zip(hits, scores, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )

        reranked: list[RetrievalResult] = []
        for rank, (hit, score) in enumerate(ranked, start=1):
            # score 替换为重排分数（保留原召回分数到 metadata 便于调试），rank 重编
            hit.metadata["prerank_score"] = hit.score
            hit.score = float(score)
            hit.rank = rank
            reranked.append(hit)

        return reranked[:top_k] if top_k is not None else reranked
