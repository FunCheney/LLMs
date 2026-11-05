"""知识层（W5）：文档 → 块 → 可引用的证据。

当前只落契约；后续按 W5 逐日填充：

- `base.py`        数据契约 Chunk / ScoredChunk + EmbeddingProvider / VectorStore 抽象
- `chunking.py`    页感知切分（D1）
- `pgvector_store.py`  向量后端（D2）

纪律：本包**不 import 任何模型推理库**。embedding 实现由 apps 层注入，
否则框架包会被 torch 绑死。
"""

from .base import Chunk, EmbeddingProvider, ScoredChunk, VectorStore

__all__ = ["Chunk", "EmbeddingProvider", "ScoredChunk", "VectorStore"]
