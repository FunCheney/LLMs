"""
向量库配置：local / milvus 双后端的统一配置载体。

连接参数只允许出现在这里（Composition Root 的一部分）：
- CLI 参数优先，未传时回落到环境变量（MILVUS_URI / MILVUS_TOKEN / MILVUS_COLLECTION）；
- 业务代码（检索器、构建脚本）不再感知 uri/token/collection 这些细节。

Milvus Lite 用法：uri 传本地文件路径（如 data/index/milvus.db）即可，
无需启动 Milvus 服务，适合开发与冒烟测试。
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class VectorStoreConfig:
    """向量库后端配置（不可变；store_type 决定使用哪种后端）。"""

    store_type: str = "local"  # local | milvus
    # local 后端：pickle 索引文件路径
    index_path: Path = Path("data/index/local.pkl")
    # milvus 后端：连接参数（env 兜底，便于 CI/部署注入）
    uri: str = field(default_factory=lambda: os.getenv("MILVUS_URI", "http://localhost:19530"))
    token: str = field(default_factory=lambda: os.getenv("MILVUS_TOKEN", "root:Milvus"))
    collection: str = field(default_factory=lambda: os.getenv("MILVUS_COLLECTION", "rag_mvp_chunks"))
    dimension: int = 1024  # BGE-M3 向量维度
    # 构建索引时是否先 drop 重建（milvus 后端；local 后端 save 本身就是全量覆盖）
    drop_existing: bool = False
