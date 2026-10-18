"""
向量库工厂：全项目唯一同时知道 local / milvus 具体类的地方（Composition Root）。

- open_reader(config)：检索侧入口，返回 VectorReader；
- create_writer(config)：构建侧入口，返回 VectorWriter（milvus 且 drop_existing 时先重建）。

pymilvus 在函数体内惰性导入：local 模式完全不加载 pymilvus，
即使未安装 pymilvus，local 链路也能正常工作。
"""

from vector_store.base import VectorReader, VectorWriter
from vector_store.config import VectorStoreConfig


def open_reader(config: VectorStoreConfig) -> VectorReader:
    """打开已有索引用于检索。"""
    store_type = config.store_type.lower()
    if store_type == "local":
        from vector_store.local import LocalVectorIndex

        if not config.index_path.exists():
            raise FileNotFoundError(
                f"Vector index not found: {config.index_path}. "
                "Run scripts/build_vector_index.py first."
            )
        return LocalVectorIndex.load(config.index_path)
    if store_type == "milvus":
        # 惰性导入：避免 local 模式连带加载 pymilvus
        from vector_store.milvus import MilvusVectorOpt

        return MilvusVectorOpt(
            milvus_uri=config.uri,
            token=config.token,
            collection_name=config.collection,
            embedding_dim=config.dimension,
        )
    raise ValueError(f"Unsupported vector store type: {config.store_type}")


def create_writer(config: VectorStoreConfig) -> VectorWriter:
    """创建写侧实例用于索引构建；milvus + drop_existing 时先 drop 重建空 collection。"""
    store_type = config.store_type.lower()
    if store_type == "local":
        from vector_store.local import LocalVectorIndex

        return LocalVectorIndex()
    if store_type == "milvus":
        from vector_store.milvus import MilvusVectorOpt

        store = MilvusVectorOpt(
            milvus_uri=config.uri,
            token=config.token,
            collection_name=config.collection,
            embedding_dim=config.dimension,
        )
        if config.drop_existing:
            store.drop()
        return store
    raise ValueError(f"Unsupported vector store type: {config.store_type}")
