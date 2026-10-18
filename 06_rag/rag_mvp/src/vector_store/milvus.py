"""
Milvus 向量库实现（VectorStore 接口的数据库后端版本）。

通过 MilvusClient 连接 Milvus 服务（也支持 Milvus Lite：uri 传本地 .db 文件路径即可，
无需启动服务，适合开发与冒烟测试）。collection schema：
- chunk_id  : VARCHAR 主键（auto_id=False），对应 VectorRecord.id / DocumentChunk.chunk_id。
  以业务主键取代旧版的 INT64 自增主键，使 upsert 幂等：重复构建不会数据翻倍。
- embedding : FLOAT_VECTOR，向量字段（AUTOINDEX + COSINE）
- metadata  : VARCHAR，dict 的 JSON 序列化结果（读出时需 json.loads 还原）
- content   : VARCHAR，chunk 正文

语义约定（与 LocalVectorIndex 对齐，见 vector_store/base.py）：
- add() 只把记录放进内存缓冲，save() 才 upsert 进 Milvus 并 flush；
- save(path) 的 path 参数仅为满足接口签名，DB 自身负责持久化；
- count()/search() 面向服务端已持久化数据（count 走 get_collection_stats）；
- drop() 删除并重建空 collection，用于索引重建。
"""
import json
from pathlib import Path
from typing import List

from pymilvus import MilvusClient, DataType

from vector_store.vector_store_types import VectorRecord
from vector_store.base import VectorStore


class MilvusVectorOpt(VectorStore):
    def __init__(self,
                 milvus_uri: str,
                 token: str,
                 collection_name: str,
                 embedding_dim: int
                 ) -> None:
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.client: MilvusClient = MilvusClient(
            uri=milvus_uri,
            token=token,
        )
        self.records: List[VectorRecord] = []

        if not self.client.has_collection(self.collection_name):
            self._create_collection()

        # collection 已存在时也必须显式加载：Milvus 重启后 collection 不会自动加载，
        # 不加载直接 search 会报 collection not loaded
        self.client.load_collection(collection_name=self.collection_name)

    def _create_collection(self) -> None:
        """创建 collection：chunk_id 为业务主键（非自增），向量字段 AUTOINDEX + COSINE。"""
        # 1. 创建 schema（auto_id=False：主键由业务侧 chunk_id 提供，支持 upsert 幂等）
        schema = self.client.create_schema(auto_id=False, enable_automatic_fields=False)

        # 2. 添加字段
        # VARCHAR 上限取 Milvus 最大值 65535，避免 chunk_size 调大后 content 溢出；
        # metadata 是 JSON 字符串，4096 足够容纳章节路径等元信息
        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, is_primary=True, max_length=512)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        schema.add_field(field_name="metadata", datatype=DataType.VARCHAR, max_length=4096)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)

        # 3. 索引参数（只给向量字段建索引）
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")

        # 4. 创建 collection（自动建索引 + 加载）
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def add(self, vector: VectorRecord):
        """追加一条记录到写缓冲（save 时才真正写入 Milvus）。"""
        self.records.append(vector)

    def count(self) -> int:
        """统计服务端已持久化的记录数（而不是内存缓冲）。"""
        stats = self.client.get_collection_stats(collection_name=self.collection_name)
        return int(stats["row_count"])

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[tuple[VectorRecord, float]]:
        search_results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],  # 二维列表
            anns_field="embedding",  # 明确指定向量字段（可选但推荐）
            # 必须带 chunk_id 才能回填业务主键；不回取 embedding（1024 维向量白白占带宽）
            output_fields=["chunk_id", "metadata", "content"],
            limit=top_k,  # 使用 top_k
            search_params={
                "metric_type": "COSINE",
                "params": {"ef": 128}
            }
        )[0]  # 取第一个查询向量的结果列表

        results: List[tuple[VectorRecord, float]] = []
        for hit in search_results:
            # 新版 PyMilvus 中 hit 是 Hit 对象，可用 hit["id"] / hit["distance"] / hit["entity"]
            record = VectorRecord(
                # 主键即 chunk_id，hit["id"] 与 hit["entity"]["chunk_id"] 等价，
                # 统一从 entity 取，语义更明确
                id=hit["entity"]["chunk_id"],
                text=hit["entity"]["content"],
                # 写入时做了 json.dumps，读出必须 json.loads 还原成 dict，
                # 否则下游对 metadata 调 .get() 会 AttributeError
                metadata=json.loads(hit["entity"]["metadata"]),
                embedding=[],  # 检索结果不需要向量本体，与 local 后端行为一致
            )
            score = hit["distance"]
            results.append((record, score))

        return results

    def _to_row(self, record: VectorRecord) -> dict:
        return {
            "chunk_id": record.id,
            "embedding": record.embedding,
            "metadata": json.dumps(record.metadata, ensure_ascii=False),
            "content": record.text,
        }

    def save(self, path: str | Path = "") -> None:
        # path 参数仅为满足 VectorWriter 接口签名：DB 后端自身持久化，无需落盘文件。
        # 空数据写入会被 Milvus 拒绝（empty data），没有缓冲记录时直接跳过
        if not self.records:
            return

        data_to_upsert = [self._to_row(record) for record in self.records]
        # upsert 而非 insert：chunk_id 为主键，重复执行构建脚本不会数据翻倍
        result = self.client.upsert(collection_name=self.collection_name, data=data_to_upsert)
        # flush 保证数据立即可检索/可统计（否则 count/search 可能读不到刚写入的数据）
        self.client.flush(collection_name=self.collection_name)
        print(f"写入成功数据 {result['upsert_count']} 条（upsert，幂等）")
        # 清空缓冲：已持久化的记录不应重复占用内存
        self.records.clear()

    def upsert(self, vector: VectorRecord) -> None:
        """单条幂等写入并立即持久化。"""
        self.client.upsert(collection_name=self.collection_name, data=[self._to_row(vector)])
        self.client.flush(collection_name=self.collection_name)

    def drop(self) -> None:
        """删除并重建空 collection（用于索引重建）；同时清空内存缓冲。"""
        self.records.clear()
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)
        self._create_collection()
        self.client.load_collection(collection_name=self.collection_name)
