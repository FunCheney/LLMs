"""
Milvus 向量库实现（VectorOpt 接口的数据库后端版本）。

通过 MilvusClient 连接 Milvus 服务，collection schema：
- id        : INT64 自增主键（Milvus 内部存储主键，不暴露给上层）
- chunk_id  : VARCHAR，业务主键（对应 VectorRecord.id / DocumentChunk.chunk_id）
- embedding : FLOAT_VECTOR，向量字段（AUTOINDEX + COSINE）
- metadata  : VARCHAR，dict 的 JSON 序列化结果（读出时需 json.loads 还原）
- content   : VARCHAR，chunk 正文

与本地方案（LocalVectorIndex）的语义差异：
- add() 只把记录放进内存缓冲，save() 才真正 insert 进 Milvus；
- save(path) 的 path 参数仅为满足接口签名，DB 自身负责持久化；
- auto_id 主键无法按 chunk_id 去重/upsert，重复执行构建脚本会导致数据翻倍；
  如需重建索引，请先 drop collection 再重新构建（见 build_vector_db_index 的调用方）。
"""
import json
from pathlib import Path
from typing import List

from pymilvus import MilvusClient, DataType

from vector_store import VectorRecord
from vector_store.base import VectorOpt


class MilvusVectorOpt(VectorOpt):
    def __init__(self,
                 milvus_uri: str,
                 token: str,
                 collection_name: str,
                 embedding_dim: int
                 ) -> None:
        self.collection_name = collection_name
        self.client: MilvusClient = MilvusClient(
            uri=milvus_uri,
            token=token,
        )
        self.records: List[VectorRecord] = []

        if not self.client.has_collection(self.collection_name):
            # 1. 创建 schema
            schema = self.client.create_schema(auto_id=True, enable_automatic_fields=False)

            # 2. 添加字段
            # VARCHAR 上限取 Milvus 最大值 65535，避免 chunk_size 调大后 content 溢出；
            # metadata 是 JSON 字符串，4096 足够容纳章节路径等元信息
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=512)
            schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=embedding_dim)
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

        # collection 已存在时也必须显式加载：Milvus 重启后 collection 不会自动加载，
        # 不加载直接 search 会报 collection not loaded
        self.client.load_collection(collection_name=self.collection_name)

    def add(self, vector: VectorRecord):
        self.records.append(vector)

    def count(self) -> int:
        return len(self.records)

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
                # 回填业务主键 chunk_id，而不是 Milvus 自增的 INT64 主键 hit["id"]，
                # 否则下游 mappers.vector_record_to_hit 拿不到正确的 chunk_id
                id=hit["entity"]["chunk_id"],
                text=hit["entity"]["content"],
                # 写入时做了 json.dumps，读出必须 json.loads 还原成 dict，
                # 否则下游对 metadata 调 .get() 会 AttributeError
                metadata=json.loads(hit["entity"]["metadata"]),
                embedding=[],  # 检索结果不需要向量本体，与 search_local_index 的行为一致
            )
            score = hit["distance"]
            results.append((record, score))

        return results

    def save(self, path: str | Path = "") -> None:
        # path 参数仅为满足 VectorOpt 接口签名：DB 后端自身持久化，无需落盘文件。
        # 空数据插入会被 Milvus 拒绝（empty data），没有缓冲记录时直接跳过
        if not self.records:
            return

        data_to_insert = [
            {
                "chunk_id": record.id,
                "embedding": record.embedding,
                "metadata": json.dumps(record.metadata, ensure_ascii=False),
                "content": record.text,
            }
            for record in self.records
        ]
        result = self.client.insert(collection_name=self.collection_name, data=data_to_insert)
        print(f"插入成功数据 {result['insert_count']} 条")
