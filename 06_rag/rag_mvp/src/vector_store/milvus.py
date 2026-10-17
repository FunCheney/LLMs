"""
Milvus 向量库实现的占位骨架（尚未完成）。

现状：目前只实现了与本地方案一致的 add / count 与 save 目录创建，

后续待办：
1. 用 MilvusClient 连接（见 example/test_milvus.py），建 collection / schema；
2. add 时写入向量 + metadata，search 时调用 client.search 并映射回 VectorRecord；
3. save 可改为「导出 collection 快照」或直接去掉（DB 自身持久化）。

在此之前，本项目统一使用 vector_store.local.LocalVectorIndex。
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
            # self.client.drop_collection(self.collection_name)
            # 1. 创建 schema
            schema = self.client.create_schema(auto_id=True, enable_automatic_fields=False)

            # 2. 添加字段
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=512)
            schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=embedding_dim)
            schema.add_field(field_name="metadata", datatype=DataType.VARCHAR, max_length=2048)
            schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=2048)

            # 3. 索引参数（只给向量字段建索引）
            index_params = self.client.prepare_index_params()
            index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")

            # 4. 创建 collection（自动建索引 + 加载）
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params,
            )

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
            output_fields=["embedding","metadata", "content"],  # 需要返回的字段
            limit=top_k,  # 使用 top_k
            search_params={
                "metric_type": "COSINE",
                "params": {"ef": 128}
            }
        )[0]  # 取第一个查询向量的结果列表

        results: List[tuple[VectorRecord, float]] = []
        # print(f"search results: {search_results}")
        for hit in search_results:
            # 新版 PyMilvus 中 hit 是 Hit 对象，可用 hit["id"] / hit["distance"] / hit["entity"]
            record = VectorRecord(
                id=hit["id"],
                text=hit["entity"]["content"],
                metadata=hit["entity"]["metadata"],
                # 如有其他标量字段，也从 hit["entity"] 里取
                embedding=hit["entity"]["embedding"],
            )
            score = hit["distance"]
            results.append((record, score))

        return results

    def save(self, path: str | Path) -> None:
        # 插入数据库
        data_to_insert = []
        if self.records:
            for record in self.records:
                data_to_insert.append(
                    {"chunk_id": record.id,
                     "embedding": record.embedding,
                     "metadata": json.dumps(record.metadata),
                     "content": record.text,
                     }
                )

        result = self.client.insert(collection_name=self.collection_name, data=data_to_insert)
        print(f"插入成功数据 {result["insert_count"]} 条")
