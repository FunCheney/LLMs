"""
示例：连接本地 Milvus 并创建一个测试数据库（Milvus 接入的前置验证）。

前提：本地已启动 Milvus（默认 http://localhost:19530）。
运行：
    uv run python example/milvus_opt.py
"""

from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus",
)

print("Milvus connected:", client.list_collections())

# 创建数据库
# client.create_database(db_name="test_database_1")
# 查找you哪些数据库
databases = client.list_databases()

print("databases:", databases)

client.close()