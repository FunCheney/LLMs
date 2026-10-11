from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus",
)

print("Milvus connected:", client.list_collections())

# 创建数据库
client.create_database(db_name="test_database_1")



client.close()