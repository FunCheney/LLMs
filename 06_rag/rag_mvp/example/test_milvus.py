from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus",
)

print("Milvus connected:", client.list_collections())

client.close()