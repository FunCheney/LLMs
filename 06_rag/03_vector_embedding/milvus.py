from pymilvus import MilvusClient, DataType

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus",
)

print("Milvus connected:", client.list_collections())

client.drop_collection('vector_db_collection')
client.drop_collection('vector_collection')
client.drop_collection('customized_setup_1')
client.drop_collection('customized_setup_2')



print("Milvus collection created:", client.list_collections())