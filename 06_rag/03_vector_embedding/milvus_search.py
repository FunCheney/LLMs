from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus",
)

collections = client.list_collections()

print(collections)

def single_vector_search():
    query_vector = [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354,
                    0.9029438446296592]

    search = client.search(collection_name="customized_setup_1", anns_field="my_vector", data=[query_vector], limit=3,
                           search_params={"metric_type": "COSINE"})

    # data: [[
    # {'my_id': 0, 'distance': 1.0, 'entity': {}},
    # {'my_id': 1, 'distance': 0.6290165185928345, 'entity': {}},
    # {'my_id': 4, 'distance': 0.5975797176361084, 'entity': {}}
    # ]]
    print(search)

    for hit in search:
        print(hit)
        for item in hit:
            print(item)

    print(search[0])



def bulk_vector_search():
    query_vectors = [
        [0.041732933, 0.013779674, -0.027564144, -0.013061441, 0.009748648],
        [0.0039737443, 0.003020432, -0.0006188639, 0.03913546, -0.00089768134]
    ]

    search = client.search(collection_name="customized_setup_1",
                           data=query_vectors,
                           limit=3)

    print(search[0])


def primary_key_search():
    res = client.search(
        collection_name="customized_setup_1",
        anns_field="my_vector",
        ids=[551, 296, 43],
        limit=3,
        search_params={"metric_type": "COSINE"}
    )

    for hits in res:
        for hit in hits:
            print(hit)

def search_use_output_filed():
    query_vector = [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354,
                    0.9029438446296592]

    res = client.search(
        collection_name="customized_setup_1",
        data=[query_vector],
        limit=3,  # The number of results to return
        search_params={"metric_type": "COSINE"},
        output_fields = ["my_varchar"]
    )

    print(res)



if __name__ == "__main__":
    # single_vector_search()
    # bulk_vector_search()
    # primary_key_search()
    search_use_output_filed()
