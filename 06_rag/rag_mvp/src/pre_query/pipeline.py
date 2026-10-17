from pre_query.normalizer import normalize_query
from pre_query.shcema import ProcessedQuery, RetrievalMode


class QueryPreProcessor:

    def __init__(self, * ,max_length: int):
        self.max_length = max_length


    def process(self, query: str, * , retrieval_mode: RetrievalMode = "hybrid") -> ProcessedQuery:

        if retrieval_mode is None:
            raise ValueError(f'retrieval_mode is required for query {query}')

        normalized = normalize_query(query, max_length=self.max_length)

        return ProcessedQuery(original_query=query,rewritten_query=normalized,retrieval_mode=retrieval_mode)
