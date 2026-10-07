from typing import List

from rank_bm25 import BM25Okapi

from vector_store.vector_store_types import VectorRecord
from indexing.tokenizer import tokenize

class BM25Index:
    def __init__(self):
        self.records: List[VectorRecord] = []
        self.bm25: BM25Okapi | None = None

    def build(self, records: List[VectorRecord]) -> None:
        self.records = records
        tokenized_document = [
            tokenize(record.text)
            for record in records
        ]

        self.bm25 = BM25Okapi(tokenized_document)

    def search(self, query, top_k) -> List[tuple[VectorRecord, float]]:
        if self.bm25 is None:
            raise RuntimeError('BM25 not built yet')

        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        rank_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        return [
            (self.records[i], float(scores[i]))
            for i in rank_indices
        ]