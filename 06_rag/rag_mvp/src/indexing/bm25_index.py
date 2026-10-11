import pickle
from pathlib import Path
from typing import List

from rank_bm25 import BM25Okapi

from indexing.chunk_metadata import chunk_to_metadata
from indexing.tokenizer import tokenize
from ingestion.storage import load_chunks_jsonl
from vector_store.vector_store_types import VectorRecord


class BM25Index:
    def __init__(self) -> None:
        self.records: List[VectorRecord] = []
        self.bm25: BM25Okapi | None = None

    def build(self, records: List[VectorRecord]) -> None:
        self.records = records
        tokenized_documents = [tokenize(record.text) for record in records]
        self.bm25 = BM25Okapi(tokenized_documents)

    def count(self) -> int:
        return len(self.records)

    def search(self, query: str, top_k: int) -> List[tuple[VectorRecord, float]]:
        if self.bm25 is None:
            raise RuntimeError("BM25 index is not built yet")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        rank_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        return [
            (self.records[index], float(scores[index]))
            for index in rank_indices[:top_k]
        ]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        path = Path(path)
        with path.open("rb") as file:
            index = pickle.load(file)
        if not isinstance(index, cls):
            raise TypeError(f"Expected BM25Index at {path}")
        return index


def build_bm25_index(
    chunks_path: str | Path,
    index_path: str | Path,
) -> BM25Index:
    chunks = load_chunks_jsonl(chunks_path)
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")

    records = [
        VectorRecord(
            id=chunk["chunk_id"],
            text=chunk["content"],
            metadata=chunk_to_metadata(chunk),
            embedding=[],
        )
        for chunk in chunks
    ]

    index = BM25Index()
    index.build(records)
    index.save(index_path)
    return index
