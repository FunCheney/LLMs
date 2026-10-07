from retrieval.rrf import reciprocal_rank_fusion
from retrieval.types import RetrievalResult


def result(id: str) -> RetrievalResult:
    return RetrievalResult(
        id=id,
        text=f"text-{id}",
        metadata={},
        score=0.0,
    )


def test_rrf():

    dense_results = [
        result("A"),
        result("B"),
        result("C"),
        result("D"),
    ]

    bm25_results = [
        result("D"),
        result("B"),
        result("E"),
        result("A"),
    ]

    results = reciprocal_rank_fusion(
        [
            dense_results,
            bm25_results,
        ]
    )

    for item in results:
        print(item.id, item.score)

    assert results[0].id == "B"