from generation.shcame import Citation, RAGResponse


def format_citation(citation: Citation) -> str:
    path = " > ".join(citation.section_path) if citation.section_path else "-"
    return (
        f"[{citation.index}] {citation.section_title} "
        f"(p.{citation.page_start}-{citation.page_end}) "
        f"chunk_id={citation.chunk_id}\n"
        f"    path: {path}\n"
        f"    source: {citation.source_uri}\n"
        f"    quote: {citation.quote}"
    )


def print_rag_response(response: RAGResponse) -> None:
    print(f"Query: {response.query}")
    print(f"Mode: {response.retrieval_mode}")
    print(f"Insufficient evidence: {response.insufficient_evidence}")
    print(f"Retrieved chunks: {len(response.retrieved_chunk_ids)}")
    print()
    print("Answer:")
    print(response.answer)
    print()
    if response.citations:
        print("Citations:")
        for citation in response.citations:
            print(format_citation(citation))
            print()
