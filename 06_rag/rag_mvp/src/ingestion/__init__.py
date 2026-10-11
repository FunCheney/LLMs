"""
ingestion 包：文档解析与分块（Phase 1）。

主要入口：
- pipeline.ingest_document：解析 → 结构化分块 → 质量校验；
- storage.save_chunks_jsonl / load_chunks_jsonl：chunks.jsonl 读写。

子模块：pdf_parser / html_parser（解析），heading_detector / structured_parser（结构识别），
chunker / page_aware_splitter / paragraph_splitter / text_splitter（分块策略）。
"""
