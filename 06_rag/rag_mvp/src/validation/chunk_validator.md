# Chunk 质量校验

入库完成后检查：

- `chunk_id` 存在且不重复
- 正文非空
- `page_start` / `page_end` 合法
- 正文长度不超过 `chunk_size`（超出记 warning）
- 页码跨度过大（默认 > 8 页）
- 未知章节占比过高
- `source_uri` 是否缺失

Phase 1 验收：抽检至少 20 个 Chunk，确认章节、页码和来源可读。

```bash
PYTHONPATH=src python scripts/build_chunks.py
PYTHONPATH=src python scripts/review_chunks.py
```
