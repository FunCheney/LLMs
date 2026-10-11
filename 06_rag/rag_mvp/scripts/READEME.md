### 使用方式
#### build_chunks 使用
将资源转成对应的 jsonl 文件，资源可以是 pdf，markdown 等
```shell
cd rag_mvp
uv run python scripts/build_chunks.py          # 生成 jsonl
```

#### build_vector_index 使用
创建向量索引，用户相似度查询

```shell
cd rag_mvp
uv run python scripts/build_vector_index.py   
```

#### build_bm25_index 使用
创建 bm25_index，用于关键词搜索

```shell
cd rag_mvp
uv run python scripts/build_bm25_index.py
```

#### search_cli 使用

```shell
cd rag_mvp

uv run python scripts/search_cli.py "MCP 是什么" --mode dense --k 5
uv run python scripts/search_cli.py "CLAUDE.md" --mode sparse --k 5
uv run python scripts/search_cli.py "MCP 是什么" --mode hybrid --k 5
```

####  ask_cli 使用
```shell
cd rag_mvp

# 先确认检索（不调 LLM，仍会加载 embedding）
uv run python scripts/ask_cli.py "MCP 是什么" --dry-run
# 完整问答
uv run python scripts/ask_cli.py "MCP 是什么"
uv run python scripts/ask_cli.py "CLAUDE.md 做什么" --k 5 --mode hybrid
```