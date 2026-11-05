### 内容

||  |
|--|--|
|路径|作用|
|data/eval/smoke_queries.yaml|10 条冒烟题（9 条正例 + 1 条负例 smoke-010）|
|src/eval/smoke_schema.py|解析 YAML|
|src/eval/matchers.py|section_title / section_path 匹配（忽略空白）|
|src/eval/smoke_runner.py|多 mode 跑检索、汇总、JSON 报告|
|scripts/run_smoke_retrieval.py|CLI 入口|
|test/test_smoke_matchers.py|匹配规则单测（不加载 BGE）|

### 判定规则
* 正例（expect_hit: true）：Top-K 里任意一条的 section_title 或 section_path 命中 any_section_title → 通过
* 负例（smoke-010）：Top-K 不应命中规则 → 通过
* 可选：any_content_contains（当前 YAML 未用）

### 使用方式

```shell
cd rag_mvp

# 需先有索引
uv run python scripts/build_vector_index.py
uv run python scripts/build_bm25_index.py

# 默认跑 yaml 里三种 mode
uv run python scripts/run_smoke_retrieval.py

# 只测 hybrid
uv run python scripts/run_smoke_retrieval.py --modes hybrid --k 5

# 失败也 exit 0（调试用）
uv run python scripts/run_smoke_retrieval.py --no-fail-on-miss
```
报告：data/eval/reports/smoke_latest.json；控制台会打印按 tag 统计和 FAIL 详情（top sections / chunk_ids）。

### 本地试跑结果（hybrid）

* 约 9/10 通过（修正 smoke-002 后，14.2 CLAUDE.md 也算命中）
* smoke-009（三种代理模式）仍可能失败：Top-5 多是 17.5/17.8/17.10，未进 17.2 —— 这是检索质量问题，不是脚本 bug；可改问法、加大 --k，或后续调检索

### 维护问题集

编辑 data/eval/smoke_queries.yaml：用 search_cli 看 Top-K 的 section，把期望节名写进 any_section_title（可写多个 OR）。 同一节多 chunk 时 不必写 chunk_id。