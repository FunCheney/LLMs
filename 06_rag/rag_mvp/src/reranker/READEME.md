# reranker 重排序模块

## 定位

检索后、生成前的精排层。检索器（dense/sparse/hybrid）用便宜的打分召回较多候选，
Reranker 用交叉编码模型对候选与 query 联合打分，精排出 Top-K：

```
召回（快、粗，Top-N） → rerank（慢、准，Top-K） → 生成/展示
```

## 组成

- `base.py`：`Reranker` 抽象接口，`rerank(query, hits, top_k)`；
- `bge.py`：`BGEReranker`，基于 BAAI/bge-reranker-v2-m3（CrossEncoder，多语言，
  与 BGE-M3 召回模型配套）。首次实例化会下载模型（约 2GB，走 hf-mirror 镜像）；
- `pipeline.py`：`RerankedRetriever`，把「召回放大 + 重排精排」包装成普通 `Retriever`，
  上层（CLI / 问答 / 评测）无需感知重排的存在。

## 接入方式

通过检索工厂开启，不需要直接操作本模块：

```python
RetrieverConfig(mode="hybrid", reranker="bge")   # 默认召回放大 top_k×10（至少 20 条）
```

命令行：

```bash
rag-mvp search "..." --rerank          # 检索精排
rag-mvp ask "..." --rerank             # 问答前先精排
rag-mvp golden --modes dense --rerank  # 评测：dense vs dense+rerank 对比
```

## 约定

- 惰性导入：`import reranker` 不加载 torch；`BGEReranker()` 实例化时才加载；
- 重排后 `hit.score` 为重排分数，原召回分数保存在 `hit.metadata["prerank_score"]`；
- 收益必须用 Golden 指标验证（`rag-mvp golden --rerank`），不许凭感觉上线。
