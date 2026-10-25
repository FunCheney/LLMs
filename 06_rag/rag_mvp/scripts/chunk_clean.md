### 数据清洗
RAG 数据清洗的目标不是把文本「洗得漂亮」，而是让检索能稳定命中、生成能读到完整语义。可以按「在哪一层洗、洗什么、怎么验收」来理解。

### 清洗放在流水线哪一步

```text
原始文档 → 解析/结构化 → 分块(chunks) → 【清洗】 → 建索引 → 检索评测
```

* 解析阶段：编码、乱码、页眉页脚、水印、断行、表格/代码块是否保留结构。 
* 分块阶段：chunk 大小、overlap、是否按章节切（metadata 比正文更重要时尤其如此）。 
* 分块之后：去重、修 metadata、删噪声 chunk——你项目里 clean_chunks.py 就在这一层，且尽量不改正文，只动结构和元数据。

原则：能在解析/分块里修的就前移；只有「全库规则、评测驱动」的再写后处理脚本。

### 要做什么
||    |  |
|--|----|--|
|类型| 表现 | 对 RAG 的影响|
|重复内容|PDF 同一章印两遍、爬虫多 URL 同文|Top-K 被近重复占满，有效信息变少|
|脏 metadata|section_title / section_path 把正文误判成标题|过滤、展示、按节召回全错|
|噪声 chunk|封面、目录、版权、纯导航|误命中、浪费上下文|
|过短/过长块|只有标题、或跨很多页|embedding/BM25 信号差|
|结构断裂|句子在块边界被切断且无 overlap|语义不完整|
|域外/过期|旧版 API、无关附录|答错或幻觉|

 clean_chunks.py 的三类问题，就是上面前几项的实例：重复印刷、标题检测误判、截断标题。


### 怎么做：规则 + 抽检 + 评测闭环

#### 入库时自动体检（第一道闸）
chunk_validator.py 已在做：

* errors：chunk_id 缺失/重复、空正文、页码非法 → 必须拦下。
* warnings：超长、页码跨度过大、Unknown 章节过多、缺 source_uri → 指导你去抽检。

build_chunks.py 里 ingest_document() 会跑这套校验，先看 errors / warnings 再决定是否落盘。

#### 人工抽检（第二道闸）
README 里 Phase 1 要求：至少抽 20 个 chunk，看章节、页码、source_uri 是否和原书一致。抽检要盯：

* 块开头/结尾是否在句中间断开；
* section_path 是否突然多出一句「像正文不像标题」的节点；
* 同一主题是否出现两套页码区间（重复排版信号）。

#### 规则化后处理（第三道闸）
clean_chunks.py 的模式很值得复用：

1. 保守删除：按页码区间删重复印刷（规则 1）。 
2. 按 chunk_id 修标题/路径（规则 2）：只修已确认个案。 
3. 全量映射修 section_path（规则 2b）：脏元素 → 正确父级。 
4. --dry-run 先看统计，再写回；先 .bak 备份。 
5. 清洗后必须：重建向量 + BM25 索引，并更新 golden/smoke 里被删 chunk 的 gold_chunk_ids。

#### 用检索评测验证（第四道闸）
L1_chunk_smoke.md / smoke_queries.yaml 的思路：

* 用真实问法测 any_section_title、any_content_contains； 
* 加 负样本（知识库外问题 expect_hit: false）； 
* 改 chunk 或索引后跑 smoke / golden，看 FAIL 是否集中在某节、某页。

清洗是否有效，以 Recall@K、重复 section 占 Top-K 为准，而不是「感觉干净」。


### 通用清洗清单（可抄到 checklist）
1. 正文（慎用大改，避免改义）

* 统一空白、全角/半角（专名除外） 
* 去掉重复页眉页脚（最好在 PDF 解析层做） 
* 合并「只有标题」的块到下一块，或标记 skip_index

2. Metadata（RAG 里往往比正文更值得洗）

* section_title / section_path / document_id / page_start–page_end / source_uri 
* 保证 chunk_id 稳定，删块时要同步评测集 

3. 去重

* 完全重复：hash(content) 或相似度 > 阈值 
* 近重复：同一节多版排版 → 保留与全书结构衔接的一版（你们用页码区间删旧版）

4. 分块参数

* 技术书：chunk_size + overlap，优先 按章节边界 再按字数切
* 代码/表格：单独策略或保留原始块类型字段



### 本项目中操作流程
```shell
cd rag_mvp
uv run python scripts/build_chunks.py          # 解析 + 校验 → jsonl
uv run python scripts/clean_chunks.py --dry-run
uv run python scripts/clean_chunks.py          # 确认后写回
uv run python scripts/build_vector_index.py
uv run python scripts/build_bm25_index.py
uv run python scripts/run_smoke_retrieval.py   # 或 golden eval
```
