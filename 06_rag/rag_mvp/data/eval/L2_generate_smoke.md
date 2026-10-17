## 如何设计

### 准备 10 条验收题与评分标准

#### 确定 10 条测试问题（可直接沿用或微调 data/eval/smoke_queries.yaml 的 10 条，确保覆盖以下类型）：

1. 概念/缩写题（如：MCP 是什么？） 
2. 专有名词/配置题（如：CLAUDE.md 配置文件是做什么的？） 
3. 机制/流程题（如：query 执行循环是怎么工作的？、Auto-Compact 什么时候会触发？） 
4. 架构/列举题（如：Claude Code 的权限分几层？、Claude Code 有哪几种代理模式？） 
5. 无证据/库外题（至少 1～2 条）（如：Kafka 消费者如何提交 offset？，用来验收拒答能力）

#### 为每条问题写好「验收基准」（记录在文档或 YAML 中）：

1. 期望答案包含的 1～2 个核心要点 
2. 期望引用的章节（section_title）或页码范围 
3. 是否应当触发拒答（insufficient_evidence: true）

### 第一轮：检索上下文预检（--dry-run，不调 LLM）

1. 对 10 条问题逐一运行 ask_cli.py --dry-run
2. 检查拼出来的 [1]...[5] Context： 
   3. 前 9 条正例题：Top-5 里是否真的包含能回答该问题的原文？（如果 Context 里压根没有答案，记录为“检索未召回”，不怪 LLM） 
   4. 库外负例题：观察 Top-5 召回了什么噪声内容，预判 LLM 是否会被误导

### 第二轮：真实 LLM 问答与结果落盘

* 检查环境变量：确认 .env 中 DEEPSEEK_API_KEY 配置有效
* 对 10 条问题运行真实问答，并通过 --json 参数将每道题的 RAGResponse 保存到 data/eval/reports/qa_phase3/ 下留档

### 逐题人工验收（对照 4 个核心指标打勾）
对保存下来的 10 份问答结果逐一检查：

1. 指标 1：答案正确性（Correctness） —— 正例题是否准确回答了问题核心要点，没有答非所问 
2. 指标 2：忠实度（Faithfulness / Groundedness） —— 答案中的结论是否 100% 来自检索到的 Context，没有利用模型自身先验知识脱离文档自由发挥 
3. 指标 3：引用准确性（Citation Correctness） —— 
   1. 关键结论句末是否带有 [1]、[2] 标记 
   2. 标记的数字与下方打印的 Citations 列表的 index、section_title、页码、quote 是否真正对应（没有乱标编号） 
4. 指标 4：无证据拒答（Refusal） —— 
   1. 库外题（如 Kafka）在 Top-5 塞入无关文档块的情况下，模型是否严格输出“当前资料中没有足够依据” 
   2. 返回结构中的 insufficient_evidence 字段是否被正确置为 True

### 问题修复与工程收尾
* 修复拒答漏洞（若负例题失败）： 
  * 若模型对无关 Context 强行作答：微调 prompts.py 中的拒答判定条件，或在 pipeline.py 中增加检索最低分数阈值拦截 
* 修复引用格式问题（若漏标/错标）：
  * 若模型不标 [n] 或造出 [6] 等越界编号：在 Prompt 中强化格式约束，并在 citations_referenced_in_answer 中过滤越界编号
* 清理 Phase 3 遗留工程债务：
  * 修正文件名拼写（如 src/generation/shcame.py $\rightarrow$ schema.py）
  * 清理重复模块（确认统一保留 src/generation/ 还是 src/rag/，删除多余副本并更新测试导入）
  * 运行 uv run pytest 确保所有单元测试通过

