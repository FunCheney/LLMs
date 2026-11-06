# 开发日志

> 每周五合流仪式第 4 步：记一条「本周做了什么 / 指标变化 / 遗留问题」。
> 指标快照写在 `evals/history/YYYY-Www.json`，本文件只记**结论与判断**。

---

## 2026-W48（11-23 ~ 11-27）· W5：生产 RAG I（pgvector / 混合检索 / 精排 / 改写）

> 实际落盘 11-06 预跑；六档检索已量化，但**检索还没接进 `/v1/draft`**，
> 所以本期没有任何一次「线上请求含检索」的延迟读数。

### 本周做了什么

| Day | 产出 | 结论 |
|---|---|---|
| D1 | `knowledge/base.py` + `PgVectorStore` + `ops/pgvector/init.sql` + 十项决策 ADR | 协议先行落地：pgvector 只实现协议，业务层 grep `psycopg\|pgvector` = **0 处** |
| D2 | `retrieval/fusion.py`(RRF) + `pipeline.py` + `evals/scripts/retrieval_metrics.py` | **实测混合不如单路稠密**，默认档从 `hybrid` 改 `dense` |
| D3 | 精排装配 + 三条降级路径 + `rerank_preflight()` | 精排是负收益，但先查清了一个更糟的事实：默认超时让它在 CPU 上**永远不生效** |
| D4 | `query_rewrite.py`（multi-query / HyDE）+ 接进装配 | 改写也是负收益；顺手修掉一个死开关 |
| D5 | `rag-ab-report.md` + `case-1` + ADR 十项决策 | 「全负」不是本周失败，是本周最硬的结论 |

### 指标变化

- 离线用例：`496 → 602`（框架侧 +106），rag_mvp 侧另 +34；live 22 跳过
- 检索（28 正例 / 425 chunks / bge-m3 1024 维，`retrieval_metrics.py` 生产口径）：
  - 纯稠密：R@1 `.7857` / R@10 `1.0000` / MRR `.8628` / nDCG `.7416`，p95 `175ms`
  - 纯 BM25：R@1 `.2500` / R@10 `.7143` / MRR `.4122`，p95 `0.9ms`
  - 混合 RRF：R@1 `.5714` / R@10 `.9286` / MRR `.6921` —— **三项全低于纯稠密**
  - +精排（20 候选）：R@1 `.7143` / MRR `.8146`，p95 `5381ms`
  - +multi-query：R@1 `.7143~.7857`（两轮不同）/ MRR `.8089~.8482`，p95 `15.2s`，¥0.0888 / 25 次
- 后端等价：local 与 pgvector `5/5` 同 chunk、分数小数点后 6 位一致
- 静态检查：`ruff` 全绿、`mypy` 53 文件无问题
- 全量测试：`602 passed / 22 skipped`（59s）

> **门禁这次是红的，但红得不是地方。** `latency_p95_ms` 2.29ms → 30.81ms（+1244%）。
> 判为环境伪影，不是代码劣化，三条判据：①30ms 对每条 case 恒定、方差 0.6ms，
> 连 error case 也 ≥30ms，不符合真实链路抖动；②检索未接入 `/v1/draft`，
> 本期跑的路径与 W47 逐字节一致；③cProfile 把 3.44s 定位到
> `_socket.socket.recv` + `sitecustomize.py:361(_broker_readline)` 109 次——
> 记录这个数字的 python 进程被 CLI shim 注入了 socket broker。
> 处理：本期延迟**不进基线**，裸终端重跑后替换。同类现象 W4 已记过一次（`tmp_path` 无写权限）。

### 本周暴露的五个真问题

1. **「混合检索更好」是教科书假设，不是本地事实**：sparse 路 R@1 只有 `.25`，
   而 RRF 的 `1/(k+rank)` 把「dense 排 1、sparse 排 30」的正解算成 `0.0275`，
   「两路都排 5」的邻居页得 `0.0308`。**等权融合在惩罚单路强信号**，这是算法性质。
2. **分词器把纯标点当 token**：`tokenize("如何配置 MCP 服务器？")` 产出 `'？'`。
   golden-018 的 sparse top1 是一个含 12 个「？」的列表页。去纯标点后 sparse R@1
   `.2500 → .5357`（+114%）——**修好了分词，混合仍然不如 dense**，两件事要分开说。
3. **默认超时值可以让组件「永远不生效」**：`rerank_timeout_ms=3000`，本机 CPU 20 条要
   4.7s，于是 28/28 全降级、指标等于没开。这类默认值比「功能缺失」更危险，
   它让人以为「我试过了，没用」。
4. **死开关比没实现更危险**：`retrieval_rewrite` 字段存在、pipeline 支持 `rewriter=` 注入，
   但 `rag.py` 从没读过它。设了没有任何效果，也不报错。
5. **重排是在删除信息，不是重排**：`hybrid+rerank` 与 `dense+rerank` 五个指标**完全相同**；
   golden-022 的 gold chunk 从第 8 位掉出 top10，而同一页的邻居被顶上来。
   精排偏好「主题邻近段落」，题目要的是「恰好回答的那一段」。

### 遗留问题（带责任人与时限）

| # | 问题 | 处理 |
|---|---|---|
| 1 | 检索未接 `/v1/draft` | 唯一功能缺口；接上之前 Gate 5 #5 只能是 partial，p95 门禁也覆盖不到检索 |
| 2 | 28 条正例不足以支撑细粒度结论 | R@1 差 6 条、MRR 差 15.7pp 算系统性；R@10 差 1 条不算 |
| 3 | 桥接层不转发 `filters` | W6 的 ACL 若直接依赖会静默越权，先修桥接 |
| 4 | 镜像缺 psycopg | Dockerfile 补 `--extra pgvector`；完整构建仍卡 docker.io 502 |
| 5 | 改写两轮结果不一致 | `temperature=0.3` 有随机性；任何改写结论需三次取中位数 |

### 验收口径的修订（需确认）

Gate 5 第 3 条原写「Recall@K 相对 baseline 提升 ≥10%」。**这条无法验收**——
纯稠密 R@10 已经是 `1.0000`，没有 10% 的空间。

改为：**在 R@10 触顶的前提下以 R@1 / MRR 为主指标；混合与精排为负收益，如实记录并默认关闭。**
这不是降标准，是把「提升」从「多叠加组件」换成「选对单路 + 证明其余无效」——
后者更难，因为它要求每个被关掉的组件都有反向证据。

### 下周（W6）只做三件事

1. `/v1/draft` 接检索，把 `context_precision` 从「引用质量代理」换成真实 chunk 前 k 条；
2. ACL 走 `filters`（桥接层当前不转发，等于越权），同时做 `delete_document` 的增量更新；
3. 裸终端重跑一次 `run_eval` 换掉本期被污染的延迟块，恢复门禁可比性。

---

## 2026-W47（11-16 ~ 11-20）· W4：评测体系 II（Judge / RAGAS / 回归门禁 / 四层归因）

> 实际落盘 11-03 预跑；judge 一致性来自**规则基线**（无 key），LLM judge 与在线档待凭证。

### 本周做了什么

| Day | 产出 | 结论 |
|---|---|---|
| D1 | `evals/judge/{rubric,llm_judge,stats}.py` + `judge_agreement.py` + `adr-w4-judge-design.md` | **一致性系数才是交付物**：没有 kappa 的 Judge 等于随机数，"分数涨了"就无法被追问 |
| D2 | `evals/judge/{ragas_metrics,text}.py` + `metric-spec-ragas.md` + `metrics.RagasScores` | `None ≠ 0`：无 context 返回 None，否则 faithfulness 随数据集构成漂移 |
| D3 | `regression_gate.py` + `baseline.json` + CI 两步 + `adr-w4-gate-threshold.md` | 相对劣化 + 延迟绝对下限；**基线缺失退出 2，不静默放行** |
| D4 | `root-cause-playbook.md` + `attribute_failure.py` | 证据不足判 `unknown`：把"埋点缺失"判成"某层坏了"，团队会去改一个没坏的模块 |
| D5 | `docs/w4/cases/case-{1,2,3}.md` + 3 处真实代码改动 | 每篇都有前后数字，且都不是"全绿"——真实性来自取舍被写出来 |

### 指标变化

- 离线用例：`416 → 496`（judge 24 / ragas 20 / gate 17 / 归因 15 / 分类不变量 4），live 仍 16 跳过
- **`make run-eval-judge`：50 passed / 0 failed / 4 skipped，2.12s**
- Judge：`judge_score 1.92`（50 条，alias `heuristic-keyword`）、rubric `answer-quality@w4-v1`
- 一致性（24 条人工标注）：加权 kappa `0.343`、名义 kappa `0.200`、spearman `0.437`、MAE `1.167`
  → **阈值 0.40 未通过**（结论见下）
- RAGAS（9 条带引用）：`faithfulness 0.0`、`answer_relevancy 0.054`、`context_precision 1.00`
- 延迟：混合 p95 `2.45ms`，但 chat p95 `0.47ms`、tool_call `0.74ms`、error p50 `3.95ms`
- 回归门禁：对比 W46 基线，`2 项可比 / 4 项跳过 / 0 违规`，通过
- 静态检查：`ruff` 全绿、`mypy` 45 文件无问题
- 全链路 `make eval WEEK=2026-W47` 一次通过：`check → golden → run-eval-judge → gate`
  （`judge_score` 判跳过是因为基线是 W46、那时还没有这个指标，不是本期缺数据）

> 排障记录：若在**只读沙箱**里跑 `make test`，会看到 20+ 个
> `ERROR ...::test_x - assert not self._finalizers`，散布在 `test_config` / `test_regression_gate`
> 等互不相关的模块。这不是代码问题：`tmp_path` 无写权限建不出来，teardown 被打断后
> finalizer 留在 fixturedef 上，之后每个用 `tmp_path` 的用例都在同一行炸。判据是
> "换到可写环境立刻全绿"——别去改 conftest。

### 本周暴露的五个真问题

1. **references 只有路径时，RAGAS 结构性为 0**：context 是 `docs/architecture.md` 这样的字符串，
   实词是 `docs/architecture/md`，任何答案都命中不了。指标恒 0 不代表模型差，代表我们把路径当成了正文。
2. **单字口径会把 faithfulness 顶到 1.0**：context 换成整篇文档后，任何中文句子都能凑够单字重合。
   换成二元组后回落——**同一份数据、两种分词，结论相反**，指标口径必须写进文档。
3. **一个 p95 混四种调用形态**：混合 2.45ms 是 chat p95 的 5.2 倍，误差量级 420%，
   抬它起来的是 4 条故意重试的 error case。读者会去查一个没变慢的链路。
4. **错误码表写两遍必然漂移**：10 个已知码里 6 个判定相反，`SERVER_BUSY`（自家限流器）在
   `run_eval` 里是"质量失败"、在分类器里是"上游问题"。漂移不报错，只让两个数字互相矛盾。
5. **规则基线判不了事实**：它从不给 1–2 分，因为"事实错误"在词面上往往比正确答案更像个答案。
   kappa 0.20 就是这件事的数字，也是 LLM judge 必须存在的理由。

### 遗留问题（带责任人与时限）

| # | 问题 | 处理 |
|---|---|---|
| 1 | LLM judge 一致性未测（当前 0.20 是规则基线） | 填 key → `judge_agreement.py --live`；低于 0.40 先改锚，不改温度 |
| 2 | 24 条标注 `labeler=impl-draft` | 负责人逐条复核改成 `human`；自评自标只能证明流程通 |
| 3 | 门禁拦不住"答得差" | 离线 success_rate 恒 100%；建 `baseline-online.json` 后才真正有质量门禁 |
| 4 | `faithfulness` 离线恒 0 | 夹具答案是固定文本，与引用无关；W5 接检索后才有读数 |
| 5 | 字面判定认不出同义改写 | 需要 `LLMSupport`（走 strong 模型）；成本约 0.4 美分/条 |

### 收尾时补的一个坑

`run-eval` 与 `run-eval-judge` 写的是**同一个** `evals/history/<WEEK>-eval.json`，谁后跑谁说了算。
`make eval` 原来走 `run-eval`，于是本地一跑就把刚产出的 `judge`/`ragas` 块冲成 `null`；
下一期对比时质量指标会"本期缺失 → skipped"，趋势图悄悄断掉，而且**不会报错**。
已把 `eval` 改成依赖 `run-eval-judge`（离线档是规则基线、零成本）。同类隐患记一条原则：
同一份产物有两个写入者时，默认路径必须选"信息更全"的那个，否则隐性丢数据。

### P3 决策（W4 末 = ROADEMAP §7 第一个决策点）

**不解锁深入档。** 本周预算 12–15h 全部用于默认档，进度余量 < 15h；
按 ROI 顺序（A2A 轻量 > 多语言 > 前端极简 > 高并发 > 向量库/微调 > Computer Use），
解锁任何一个都会挤压 W5 的 RAG（P0 主线）。余量补上后再判。

### 下周（W5）只做三件事

1. 接检索（chunk + 混合召回），把 `context_precision` 从"引用质量代理"换成真实检索前 k 条；
2. 用 D3 的门禁 + D4 的归因跑第一轮真实调优，产出第 4 篇案例；
3. 首次在线档，回填 judge kappa 与 faithfulness 的真实读数。

## 2026-W46（11-09 ~ 11-13）· W3：评测体系 I（golden set / 分层指标 / trace 落地）

> 实际落盘 11-02 预跑；数字全部来自离线档，在线档与 Langfuse 真机验证待凭证。

### 本周做了什么

| Day | 产出 | 结论 |
|---|---|---|
| D1 | `docs/w3/metric-spec.md` + `evals/metrics.py` + `docs/w3/adr-w3-eval-scope.md` | **口径必须先于 runner**：四态机（passed/failed/interrupted/skipped）定完，`success_rate` 的分母才有意义 |
| D2 | `evals/golden/{golden_v1.jsonl,schema.md}` + `validate_golden.py` | 54 条、难度 12/26/16、4 条对抗；校验器对待"重复输入/悬空引用/拼错判据键"零容忍 |
| D3 | `evals/scripts/{run_eval,offline_fixtures}.py` + `Makefile` 的 `run-eval` | 一条命令出分层指标 + md + json；`--break-case` 让"单条失败不中断"可被反向证明 |
| D4 | `exporters/langfuse.py` + `server/app.py` 注入点 + `adr-w3-trace-backend.md` | **缺 key 返回 Noop 而不是抛错**：否则没有账号的机器连测试都跑不了 |
| D5 | `classify_failures.py` + `report_template.md` + 报告自动渲染 | 规则先行：`UPSTREAM` 与 `HALLUCINATION` 混淆是最贵的归因错误 |

### 指标变化

- 离线用例：`337 → 416`（新增指标层 24 / golden 校验 13 / runner 11 / 分类器 13 / exporter 18），live 仍 16 跳过
- golden：`10 → 54` 条，难度 `12 / 26 / 16`，对抗样例 4 条，引用 20 处全部可达（anchor 逐条命中）
- **`make run-eval`：50 passed / 0 failed / 0 interrupted / 4 skipped，2.17s，零网络**
- 分位：`latency p50 0.17ms / p95 1.80ms`（50 样本）；首 token 仅 4 个流式样本，**p95 按规则返回 null**
- 成本：`0.0 CNY`（fixture 不计价）；`usage_estimated_ratio 0`
- **质量：`quality` 块恒 `null`** —— 离线跑通的 100% 不是质量结论
- 静态检查：`ruff` 全绿、`mypy` 45 文件无问题

### 本周暴露的六个真问题

1. **口径先于代码**：先写 runner 后补定义，指标会变成"代码碰巧算出来的东西"。
   `success_rate` 的分母一旦把上游中断算进去，一次厂商发版就能把质量分打掉 20 个点，
   然后团队去调 prompt——查三天发现是对方 503。
2. **`usage_estimated` 混进均值 = 无意义的数**：既不是实测也不是上界。宁可 `cost_per_task=null`
   并让 `usage_estimated_ratio` 说明原因，也不给一个能画在图上但解释不了的数。
3. **"多币种"报警其实是错误路径的默认值**：错误日志没有价目，`currency` 落的是 dataclass 默认的 USD。
   把带 `error_code` 的样本排除出币种判定即可，不必回头改 W1 的 emit。
4. **对抗样例不能有离线判据**：补一个 `offline.expect` 让 CI 变绿，等于"用夹具回放证明模型会拒绝"，
   比没有测试更糟。宁可让它长期 skipped，并在报告里显式展示 7.4% 的覆盖率缺口。
5. **离线判据抄在线判据必然二选一**：恰好成立（假绿）或永远不成立（红一片后被人放宽）。
   离线只断言 `require_log` + `finish_reason`，在线才断言语义。
6. **`ruff` 的 line-length 按显示宽度算**：85 个字符的中文注释被判成 105。按 `len()` 折行写了等于没写。

### 遗留问题（带责任人与时限）

| # | 问题 | 处理 |
|---|---|---|
| 1 | 在线档质量分仍为 null | 填 `.env.local` 后 `make run-eval WEEK=2026-W47 --live`，回填 quality 与模型快照日期 |
| 2 | Langfuse 只做过单测与降级验证 | 注册账号 → 配 `LANGFUSE_*` → 跑一次 `/v1/chat` → 确认 `llm.call` span 与 trace 树 |
| 3 | 4 条对抗样例长期 skipped | W4 接 Judge 后改为在线档评估；每期报告保留 skip_rate |
| 4 | `judge_score` 字段恒 null | W4 接 LLM-as-judge，在此之前"质量趋势"只有成功率一条线 |

### 下周（W4）只做三件事

1. 接 LLM-as-judge，把 `judge_score` 从 null 变成数字（含 judge 与被评模型不同的证据）；
2. 把 `make eval` 升级成"指标不得劣化"的对比门禁（对上一期 history 做差）；
3. 首次在线档跑通，产出第一批真实延迟/成本/质量数据点。

---

## 2026-W45（11-02 ~ 11-06）· W2：运行底座 + 可观测基线 + 评测冒烟集

> 实际落盘时间 11-01（周日）预跑；数字全部来自离线档，在线档待 key。

### 本周做了什么

| Day | 产出 | 结论 |
|---|---|---|
| D1 | `session/{base,memory,redis,factory}.py` + `workflow/research_workflow.py` + `docs/w2/adr-w2-session-boundary.md` | 幂等键必须三档解析（显式 → contextvar → uuid）；少第三档，脚本与队列消费者一调就误判重复 |
| D2 | `resilience/{concurrency,errors}.py` + `queue/{base,inmemory}.py` + `server/routes_task.py` | 入口过载 `retryable=False`：让 gateway 再重试一次，100 并发立刻变 300（压测直接崩） |
| D3 | `observability.py`（+`trace_id`/`span_id`）+ `tracing.py`（Span/span()/Exporter） | `trace_id` 必须在中间件生成，放进 `span()` 会让同一请求的多个 span 各拿一个新 trace |
| D4 | `ops/{Dockerfile,docker-compose.yml,.env.example}` + `evals/scripts/locustfile.py` + 压测报告 | 客户端口径与服务端口径差一个数量级，延迟结论只能取 `RequestLog` |
| D5 | `evals/golden/smoke_v1.jsonl`（10 条）+ `run_smoke.py` + CI + `Makefile` | 门禁必须先证明"能红"：CI 里改坏一条 expect，退出码非 0 才算门禁成立 |

### 指标变化

- 离线用例：`278 → 337`（新增会话/限流/队列/trace/任务端点/smoke runner 六组），live 仍 16 个跳过
- **100 并发离线压测：1224 请求 / 0 失败 / `retry_count>0` 为 0 / 994 条日志 994 个 trace_id**
- **端到端自检（`make e2e`）**：限额 1 + 20 并发（每请求 200ms）→ `200×1 / 503×19`、`Retry-After: 1`、`code=SERVER_BUSY`；`/v1/tasks` 走到 `succeeded`；响应头 `X-Trace-ID` 与日志 `trace_id` 一致
- 首 token（port 口径）：p50 `20.9 ms`、p95 `32.5 ms` —— **仅证明埋点链路正确**，fixture 回放不代表真实模型
- 成本：仍为 0（fixture 不计价）；质量：smoke `10/10`，只有"没坏"这一档
- 静态检查：`ruff` 全绿、`mypy` 43 文件无问题

### 本周暴露的五个真问题（都不是写代码时能想到的）

1. **"实现慢"可能是机器慢**：全量串跑 15 分钟、单文件跑 6 秒，同一份代码。宿主调度一停顿，
   客户端 p95 就从 35 ms 涨到 1200 ms。**结论**：延迟只认服务端 `RequestLog`，客户端分位数只能横向比。
2. **幂等的兜底键不是可选项**：没有"显式 → contextvar → 本地 uuid"第三档，`request_id=""` 会让
   第二次 `append` 被判成重复，表现为"多轮对话第二轮丢消息"——只在真实多轮里复现。
3. **入队前必须 `precheck`**：alias 打错这类注定失败的请求，占了队列槽位排 30 秒再 failed，
   等于把调用方的输入错误升级成我们的容量问题。
4. **compose 的 healthcheck 不能指向未就绪端口**：`depends_on` 只保证"容器起了"，不保证"能连"，
   `agent-server` 起得比 redis 慢时，首批请求全是 `Connection refused`。
5. **`mock` 压不出并发**：回放是同步读文件、全程不 await，事件循环里天然一次只有一个在飞。
   用它验证"限额 1 会不会 503"会得到 30 并发全 200 的假绿——压测对象先要能并发，结论才有意义。

### 遗留问题（带责任人与时限）

| # | 问题 | 处理 |
|---|---|---|
| 1 | 在线档延迟/成本未测 | 有 key 时 `MODEL=qwen3.7-plus make load`，报告记时段（峰谷差 2 倍） |
| 2 | `docker build` 未在本机执行（无 docker） | compose 仅静态检查；首次 `make up` 时补 `docker compose ps` 无 restart 的记录 |
| 3 | 队列进程内实现 | 多副本会丢任务；换 Redis Stream 时 `TaskQueue` 签名不变 |
| 4 | 满载 503 已由 `make e2e` 覆盖，但只在进程内验证 | 真机/容器内复跑一次 `make e2e`，确认反向代理层没有改写 `Retry-After` |
| 5 | smoke 只判"全过" | W4 升级为"指标不得劣化"，需要先有 3 周基线 |

### 下周（W3）只做三件事

1. golden set 扩到 50+（每类覆盖从 2–3 条提到 8–10 条），判据从"文本包含"升级到结构化断言；
2. 接 Langfuse：只实现 `LangfuseExporter` 替换 `NoopExporter`，`RequestLog` 字段一个不改；
3. 记忆层（W7 前置）只定接口，不实现——先把"哪些事实值得记"的判据写进文档。

---

## 2026-W44（10-26 ~ 10-30）· W1：LLM 适配层 + 首个纵向切片

### 本周做了什么

| Day | 产出 | 结论 |
|---|---|---|
| D1 | 四家 API 差异矩阵（`docs/w1/*.md`）+ uv workspace 重构 | "system 顶层 / tool_result 成组 / arguments 是字符串 / usage 挂在哪"这四件事决定归一化粒度 |
| D2 | 契约层 `types / errors / base` | 唯一一张 finish_reason 映射表；`provider_meta` 承载"框架不解释但必须回传"的字段 |
| D3 | `openai_compat` / `anthropic` / `mock` + conformance A1–A8 | 差异走 flags 表，不写 `if "deepseek" in base_url`；mock 只回放 wire，不自己造 `LLMResponse` |
| D4 | `config / observability / registry / router / gateway / server` + A9–A12 | 重试与 committed 状态机是客户端层唯一职责；`precheck` 必须在 `StreamingResponse` 之前 |
| D5 | 录制器 / 基线表 / live 用例 / A13–A14 / 产品端 `/v1/draft` | 框架"能被消费"这件事必须由产品端真跑一遍来证明，自测不算 |

### 指标变化

- 离线用例：`136 → 222 → 278`（Day3 → Day4 → Day5），live 用例 16 个默认跳过
- conformance 断言：`A1–A12 → A1–A14`（新增"录制 ⊇ 手写基线"与"工具调用落在声明 schema 内"）
- 首 token / 成本 / 质量：**仍为 null**，等真机录制与 W2 golden set
- 静态检查：`ruff` 全绿、`mypy` 26 文件无问题

### Day5 过程中暴露的四个真问题（都不是写代码时能想到的）

1. **日志静默丢弃**：`RequestLog` 打 `INFO`，而 root logger 是 `WARNING`。代码里 `emit()` 明明调了，
   `uvicorn` 起来后日志里一个字都没有——Gate 验收项「`first_token_ms` 入日志」会假通过。
   修法：`observability.configure_logging()`，在应用边界（`create_app`）调用，库本身不改宿主配置。
2. **DeepSeek 是峰谷分时计价**（工作日 09–12 / 14–18 高峰翻倍），Qwen 是**按上下文长度阶梯**
   （≤256K 为 2/8，>256K 跳到 6/24）。单一 `pricing.input/output` 表达不了，W1 取常规档并在
   `models.yaml` 里注明；基线表必须记时间戳，否则"成本降了 40%"可能只是换到了空闲时段。
3. **产品层拿不到中间件**：`RequestIDMiddleware` 原本长在 `server/app.py` 里，产品要复用它就得
   执行该模块顶层的 `app = create_app()`——为了拿一个中间件而启动一个框架应用，是边界画错的信号。
   抽到 `server/middleware.py` + `server/errors.py` 后，产品端零 `try/except` 就复用了 6 类错误码。
4. **录制器不能覆盖 `request.json`**：它是 A1 的断言锚（`build_request` 必须精确产出它）。
   覆盖了 A1 就变成"自己跟自己比对"。改为：录到不一致就硬失败，让人先决定锚该不该更新。

### 遗留问题（带责任人与时限）

| # | 问题 | 处理 |
|---|---|---|
| 1 | 真机录制未跑（手写夹具可能偏离真实协议） | 有 key 时执行 `README` 的 Day5 三连；A13 会在录制后清算 |
| 2 | `first_token_timeout_s=10s` 未经真机校准 | `uv run pytest -m live -k first_token` 取实测值后调整 |
| 3 | Anthropic 欠费仍靠文案 `credit balance` 识别 | 真机确认后收敛成固定 code 白名单 |
| 4 | 峰谷 / 阶梯计价未建模 | W11 做成本看板时改为按 (时段, 上下文长度) 取价 |
| 5 | 错误路径（`usage_missing` / `error_*`）无法安全录制 | 长期保留手写 + `error_auth` 真机探测这两条路 |

### 下周（W2）只做三件事

1. golden set 最小 10 条（含 3 条对抗样例），接进 `pytest` 做门禁；
2. 压测脚本：并发 20 × 60s，产出 p50/p95/错误率，回填 `evals/history`；
3. Temporal 长任务骨架（`legacy/temporal_demo` 的思路复用），先跑通"调研分 3 步可续跑"。
