# Phase 3 QA 人工验收清单

生成时间：2026-10-17T12:31:14

自动检查已预填到对应指标后；「正确性 / 忠实度」需人工对照期望要点打勾。

---

## qa-001：MCP 是什么？

- tags：concept, english_abbr　期望拒答：False
- 自动检查：通过

期望要点：
  - [ ] Model Context Protocol（开放协议）
  - [ ] 定义 AI 模型与外部工具/资源之间的标准通信方式（或提及 Tools/Resources/Prompts）

四项指标：
  - [ ] 1. 答案正确性（核心要点是否准确、有无答非所问）
  - [ ] 2. 忠实度（结论是否 100% 来自 Context，无模型自由发挥）
  - [x] 3. 引用准确性（自动：标记 [1, 2, 4]，越界 无；仍需人工核对编号与 Citations 是否对应）
  - [x] 4. 无证据拒答（自动：insufficient_evidence=False，符合期望）

答案与引用存档：`data/eval/reports/qa_phase3/qa-001.json`

---

## qa-002：CLAUDE.md 配置文件是做什么的？

- tags：keyword, exact_name　期望拒答：False
- 自动检查：通过

期望要点：
  - [ ] 作为项目的显式记忆/上下文说明文件，启动时自动读取
  - [ ] 支持全局（~/.claude/CLAUDE.md）、项目根目录、子目录多级合并

四项指标：
  - [ ] 1. 答案正确性（核心要点是否准确、有无答非所问）
  - [ ] 2. 忠实度（结论是否 100% 来自 Context，无模型自由发挥）
  - [x] 3. 引用准确性（自动：标记 [1, 2, 4]，越界 无；仍需人工核对编号与 Citations 是否对应）
  - [x] 4. 无证据拒答（自动：insufficient_evidence=False，符合期望）

答案与引用存档：`data/eval/reports/qa_phase3/qa-002.json`

---

## qa-003：query 执行循环是怎么工作的？

- tags：semantic, paraphrase　期望拒答：False
- 自动检查：未通过（拒答不符期望：insufficient_evidence=True，期望 False）

期望要点：
  - [ ] 检查轮次限制并流式调用 Claude API，解析 text / thinking / tool_use
  - [ ] 若有工具调用则（并行）执行工具并将结果追加回 messages，检查 token 预算后继续循环直到无工具调用

四项指标：
  - [ ] 1. 答案正确性（核心要点是否准确、有无答非所问）
  - [ ] 2. 忠实度（结论是否 100% 来自 Context，无模型自由发挥）
  - [x] 3. 引用准确性（自动：标记 [1, 2, 3, 5]，越界 无；仍需人工核对编号与 Citations 是否对应）
  - [ ] 4. 无证据拒答（自动：insufficient_evidence=True，不符合期望）

答案与引用存档：`data/eval/reports/qa_phase3/qa-003.json`

---

## qa-004：Auto-Compact 什么时候会触发？

- tags：feature_name　期望拒答：False
- 自动检查：通过

期望要点：
  - [ ] token 使用量超过 85% 时进入 warning（建议压缩/显示警告）
  - [ ] token 使用量超过 95% 时进入 critical（必须立即/强制压缩）

四项指标：
  - [ ] 1. 答案正确性（核心要点是否准确、有无答非所问）
  - [ ] 2. 忠实度（结论是否 100% 来自 Context，无模型自由发挥）
  - [x] 3. 引用准确性（自动：标记 [1, 2, 5]，越界 无；仍需人工核对编号与 Citations 是否对应）
  - [x] 4. 无证据拒答（自动：insufficient_evidence=False，符合期望）

答案与引用存档：`data/eval/reports/qa_phase3/qa-004.json`

---

## qa-005：对话历史压缩的核心思路是什么？

- tags：semantic　期望拒答：False
- 自动检查：通过

期望要点：
  - [ ] 用 Claude 自己总结较早的对话历史生成摘要
  - [ ] 找到压缩边界，保留最近 N 条（如最近 10 条）完整消息不压缩，将摘要与保留消息拼接

四项指标：
  - [ ] 1. 答案正确性（核心要点是否准确、有无答非所问）
  - [ ] 2. 忠实度（结论是否 100% 来自 Context，无模型自由发挥）
  - [x] 3. 引用准确性（自动：标记 [1, 2, 3, 5]，越界 无；仍需人工核对编号与 Citations 是否对应）
  - [x] 4. 无证据拒答（自动：insufficient_evidence=False，符合期望）

答案与引用存档：`data/eval/reports/qa_phase3/qa-005.json`

---

## qa-006：Claude Code 的权限分几层？

- tags：semantic, architecture　期望拒答：False
- 自动检查：通过

期望要点：
  - [ ] 分为五层（从粗到细）
  - [ ] 会话模式、工具白名单/黑名单、工具级权限、操作级权限、路径/命令级权限

四项指标：
  - [ ] 1. 答案正确性（核心要点是否准确、有无答非所问）
  - [ ] 2. 忠实度（结论是否 100% 来自 Context，无模型自由发挥）
  - [x] 3. 引用准确性（自动：标记 [1, 2, 3]，越界 无；仍需人工核对编号与 Citations 是否对应）
  - [x] 4. 无证据拒答（自动：insufficient_evidence=False，符合期望）

答案与引用存档：`data/eval/reports/qa_phase3/qa-006.json`

---

## qa-007：QueryEngine 的 submitMessage 做什么？

- tags：keyword, section_number　期望拒答：False
- 自动检查：未通过（拒答不符期望：insufficient_evidence=True，期望 False）

期望要点：
  - [ ] 预处理用户输入（斜杠命令、附件、Memory）并构建消息列表
  - [ ] 调用 query() 执行核心循环，随后做后处理（记录 token、保存历史、触发 hooks、检查压缩）并流式返回结果

四项指标：
  - [ ] 1. 答案正确性（核心要点是否准确、有无答非所问）
  - [ ] 2. 忠实度（结论是否 100% 来自 Context，无模型自由发挥）
  - [x] 3. 引用准确性（自动：标记 [1, 3]，越界 无；仍需人工核对编号与 Citations 是否对应）
  - [ ] 4. 无证据拒答（自动：insufficient_evidence=True，不符合期望）

答案与引用存档：`data/eval/reports/qa_phase3/qa-007.json`

---

## qa-008：流式响应是怎么解析的？

- tags：semantic　期望拒答：False
- 自动检查：通过

期望要点：
  - [ ] 基于 SSE 事件流（content_block_start / content_block_delta / content_block_stop / message_stop）解析
  - [ ] 文本块（text_delta）实时输出给用户，思考块内部处理，工具调用块（input_json_delta）累积完整 JSON 后再执行

四项指标：
  - [ ] 1. 答案正确性（核心要点是否准确、有无答非所问）
  - [ ] 2. 忠实度（结论是否 100% 来自 Context，无模型自由发挥）
  - [x] 3. 引用准确性（自动：标记 [2, 3, 5]，越界 无；仍需人工核对编号与 Citations 是否对应）
  - [x] 4. 无证据拒答（自动：insufficient_evidence=False，符合期望）

答案与引用存档：`data/eval/reports/qa_phase3/qa-008.json`

---

## qa-009：Claude Code 有哪几种代理模式？

- tags：semantic　期望拒答：False
- 自动检查：通过

期望要点：
  - [ ] 三种代理模式：子代理（AgentTool）、后台代理（run_in_background）、Worktree 隔离代理
  - [ ] （或若检索落在 17.3）内置专业化代理类型：general-purpose、Explore、Plan

四项指标：
  - [ ] 1. 答案正确性（核心要点是否准确、有无答非所问）
  - [ ] 2. 忠实度（结论是否 100% 来自 Context，无模型自由发挥）
  - [x] 3. 引用准确性（自动：标记 [1]，越界 无；仍需人工核对编号与 Citations 是否对应）
  - [x] 4. 无证据拒答（自动：insufficient_evidence=False，符合期望）

答案与引用存档：`data/eval/reports/qa_phase3/qa-009.json`

---

## qa-010：Kafka 消费者如何提交 offset？

- tags：negative, out_of_domain　期望拒答：True
- 自动检查：通过

期望要点：
  - [ ] 明确说明当前参考资料中没有足够依据，不利用模型自身知识回答 Kafka 机制

四项指标：
  - [ ] 1. 答案正确性（核心要点是否准确、有无答非所问）
  - [ ] 2. 忠实度（结论是否 100% 来自 Context，无模型自由发挥）
  - [x] 3. 引用准确性（自动：标记 [1, 2, 4]，越界 无；仍需人工核对编号与 Citations 是否对应）
  - [x] 4. 无证据拒答（自动：insufficient_evidence=True，符合期望）

答案与引用存档：`data/eval/reports/qa_phase3/qa-010.json`
