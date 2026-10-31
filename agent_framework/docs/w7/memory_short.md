### 短期记忆如何实现

短期记忆实现为：一份结构化任务状态、一份追加式事件日志，以及一份可更新的历史摘要。每次调用模型，再从这三者中构造上下文。

#### 1. 先确定保存结构

```text
TaskState：当前任务的工作状态
├─ goal                  当前目标
├─ constraints           当前有效约束
├─ completed             已完成事项
├─ next_action           下一步
├─ open_questions        未决问题
├─ artifacts             产物引用
└─ version               状态版本

Events：按顺序追加的原始记录
├─ seq                   任务内递增序号
├─ event_id              事件唯一标识，用于幂等写入
├─ interaction_id        所属完整交互组
├─ role                  user / assistant / tool
├─ payload               消息、工具调用或结果
└─ created_at

Summary：历史压缩视图
├─ body                  摘要内容
├─ covered_through_seq   已覆盖到哪条事件
├─ source_refs           重要结论的原文引用
└─ version
```

这些数据都带上经过应用鉴权的任务定位信息，例如：

```text
tenant_id + user_id + task_id
```
任务状态负责“现在是什么情况”，事件日志负责“具体发生过什么”，摘要负责“以较小成本回顾过去”。

#### 2.保存：什么时间写，怎么写
在以下位置保存：

| 触发时机 | 保存内容 |
|---|---|
| 收到用户消息 | 原始消息；目标或约束的变化 |
| 模型生成工具调用 | 工具调用意图和参数 |
| 工具返回 | 执行结果；进度变化 |
| 模型生成最终回答 | 回答；任务状态变化 |
| 暂停或中断 | 当前执行检查点 |

##### 怎么写
使用版本化的检查点：
```text
checkpoints
├─ tenant_id
├─ user_id
├─ project_id
├─ thread_id
├─ version
├─ state_json
└─ created_at
```


#### 3. 读取：恢复执行和构造模型输入，是两条读取路径

路径 A：恢复执行。

读取任务工作状态和框架检查点：

```python
checkpoint = checkpoint_store.latest(task_id)

state = checkpoint.task_state
next_node = checkpoint.next_node
pending_calls = checkpoint.pending_calls
```
它用于让程序知道接下来执行哪个节点、是否还有待完成调用。

如果采用“快照 + 事件重放”，检查点还需要保存 state_applied_seq，只重放之后的事件。不能恢复一次就把全部事件重新应用一遍。

路径 B：构造下一次模型输入。

通常读取：

```text
当前任务状态
+ 最新历史摘要
+ 摘要覆盖范围之后的完整交互
+ 当前步骤需要回读的原始材料
```
伪代码：
```python
def read_context_material(task_id):
    # 在一致的数据库视图中读取，避免读到半更新状态
    with db.read_snapshot():
        state = db.read_task_state(task_id)
        summary = db.read_latest_summary(task_id)

        cursor = summary.covered_through_seq if summary else 0

        interactions = db.read_interactions(
            task_id=task_id,
            after_seq=cursor,
        )

    return {
        "state": state,
        "summary": summary,
        "interactions": interactions,
    }
```
例如：
```text
全部事件：1～120
摘要覆盖：1～90
当前读取：任务状态 + 摘要 + 事件91～120
```
这里有两个不同的游标：
- state_applied_seq：哪些事件已经反映到任务状态。
- covered_through_seq：哪些事件已经被历史摘要覆盖。
它们解决的问题不同，不能混用。

#### 4.压缩：什么时候触发
在模型调用前检查实际输入长度。

```text
消息可用预算 =
模型上下文上限
- 应用规则
- 工具定义
- 预留输出
- 其他协议与输入开销
```
然后统计准备发送的完整输入：
```python
messages = build_messages(material)
used = tokenizer.count_request(messages, tools)

if used > input_budget:
    compact_history(task_id)
```
也可以设置提前压缩阈值，例如达到可用预算的 80% 时启动，目标降到 60%。这些是可调参数，不是固定标准。

关键是：按 token 预算触发，消息条数只作为辅助条件。

十条长工具结果，可能比一百条短对话更大。

##### 5.压缩：具体压缩什么，怎么压缩
按成本从低到高处理。

第一步：缩小大型工具结果。

例如搜索工具返回 500 条结果，本轮只需要其中 5 条：

```text
原文：
完整搜索结果，保存到事件或对象存储。

上下文：
匹配总数、相关的5条、原文引用。
```
结构示例：
```json
{
  "summary": "找到12处订单写入，其中2处缺少唯一性保护",
  "relevant_items": ["位置A", "位置B"],
  "raw_result_ref": "event:108"
}
```
这比先让模型总结全部聊天历史更便宜，也更容易验证。

第二步：把较早的完整交互压缩成结构化摘要。

选择边界：
```text
较早、已结束的交互 → 可以压缩
最近若干轮交互     → 尽量保留原文
未完成的工具调用组 → 必须保留合法协议结构
```
不能机械地截取前 N 条消息。比如：
```text
assistant：调用工具 call_123
tool：返回 call_123 的结果
```
压缩、裁剪时需要按完整交互组处理，不能留下孤立的工具结果。

摘要可以采用这种结构：

```json
{
  "decisions": [
    {
      "content": "暂不修改数据库结构",
      "source_refs": [12]
    }
  ],
  "verified_observations": [
    {
      "content": "并发测试产生重复订单",
      "source_refs": [47]
    }
  ],
  "failed_attempts": [
    {
      "content": "仅增加应用层查询未解决并发问题",
      "source_refs": [53, 59]
    }
  ],
  "unresolved": [
    "仍需确定幂等键的有效期"
  ]
}
```

而不是笼统总结成“讨论了幂等处理，准备继续”。

当前目标、有效约束、未完成事项，仍保存在 TaskState 中。 摘要不能成为它们唯一的保存位置。


#### 6. 如何安全地提交压缩结果

压缩不是直接“总结完就删除旧消息”。它应当先生成新视图，确认可用后再切换。
```python
def compact_history(task_id):
    snapshot = db.read_memory_snapshot(task_id)

    prefix = choose_old_complete_interactions(
        snapshot,
        keep_recent_turns=6,
    )

    if not prefix:
        return False

    candidate = summarizer.summarize(
        previous_summary=snapshot.summary,
        new_events=prefix,
    )

    validate_summary(
        candidate,
        source_events=prefix,
        previous_summary=snapshot.summary,
    )

    with db.transaction():
        # 压缩期间如果已有另一份摘要提交，重新读取再处理
        db.assert_summary_version(
            task_id,
            snapshot.summary_version,
        )

        db.save_summary(
            task_id=task_id,
            body=candidate,
            covered_through_seq=prefix.last_seq,
        )

    return True
```

校验至少包括：
- 输出结构合法。
- 来源引用确实存在，处于允许引用的范围。
- 目标和关键约束没有被错误改写。
- 假设没有被表达成已验证事实。
- 新摘要确实降低了上下文成本。
其中结构、引用、长度可以由代码检查；语义保真只能部分自动验证，不能宣称摘要是无损的。因此保留原文引用和回读能力很重要。

压缩失败时，不推进摘要覆盖游标，继续使用旧视图，或者缩小压缩批次重试。

连续使用“旧摘要 + 新事件”会累积误差。我会保留原始事件，并在必要时从原文重建相关摘要，而不是无限总结摘要。


#### 7.移除：必须明确移除的是哪一层

区分三种操作：

| 操作 | 实际效果 | 原始记录是否还在 |
|---|---|---|
| 移出当前上下文 | 本次不发送给模型 | 在 |
| 压缩后退出原文窗口 | 用摘要代表较早交互 | 在 |
| 存储清理 | 删除事件、摘要或检查点 | 取决于清理范围 |


第一种：只移出当前上下文。

比如当前正在修改接口，不需要再发送早期的环境安装日志：

```text
context = select_relevant_material(all_material)
```

没有修改原始记录，只是本次不选它。

第二种：用摘要替代旧原文。

只需要更新摘要和覆盖游标：

```text
原来：
摘要覆盖1～60，原文窗口61～120

压缩后：
摘要覆盖1～90，原文窗口91～120
```

事件 61～90 仍然保存在日志里，需要时回读。它们仍是这个任务的历史记录，不会因为退出上下文窗口就自动变成长期记忆。

第三种：从存储中清理。

这通常根据任务状态和保留策略执行：

```text
活动任务：
保留恢复所需的状态、事件、检查点。

已完成任务：
归档，经过保留期后清理。

明确要求彻底删除：
清理涉及的事件、摘要、检查点和派生副本。
```
只删除事件表不一定足够，因为旧检查点和摘要可能仍包含相同内容。


#### 8. 把四个操作放回完整流程
```python
def before_model_call(task_id):
    material = read_context_material(task_id)
    messages = build_messages(material)

    if count_request_tokens(messages) > input_budget:
        compact_history(task_id)

        material = read_context_material(task_id)
        messages = build_messages(material)

    if count_request_tokens(messages) > input_budget:
        # 单条超大输入等问题，摘要历史未必能解决
        messages = offload_large_results_and_rebuild(material)

    if count_request_tokens(messages) > input_budget:
        raise ContextBudgetExceeded()

    return messages
```

不能无限循环压缩，也不能为了塞进窗口悄悄删掉当前用户要求。如果最低必要信息仍然超过预算，就需要拆分输入、分步处理或明确返回容量问题。

在一次任务中的实际变化会是：

```text
收到消息 → 追加事件、更新状态
执行工具 → 保存调用和结果、保存检查点
调用模型前 → 读取状态、摘要、最近交互
输入超预算 → 压缩旧交互、重新组装
任务继续 → 按引用回读被移出窗口的细节
任务完成 → 归档，之后按保留策略清理
```
在 LangGraph 中，可以使用 CheckPointer 保存线程状态，再在模型节点前加入裁剪或摘要逻辑。框架提供持久化、裁剪、删除和摘要相关能力；具体保留什么、如何生成摘要，仍需要应用定义。

这套实现的核心，是让“保存的历史”与“当前发送给模型的内容”分开管理：保存保证可恢复，读取保证相关，压缩控制成本，移除控制窗口和存储生命周期。