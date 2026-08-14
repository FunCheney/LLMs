## 验证 KV Cache 失效设计

### 验证思路总览

在 可控、可复现 的 API 实验里，用与 Cursor Agent 同构的 prompt 分段，构造 1 个正确基线 + 5 类故意破坏前缀一致性 的对照；
以 缓存命中为因果、TTFT/成本为后果，证明「哪些改动会让 KV/Prompt Cache 失效，以及失效有多贵」。

#### 为什么这样验

|问题| 思路 |
|---|----|
|Cache 是序列级、非语义级|用 字节/顺序级 扰动（时间戳、工具 shuffle、格式 flatten），而非「意思差不多」|
|TTFT 会骗人|sliding_window 更短可能更快，但每轮 rewrite；必须 cache + 归一化 TTFT + 多轮成本 联合看|
|要能指导配置|Case 映射真实反模式：动态 system、工具顺序不稳、用户额度塞 context、截断历史、纯文本拼历史|

#### 核心机制假设
```text
Prompt 从左到右拼接。
从第一个与上次不同的 token 起，其后整段 KV/Prompt Cache 失效。

因此：
  稳定段越长、越靠前、越少改动 → hit 越高、TTFT/成本越优
  任意「无关动态」插在前缀 → miss，即使业务语义不变
```
#### 验证逻辑链
```text
① 建稳定 prefix（system + tools + rules stub，≥4k tokens）
② C0 correct：只 append → 期望高 hit（证明「能 cache」）
③ C1–C5：各改一类变量 → 期望 hit↓（证明「如何失效」）
④ 指标优先级：
     cache_hit / creation  → 机制是否成立
     TTFT_per_1k_input     → 性能是否变差（同量级可比）
     10 轮 est_cost        → 钱是否变多
     raw TTFT / 总时间     → 体感（仅作辅证）
⑤ 与 Cursor 的关系：机制外推到 Rules/MCP/Skills/timestamp；不声称等于线上实现
```

#### 设计必要的 case
1. correct：正向：稳定上下文 + 一致工具顺序 + 结构化消息 → cache 可以持续命中
2. dynamic_system：系统提示每轮加时间戳（@head）→ 整表级失效（最坏）
3. shuffled_tools：工具功能相同、顺序不同 → 仍失效（序列级，非语义级）
4. dynamic_profile：用户额度等无关动态塞进 user 前缀 → 生产反模式；失效弱于或异于 1（位置效应）
5. sliding_window：只留最近 5 条 → 打断连续性；假优化风险（更短≠更省）
6. text_format：历史 flatten 成纯文本 → 破坏约定结构与 cache

#### 如何判定
```text
有 hit 且归一化 TTFT/成本优     → 策略正确（correct）
无 hit 且 TTFT_per_1k / 成本差 → 真实失效（dynamic_* / shuffle / text）
无 hit 但 raw TTFT 更好        → 假优化，看累计成本（典型 sliding_window）
```


### 验证目标
用可控实验验证「prompt 前缀变化是否导致 KV/Prompt Cache 失效」，并用 TTFT、总时间、缓存命中、Token 四层指标做可复现对比。

#### 主目标
证明并量化：

1. 稳定 prefix + append-only → 高 cache hit，TTFT/成本更优。 
2. 系统提示 / 工具顺序 / 用户动态字段 / 历史截断 / 消息格式 破坏前缀一致性 → miss 或 hit 显著下降。 
3. TTFT 下降 ≠ cache 更好；必须以 cache + 归一化 TTFT + 多轮成本联合判定。

#### 非目标
不测生成质量；max_tokens 极小，只测 prefill 侧。


#### 成功标准
| 标准 | 定义 |
|----|----|
|可复现|同配置跑 2 次，hit_rate 相对排序一致|
|可区分|C0 与失效 case 在主指标上可观测分离|
|可解释|每 case 有假设、预期、反例判定|


### 评测指标
| 层 | 指标 |用途|
|----|----|---|
|机制|cache_read / cache_creation / hit_rate=read/input|因果判定（优先）|
|性能|TTFT_ms、TTFT_per_1k_input、total_latency_ms|体感；必须归一化|
|成本|分项 token + est_cost（10 轮累计）|钱|
|控制|round、input/output tokens、prefix_tokens、prompt hash|可比性与可审计|

判定优先级：机制 → 归一化 TTFT → 累计成本 → raw TTFT/总时间。




