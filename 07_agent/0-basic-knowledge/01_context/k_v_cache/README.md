### 什么是 KV Cache

大模型生成文本时，每处理一个 token，都要做 Self-Attention：当前 token 需要「看见」前面所有 token 的信息。

Attention 会为每个 token 算出两组向量：

K（Key）：像「索引标签」——「我是什么、怎么被检索」
V（Value）：像「内容本身」——「我的实际语义信息」
每来一个新 token，模型都要对之前所有 token 重新算一遍 K/V。对话越长，计算量越大（大致随长度平方增长）。

KV Cache 的做法是：已经处理过的 token，其 K/V 算一次就存起来；生成下一个 token 时直接复用，不必重算。这是 LLM 推理加速的核心优化之一。

可以把它理解成：

```text
第 1 轮：处理 [系统提示词 + 工具定义 + 用户消息]
         → 算出并缓存整段 prefix 的 K/V

第 2 轮：在同样 prefix 后面追加 [助手回复 + 新用户消息]
         → prefix 部分的 K/V 直接命中 cache，只算新增部分
```