# INC-1 · `/v1/article` 第一次跑就 500：护栏与自己的封皮互相指认

- **发现时间**：W10 D1（`POST /v1/article` 首次端到端联调）
- **严重度**：P0（功能完全不可用；且它只在"两层护栏组合"时才出现）
- **影响面**：所有走 `GuardrailPipeline` + `SEOPipeline` 的成文请求，100% 失败

## 1. 现象

```
{"code":"ARTICLE_STAGE_FAILED","stage":"input",
 "message":"主题疑似提示注入：boundary_escape(fake_delimiter)"}
```

请求体是**完全正常的**中文主题（`内容营销怎么做`）。单独跑 `SEOPipeline` 不复现，
单独跑 `GuardrailPipeline` 也不复现——**两个都"通过测试"的组件组合起来就挂**。

## 2. 根因

两道护栏各自都做了"隔离不可信文本"这件事，而隔离是加封皮：

```
外层 GuardrailPipeline：<untrusted source="user">内容营销怎么做</untrusted>
内层 SEOPipeline 再检测这段字符串
```

内层的 `fake_delimiter` 规则（防"用户伪造 `</untrusted>` 提前结束隔离"）看到
`</untrusted>` 就报越界——**它看到的是外层自己加的合法封皮**。

一句话：**两层的边界约定不一致**。外层说"我隔离过了"，内层说"我看到封皮就是伪造"。

## 3. 修复

三件事，缺一不可：

1. 新增 `unwrap(text, source=...)`：只剥**期望来源**的封皮；
2. `isolate()` 改成**幂等**：已经隔离过的不再套第二层；
3. `detect()` / `screen()` 在跑规则前先 `unwrap`——**不给规则开豁免**（那是把问题藏起来）。

第 1 条为什么强调"只剥期望来源"：如果能剥任意来源的封皮，那么"伪造
`source="system"` 的封皮"就会变成一种有效的绕过手段——"已经隔离过了"这句话可以被伪造。

## 4. 消融实验：证明"解封皮"就是那个变量

```bash
uv run python evals/scripts/w11_fork_ablation.py
```

| 臂 | 注入拦截率 | 正常放行率 | 误报规则 |
|---|---|---|---|
| A1 不隔离 | 100% | 100% | - |
| **A2 隔离后不解封皮（修复前）** | 100% | **0%** | `fake_delimiter`×6 |
| A3 隔离后解封皮（修复后） | 100% | 100% | - |

三条读数，每条都很重要：

- **A2 的拦截率也是 100%**。所以"红队 10/10 全拦"这种指标**完全发现不了这个 bug**——
  它把正常输入全拦了，拦截率只会更高。**只测攻击样本的防线，无法证伪。**
- **A1 反而正常**。单层护栏不会有这个问题——这就是为什么它躲过了 W10 之前所有的单测。
- 误报的 6 条全部来自同一个规则（`fake_delimiter`），说明**防线曾经只有一个支点在误伤**。

## 5. 回归用例

| 用例 | 断言 |
|---|---|
| `test_injection.py::test_isolate_is_idempotent` | 隔离两次与隔离一次结果相同 |
| `test_injection.py::test_unwrap_only_strips_the_expected_source` | 伪造来源的封皮不剥 |
| `test_guardrail_pipeline.py::test_execute_receives_isolated_text` | 传给 execute 的是隔离后文本 |
| `test_article_route.py::test_isolation_envelope_does_not_leak_into_the_article` | 封皮不流进成文结果 |

## 6. 教训

1. **两条防线各自正确 ≠ 组合正确。** 组合缺陷只能在端到端路径上暴露，
   而端到端路径通常是最晚才跑的那一条。
2. **"协议"必须有唯一一份定义。** 封皮格式被两处代码各自理解，就一定会分叉。
   修法不是让内层认得外层，而是让"剥封皮"变成一个共享函数。
3. **否定式断言才抓得到这类问题。** `assert allowed` 永远绿，
   `assert 正常输入必须通过` 才会红。
