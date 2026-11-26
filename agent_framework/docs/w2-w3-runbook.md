# W2 / W3 复跑手册

> **给谁看**：想亲手把 W2（运行底座）和 W3（评测体系）跑一遍的人，包括三个月后的自己。
> **两条前提**：不需要任何 API key；不需要联网（`make up` / `make load` 除外）。
> **本文所有输出都是在本仓库实跑贴的**，不是回忆：主体写于 2026-11-23，
> 2026-11-24 因 `models.yaml` 别名迁移（§8.1）整篇复跑过一次，改动处都标了日期。
> 数字会随代码变，命令不会。

W2 的交付物是「能扛并发、能追全链路、能一键起来、能自动证明没坏」的底座；
W3 的交付物是「一条命令说清当前多好、且数字可复现」的评测体系。
所以本文的每一节都对应一句**可以验证的话**，而不是「读某个文件」。

---

## 1. 三分钟：最短路径

```bash
cd agent_framework
uv sync                                    # 首次约 10–40s（已缓存则 <1s）

make smoke                                 # W2 D5：10 条 smoke
make golden                                # W3 D2：golden set 校验
uv run python evals/scripts/w3/run_eval.py --no-write   # W3 D3：54 条批量评测，不落盘
make e2e                                   # W2 D2/D3：过载 503 / 任务端点 / trace 贯穿
```

四条命令各自应当看到（实跑）：

```
10 cases: 10 passed, 0 failed（mode=offline, 1.27s）

总条数 54（离线可执行 50，skipped 4）
难度分布 1:12 2:26 3:16
OK：字段、引用、夹具、难度分布全部通过

54 条：passed 50 / failed 0 / interrupted 0 / skipped 4 → success_rate 100.0%（mode=offline）

[D] 20 并发（单请求 200ms、限额 1）：200=1 503=19 其他=0
[D] Retry-After=1 code=SERVER_BUSY
[E] 日志 4 条，retry_count>0 的 0 条，error_code=set()
全部通过：聊天/任务/过载 503+Retry-After/trace 一致/无重试风暴
```

这四条**不碰网络、不烧钱、不占端口**，是日常改完代码后最划算的一次自检。
剩下的是需要起服务的部分（W2 D1/D2/D3、W3 D4 的真机档）。

---

## 2. 环境与两个本机坑

| 项 | 要求 | 怎么确认 |
|---|---|---|
| Python | 3.12 / 3.13（本文在 3.13.12 上跑） | `python3 --version` |
| 包管理 | `uv` ≥ 0.5（本文 0.12.3） | `uv --version` |
| 端口 | 8000（框架）、8100（产品）、8011/8012（e2e 自检临时占用） | `lsof -i :8000` |
| docker | 只在 `make up` / `make load` 的容器档需要 | `docker info` |
| 密钥 | **不需要**。离线档 `model=fixture`，走 `tests/fixtures/` 回放 | — |

### 坑 1：本机 `curl` 会被 HTTP 代理劫持

```bash
env | grep -i proxy
# HTTP_PROXY=http://127.0.0.1:58902
```

代理在跑时 `curl http://127.0.0.1:8000/health` 会返回
`upstream connect failed: Connection refused (os error 61)` —— 看起来像服务没起来，
实际是请求被送去代理了。**本机端点一律加 `--noproxy '*'`**：

```bash
curl -s --noproxy '*' http://127.0.0.1:8000/health
```

### 坑 2：本机延迟数字可能被环境放大（不总是发生）

2026-11-23 那次，同一份 `run_eval` 在本机每条 case 固定多出 ~30 ms；纯读一个 1 KB 夹具实测
**47 ms/次**（`json.loads(p.read_text())`），而正常机器是 0.1 ms 量级。
`evals/history/2026-W48-w3-eval.md` 已把这一栏标成「环境伪影，勿引用」。

**2026-11-24 复跑同一命令，这个现象没再出现**：54 条 `run_eval` 耗时 2.14s、
`pipeline.latency_p95_ms` 从 30.81 回到 **2.61**，`make gate` 直接通过。
所以它更像"某些沙箱/容器状态下的文件系统放大"，不是稳定伪影。两个后果都要知道：

- 报告里的 `latency p50/p95` 只能用来做**相对比较**（改前 vs 改后同一台机、同一天），不能当基线；
- `make gate` 偶发判失败时（见 §8.2），**先重跑一次 `run_eval` 再判**，别急着改代码。

### 坑 3：`pytest` 再手动加一个 `-q`，汇总行会消失

本仓库 `pyproject.toml` 的 `addopts` 已经是 `-q --strict-markers`。
自己再写 `pytest -q` 就是双重 quiet，输出只剩一排进度点，**看不到 `23 passed` 这一行**，
很容易误判成「跑完了但没结果」。本文所有 pytest 命令都**不带** `-q`。

---

## 3. 命令地图

| 我想验证 | 命令 | 产物 / 看到什么 | 耗时 |
|---|---|---|---|
| 会话幂等、`provider_meta` 不丢 | `uv run pytest -k session` | `23 passed` | ~8s |
| 队列状态机、限流器读 yaml | `uv run pytest -k "queue or concurrency"` | `22 passed` | ~2s |
| 过载 → 503 + `Retry-After`、无重试风暴 | `make e2e` | `[A]`–`[E]` 五行 | ~2s |
| 队列端点能提交能轮询 | 起服务后打 `POST /v1/tasks` | `202` → `succeeded` | ~2s |
| trace 贯穿（HTTP → LLM → 日志） | 起服务后 `curl -D` 看响应头 | `x-trace-id` == 日志 `trace_id` | ~1s |
| smoke 10 条（W2 门禁） | `make smoke` | 10/10 | ~1.3s |
| golden 结构与引用 | `make golden` | 54 条 / 三档分布 | ~1s |
| 54 条评测 + 分层指标 | `run_eval.py --no-write` | 50 passed / 4 skipped | ~6s |
| 同一批 + judge | `run_eval.py --no-write --judge` | judge 均分 1.922 | ~6s |
| Judge 与人工标注一致性 | `make judge-agreement` | kappa 0.3429 / 0.2 | ~2s |
| 回归门禁 | `make gate WEEK=2026-W48` | 该期快照 FAIL（环境伪影），重跑版通过 → §8.2 | ~1s |
| Langfuse 上报 / 降级 | `uv run pytest -k langfuse` | `18 passed`（不联网） | ~8s |
| 压测 100 并发 | `make load`（先起服务） | locust 汇总表 | 30–60s |

`make` 不带参数会打印全部 target（`Makefile` 顶部有注释说明每条命令为什么必须免密钥可跑）。

---

## 4. W2：运行底座（D1–D5）

### 4.1 D1 · 会话层：只存取，不编排

**要验证的三件事**：多轮能存、重复 `request_id` 不翻倍、`provider_meta` 原样回来。

```bash
uv run pytest -k session
# 23 passed, 1478 deselected, 3 warnings in 8.13s
```

想直接看行为，跑这段（实跑输出已贴）：

```python
import asyncio
from agent_framework.session import InMemorySessionStore
from agent_framework.llm.types import Message, MessageRole

async def main():
    store = InMemorySessionStore()
    turn = [Message(role=MessageRole.USER, content="第一轮"),
            Message(role=MessageRole.ASSISTANT, content="回答")]
    print("第一次 append:", await store.append("s1", turn, request_id="req-1"))
    print("重复 append:", await store.append("s1", turn, request_id="req-1"))
    print("条数:", len(await store.get_history("s1")))
    print("另一会话:", len(await store.get_history("s2")))
    await store.aclose()

asyncio.run(main())
```

```
第一次 append: True
重复 append: False
条数: 2
另一会话: 0
```

`append` 的返回值是「这次真的写进去了吗」，不是「成功/失败」。重复返回 `False`
**不是异常**——这是幂等的正常形态，调用方不需要处理。

**Redis 档**（可选，需要本机 redis）：

```bash
docker run -d --rm -p 6379:6379 redis:7
SESSION_BACKEND=redis REDIS_URL=redis://localhost:6379/0 uv run agent-server
```

然后重跑上面那段，把 `InMemorySessionStore()` 换成 `RedisSessionStore(url=...)`。
要点是 `SET session:{sid}:req:{rid} NX EX ttl` 抢标记，抢到才 `RPUSH`；
用 `LRANGE` 扫一遍判断重复在并发下必错（两个请求会同时得出「没有」）。

### 4.2 D2 · 并发闸门 + 任务队列

```bash
uv run pytest -k "queue or concurrency"
# 22 passed, 1479 deselected, 3 warnings in 2.13s
```

**任务端点**（起服务后）：

```bash
uv run agent-server          # 终端 A

# 终端 B
curl -s --noproxy '*' -X POST http://127.0.0.1:8000/v1/tasks \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"说一句话"}],"model":"fixture"}'
# {"task_id":"task_9d96...","status":"queued"}

curl -s --noproxy '*' http://127.0.0.1:8000/v1/tasks/task_9d96...
# {"task_id":"...","status":"succeeded","attempts":1,"result":{...}}
```

`POST` 只做三件事：校验、入队、返回 `task_id`。**长任务不在 HTTP 请求里跑**，
否则 HTTP 超时就成了任务超时，客户端一断开任务就被取消。

**过载必须给退避信号，且不能被重试放大**——这条用 `make e2e` 验，不用手工压：

```
[0] limit_for('fixture')=1 overload_wait_s=0.02
[D] 20 并发（单请求 200ms、限额 1）：200=1 503=19 其他=0
[D] Retry-After=1 code=SERVER_BUSY
[E] 日志 4 条，retry_count>0 的 0 条，error_code=set()
```

`[E]` 那行是重点：如果 `ServerBusyError.retryable` 写成 `True`，过载会被网关重试，
20 并发会变成 60 次上游调用，`[E]` 立刻不再是 0。`make e2e` 的注释里写了为什么必须
**在进程内起服务**（限额靠临时 catalog，subprocess + env 没生效时无法区分
「背压没接上」和「限额没读对」）。

### 4.3 D3 · trace 贯穿

```bash
uv run agent-server &
curl -s --noproxy '*' -D /tmp/hdr.txt -N -X POST http://127.0.0.1:8000/v1/chat \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}],"model":"fixture","stream":true}'
grep -i 'x-trace-id\|x-request-id' /tmp/hdr.txt
```

```
x-request-id: req_825e745546d7
x-trace-id: trace_a3957931df394f9d
```

服务端日志里同一请求应有：

```json
{"request_id":"req_825e745546d7","trace_id":"trace_a3957931df394f9d",
 "span_id":"span_b06fb8a49509","first_token_ms":35.456,"latency_ms":35.652,
 "retry_count":0,"error_code":null,"cache_hit":false,...}
```

判据是**三个 id 能对上**：响应头 `X-Trace-ID` == 日志 `trace_id`，且 `span_id` 非空。
一次 `/v1/chat` 只产出一条 span 是正常的——多 span 出现在 agent 循环里（W9）。

### 4.4 D4 · 容器与压测（需 docker / 网络）

```bash
make up        # compose: agent-server / growth-agent / redis / pgvector
make ps
make logs      # 跟踪 agent-server
```

```bash
# 压测：先确保 8000 在跑（make up 或 uv run agent-server）
make load      # 离线档 20 并发 30s，model=fixture
# 100 并发加强版：
uv run --with locust locust -f evals/scripts/w2/locustfile.py \
  --headless -u 100 -r 25 -t 40s --host http://127.0.0.1:8000
```

**怎么读报告**：`evals/history/2026-W45-w2-load.md` 里同时给了两套口径——

| 口径 | p95 | 能不能当基线 |
|---|---|---|
| locust（客户端） | 1200 ms | 否：夹了进程调度停顿 |
| `RequestLog`（服务端） | 35.5 ms | 否：mock 回放，只能证明埋点对 |

当时有效的结论只有三条：错误率 0/1224、`retry_count` 全 0、994 条 `trace_id` 无重复。
延迟与吞吐在该报告里明确标为「待重跑」。**不要把 mock 回放的 35 ms 当成真实延迟**。

### 4.5 D5 · smoke eval 与门禁

```bash
make smoke                                    # 全跑
uv run python evals/scripts/w2/run_smoke.py --only smoke-07 -v   # 单条排查
```

**反向验证（门禁真的会红吗）**——这一步不做，等于没门禁。不要改仓库文件，
复制一份改坏判据：

```bash
python3 - <<'PY'
import json, pathlib
src = pathlib.Path("evals/golden/smoke_v1.jsonl")
out = []
for ln in src.read_text().splitlines():
    if ln.startswith("{"):
        d = json.loads(ln)
        if d.get("id") == "smoke-01":
            d["expect"]["text_contains"] = ["这段文字一定不会出现在回答里"]
            ln = json.dumps(d, ensure_ascii=False)
    out.append(ln)
pathlib.Path("/tmp/smoke_broken.jsonl").write_text("\n".join(out) + "\n")
PY

uv run python evals/scripts/w2/run_smoke.py --golden /tmp/smoke_broken.jsonl; echo "EXIT=$?"
```

```
10 cases: 9 passed, 1 failed（mode=offline, 1.14s）
EXIT=1
```

两个细节：

- `smoke_v1.jsonl` 第一行是 `//` 注释，runner 会跳过；自己编辑时别把它当成坏行删掉。
- **退出码要用 `; echo $?` 单独取**。写成 `... | tail -8; echo $?` 拿到的是 `tail` 的退出码，
  永远是 0 —— 这正是「CI 是绿的但 case 挂了」的一种成因。

---

## 5. W3：评测体系（D1–D5）

### 5.1 D1 · 指标定义先于代码

要读的是 `docs/w3/metric-spec.md`（指标字典）与 `docs/w3/adr-w3-eval-scope.md`（口径与复现约定）。
能跑的是边界用例：

```bash
uv run pytest -k metrics
# 52 passed
```

要确认的一件事：`success_rate` 的分母里**不含**「上游中断」。被 502/限流打断的 case 记
`interrupted` 单列，否则一次上游抖动会让「模型答得好不好」凭空下降。

### 5.2 D2 · golden set

```bash
make golden
```

```
总条数 54（离线可执行 50，skipped 4）
难度分布 1:12 2:26 3:16
kind 分布 adversarial:4 chat:12 draft:8 error:4 research:10 stream:4 tool_call:12
供应商分布 anthropic:12 deepseek:13 openai:13 qwen:12
OK：字段、引用、夹具、难度分布全部通过
```

三个数字各自的含义：

- **54 / 50 / 4**：4 条 `adversarial` 在离线档**故意 skip**（它们的预期是「拒绝」，
  夹具回放没有真模型可拒），标成 skip 而不是 pass，避免假绿。
- **难度 1:12 2:26 3:16**：3 档要求 ≥10 条，否则 W9 多步任务没有样本。
- **`adversarial:4`**：≥3 条对抗样例（注入 / 要求编造 / 引用不存在来源），预期都是拒绝。

`schema.md` 在 `evals/golden/schema.md`；新增 case 后跑一次 `make golden` 就知道
有没有悬空引用或难度缺失。

### 5.3 D3 · 批量 runner

```bash
# 只看结果，不落盘（推荐日常）
uv run python evals/scripts/w3/run_eval.py --no-write

# 单条 / 区间排查
uv run python evals/scripts/w3/run_eval.py --only gold-c01 gold-e01 -v --no-write

# 带 judge（离线档 = 规则基线，零成本）
uv run python evals/scripts/w3/run_eval.py --no-write --judge
```

输出尾部（实跑）：

```
54 条：passed 50 / failed 0 / interrupted 0 / skipped 4 → success_rate 100.0%（mode=offline）
注意：离线档是夹具回放，success_rate 只说明**执行管道**通，不代表模型质量；quality 块保持 null。
judge 均分 1.922（50 条）
耗时 5.96s
```

`--judge` 还多打一行 `judge: {"calls": 50, "parsed": 50, "parse_failures": 0, ...}` ——
`parse_failures` 非 0 说明 judge 输出格式不稳，这时分数不可信。

**单条失败不中断整批**（Gate 3 #5）：

```bash
uv run python evals/scripts/w3/run_eval.py --no-write --break-case gold-c01
# 54 条：passed 49 / failed 1 / interrupted 0 / skipped 4 → success_rate 98.0%
```

一条红、其余照跑完 —— 这是「一条 502 毁掉整晚评测」的反面。

**要落盘**（会写 `evals/history/`）：去掉 `--no-write`。默认期号是**当前 ISO 周**，
所以直接 `make run-eval` 会覆盖本周已有快照。想留痕又不覆盖：

```bash
uv run python evals/scripts/w3/run_eval.py --week 2026-W48-local
# 产出：evals/history/2026-W48-local-eval.json + -w3-eval.md
```

### 5.4 D4 · Langfuse（缺 key 是正常路径）

**没有账号时**（CI 与本机默认）：日志里应出现且只出现一次

```
INFO langfuse keys missing → NoopExporter（离线档与 CI 的正常路径）
```

主链路不受影响。这条是硬约束：没有 Langfuse 账号的机器必须能跑完整评测。

**有账号时**：

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=https://cloud.langfuse.com      # 自建就改这个
uv run agent-server
# 另开终端打一次 /v1/chat 或 /v1/chat?stream，然后去 Langfuse 的 Traces 里找 trace_id
```

字段名与 `Settings` 同名（pydantic-settings 无前缀，大写即 env 名）。
**隐私边界**：`Span.attributes` 里形似内容的键（`content`/`prompt`/`completion`/`text`/
`messages`/`answer`）一律替换成 `{len, sha256[:8]}`；超过 200 字符的字符串同样哈希。
要全文必须显式 `capture_content=True` 并在 ADR 里写合规依据。

不联网验证这三条（降级 / 隐私 / 失败不冒泡）：

```bash
uv run pytest -k langfuse
# 18 passed（httpx.MockTransport，不打真实网络）
```

### 5.5 D5 · 报告与失败分类

报告模板 `evals/report_template.md`；一次正式跑产出
`evals/history/2026-W48-w3-eval.md`，四块固定内容：

1. 分层指标表（任务 / 工具 / 延迟 / 成本 / 质量）
2. 按难度、tag 的成功率分布
3. 与上周 history 的趋势对比
4. Top 失败 case 分类

Judge 一致性（离线档是规则基线，作为**下限**）：

```bash
make judge-agreement        # 等价于 uv run python evals/scripts/w4/judge_agreement.py
```

```
24 条可用 / 0 条不可用 · 加权 kappa 0.3429 · 名义 kappa 0.2 · spearman 0.4368 · MAE 1.1667
阈值 0.4 → 未通过（judge=heuristic-keyword）
```

**「未通过」在这里是预期结果**：离线 judge 是关键词规则基线，本来就不该过 0.4。
它的价值是给出「换真 judge 后至少要超过这条线」的锚点，而不是一个待修的 bug。

---

## 6. 反向验证清单（门禁会不会红）

只跑绿灯等于没跑。以下三条都可以在 2 分钟内做完，且**不改仓库文件**（或改完能一键还原）。

| 想证明 | 做法 | 期望 |
|---|---|---|
| smoke 判据真的会红 | 复制 jsonl 改坏 `text_contains`，`--golden` 指过去 | `9 passed, 1 failed`，`EXIT=1` |
| 单条失败不中断整批 | `run_eval.py --break-case gold-c01` | `passed 49 / failed 1`，其余照跑 |
| 过载真给 503 且不被重试 | `make e2e` 看 `[D]`/`[E]` | `503=19`、`Retry-After=1`、`retry_count>0` 为 0 |

想验证「改了代码后 gate 真会拦」，用 `--break-case` 之外的方式也可以：
`make gate WEEK=<某一期>` 会拿 `evals/history/<期>-eval.json` 与基线比。

---

## 7. 产物在哪

| 文件 | 谁写的 | 里面有什么 |
|---|---|---|
| `evals/history/<期>-eval.json` | `run_eval.py`（不带 `--no-write`） | metrics / cases / quality 三块，W11 曲线读它 |
| `evals/history/<期>-w3-eval.md` | 同上 | 人读的六节报告（含环境伪影告警） |
| `evals/history/<期>.json` | 手动/合流仪式 | 周快照，含 latency/cost/retrieval 等历史指标 |
| `evals/history/2026-W45-w2-load.md` | 手动（压测后） | p50/p95/错误率/并发数 + 偏差说明 |
| `evals/golden/*.jsonl` | 人写 | case 定义；`smoke_v1` 10 条、`golden_v1` 54 条 |
| `evals/coverage-report.json` | `coverage_report.py`（W12） | 覆盖率，与 W2/W3 无关但同目录 |

`metrics` 字段名以 `evals/history/2026-W44.json` 为准（W2/W3 都对齐它，W11 读它）。

---

## 8. 仓库当前真实状态（2026-11-24）

**结论先说：现在全绿。** `1471 passed, 30 skipped`、`make smoke` 10/10、`make golden` 通过、
`make e2e` 通过、54 条评测 50 passed / 4 skipped、`make gate` 通过。
下面是 11-23 那两处红的历史与修法——**留着是因为「为什么会红」比「红了」更值得记**。

### 8.1 已修复：`make test` 曾 3 failed（模型别名改名只改了一个文件）

11-23 的现象：

```
FAILED packages/agent_framework/tests/test_config.py::test_repository_models_yaml_is_valid
FAILED packages/agent_framework/tests/test_concurrency.py::test_limiter_reads_limits_from_yaml
FAILED packages/agent_framework/tests/test_bench_baseline.py::test_resolve_targets_excludes_mock_by_default
```

起因是一笔提交把 `models.yaml` 的 `qwen3.7-plus` 改成 `qwen3.8-flash` 并设为
`default`/`cheap` 首选，但只改了这一处。全仓库当时还有 **36 个文件、97 处**引用旧别名：

| 位置 | 处数 | 性质 |
|---|---|---|
| `evals/golden/golden_v1.jsonl` | 55 | case 的 `online_model`（在线档按它选模型）+ 1 条题干 |
| `tests/fixtures/qwen/**`、`_synthetic/qwen/**` | 20 | 录制回放的请求/响应载荷 |
| `tests/test_*.py` 8 个文件 | 14 | 断言与 `conftest.PROVIDER_MODEL` |
| `evals/scripts/*.py`、`ops/RUNBOOK.md` | 4 | 命令示例与注释 |
| `docs/**` | 若干 | 历史报告里的模型名 |

所以这不是"改 3 条测试"的事：**同一份 yaml 被 5 类消费者同时依赖**，改一处等于让其余 4 类
各自漂一次。修法（2026-11-24 已做）：

1. 代码/测试/夹具/golden 一次性迁到 `qwen3.8-flash`（36 文件 97 处）；
2. `test_config.py` 里那条「wire 名必须是 `qwen3.7-plus-2026-05-26` 快照」的断言，
   改成断言「wire 名与 alias 一致」——因为 3.8-flash 厂商**没有给带日期的快照 ID**；
3. `models.yaml` 第 34 行那句「Qwen 必须钉日期快照」改成承认例外的写法，并在
   `docs/w1/w1-day4.md` 补了 **ADR-009 附注二**，写明代价：跨周对比必须记模型版本。

**故意没改的**：`evals/history/*.json`（历史快照，改了就是伪造测量）与历史评测报告里的
模型名（那些数字是用 3.7-plus 测的）。另有两处测量注释（`ops/RUNBOOK.md` 的 18.0s/990 tokens、
`context_cost_ab.py` 的思考模式 token 数）标注了「用 3.7-plus 实测、未在 3.8 上重测」，
**没有**跟着改名——否则等于给旧数字换了个新被测对象。

### 8.2 `make gate` 偶发 FAIL：先重跑，再怀疑代码

11-23 的现象：

```
基线：2026-W46（commit afbded2）
当前：2026-W48-eval.json
[FAIL] latency_p95_ms: 上升 1243.98%（允许 20%，且需超过 5 绝对增量）：2.2924 → 30.8094
门禁失败：指标劣化
```

`make gate` 读的是**已落盘的快照**，不是现场重跑——所以它会把 11-23 那天的环境噪声
永久保留在 `2026-W48-eval.json` 里。`error_code`、`retry_count`、`trace_id` 三项当时都是干净的，
只有延迟这一列被放大（原因见 §2 坑 2）。

2026-11-24 实测的两步处置（已验证有效）：

```bash
uv run python evals/scripts/w3/run_eval.py --week 2026-W48-local   # 不覆盖旧快照
uv run python evals/scripts/w4/regression_gate.py --current evals/history/2026-W48-local-eval.json
# 基线：2026-W46（commit afbded2） / 对比了 2 个指标 / 门禁通过
```

新快照的 `pipeline.latency_p95_ms = 2.61`（旧快照 30.81）。**所以 `make gate WEEK=2026-W48`
现在仍会 FAIL，那是历史快照的问题，不是当前代码的问题**——这点最容易误判。
如果重跑后仍是 30 ms 量级，再按 `2026-W48-w3-eval.md` 里的 cProfile 结论查
`sitecustomize.py` 是否又注入了 socket broker。

### 8.3 其他已知边界

- `make up` / `make load` 需要 docker 与网络（拉 locust、拉镜像），本手册未在当前沙箱验证。
- 离线档的 `success_rate = 100%` **不代表模型质量**；`quality` 块保持 `null` 是刻意的。
  W48 报告里 `faithfulness 0.00 / relevancy 0.05` 也是夹具回放的产物，不是结论。

---

## 9. 排错表

| 现象 | 原因 | 处理 |
|---|---|---|
| `curl` 报 `Connection refused (os error 61)` | 请求被 HTTP 代理劫持 | 加 `--noproxy '*'` |
| `Address already in use` | 8000/8011/8012 被上次进程占用 | `lsof -i :8000` 后 kill；e2e 换端口见脚本顶部常量 |
| 日志里 `built adapter alias=fixture provider=mock` | 正常：离线档走夹具 | — |
| `make smoke` 全红且日志里有 `UPSTREAM_AUTH` | fixture 夹具有意注入的失败帧被误当成全局错误 | 只关注汇总行；`smoke-07/08` 就是错误映射用例 |
| `run_eval` 某条 `skipped` | 4 条 adversarial 在离线档设计上跳过 | 正常，别改成 pass |
| `make gate` 延迟项 FAIL | 读的是旧快照，那天本机文件读被放大 | 用 `--week <期>-local` 重跑再 gate，见 §8.2 |
| `pytest` 只输出一排点、没有 `N passed` | 自己又加了一个 `-q`（`addopts` 已带） | 去掉手写的 `-q` |
| `pytest` 收集阶段被 SIGTERM | 在本机沙箱里连续跑多次 `uv run` 会被限流 | 一次跑一个，或合并成一条命令 |
| Langfuse 无 trace | 缺 key 静默降级为 `NoopExporter` | 看日志有没有那行 INFO；有就说明 key 没读进来 |
| `TASK_QUEUE_SIZE` 设很小后提交报 503 | 队列满与限流过载共用 `ServerBusyError` → 503 + `Retry-After` | 正常；退避后重试 |

---

## 10. 清理

```bash
make clean                                  # 删 .pytest_tmp / .pytest_cache / .ruff_cache / .mypy_cache
rm -f /tmp/smoke_broken.jsonl /tmp/hdr.txt  # 反向验证留下的临时文件
docker compose -f ops/docker-compose.yml down   # 起过容器才需要
```

`make clean` 不碰 `.venv` 与 `.env*` —— 这两样重建代价高，且删掉会让人以为「环境坏了」。

---

## 附：本文与 W2/W3 任务书的关系

| 想读「为什么这么设计」 | 想读「怎么跑」 |
|---|---|
| `docs/w2/w2-tasks.md`、`docs/w2/adr-w2-session-boundary.md` | §4 |
| `docs/w3/w3-tasks.md`、`docs/w3/metric-spec.md`、`docs/w3/adr-w3-eval-scope.md`、`docs/w3/adr-w3-trace-backend.md` | §5 |

行号会漂，**函数名不会**：本文按函数名与命令定位，不按行号。
