### 读代码从哪入手

看一次请求怎么流出去又怎么回来。

		
|文件| 看文件 | 回答问题 |
|---|-----|------|
|server/routes_chat.py:34``chat()|   它只做解析/投影/错误映射，不拼 prompt、不重试  |   为什么业务判断不许下沉到这一层   |
|llm/client.py:105``LLMGateway|precheck() → chat()/stream()，看 committed 状态机与 _should_retry|重试只在首 token 前、为什么 adapter 不能自己重试（ADR-003）|
|llm/registry.py:78``_build()|alias → adapter 的懒加载、三家差异 flags 在哪合并|换模型为什么"改 yaml 即可、代码零改动"|
|providers/openai_compat.py:436``build_request / :616``parse_response / :201 组装器|三段式拆分：build_request→发→parse_response/assembler|tool_call 增量按 index 累加、四家 usage/错误怎么归一（ADR-002/006/007）|
|llm/types.py + server/sse.py|ToolCall.arguments 恒为 dict、StreamEventType 只有 TOOL_CALL|内部契约长什么样、SSE 帧怎么拼|	

### 如何验证

#### L0 · 离线测试（不联网、不花钱，最先跑

```shell
cd /Users/fanchen/PycharmProjects/LLMs/agent_framework
uv run pytest --basetemp=.pytest_tmp -q        # 现在是 278 passed / 16 skipped
uv run pytest -k a3 --basetemp=.pytest_tmp -q  # 只跑某条 conformance 断言
uv run pytest packages/agent_framework/tests/test_gateway_retry.py -q
```
```text
沙箱里 pytest 建不了系统临时目录，必须带 --basetemp=.pytest_tmp（README 已注明）。16 个 s 是 -m live 用例默认跳过。
```

#### L1 · 本地服务 + curl（不联网、零成本，用 model: fixture 回放夹具）
```shell
uv run agent-server                             # 另开一个终端，:8000
curl -s localhost:8000/health
curl -s -X POST localhost:8000/v1/chat -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}],"model":"fixture"}'      # 非流式
curl -sN -X POST localhost:8000/v1/chat -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}],"model":"fixture","stream":true}'  # SSE 逐帧
curl -s -o /dev/null -w "%{http_code}\n" -X POST localhost:8000/v1/chat \
  -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"x"}],"model":"ghost"}'  # → 400 验证错误映射

```
刚才我实测过：health 正常、非流式拿到规范 JSON、SSE 逐帧可见、日志里有 first_token_ms/cost。

#### L2 · 反向验证

```shell
# 例：往手写基线塞一个真实协议里没有的字段
python3 - <<'PY'
import json,pathlib
p=pathlib.Path("packages/agent_framework/tests/fixtures/_synthetic/openai/text_basic/response.json")
p.write_text(json.dumps({**json.loads(p.read_text()),"hallucinated":{"x":1}},indent=2))
PY
uv run pytest -k a13 --basetemp=.pytest_tmp -q   # 应当 FAIL，并点名 $.hallucinated.x
cp packages/agent_framework/tests/fixtures/openai/text_basic/response.json \
   packages/agent_framework/tests/fixtures/_synthetic/openai/text_basic/response.json  # 还原
uv run pytest -k a13 --basetemp=.pytest_tmp -q   # 应当全绿

```

#### L3 · 真机

```shell
cp .env.example .env.local   # 填 ≥3 家的 key，确认 git status 看不到它
uv run python evals/scripts/record_fixtures.py --apply   # 录真实响应替换手写夹具
uv run pytest --basetemp=.pytest_tmp                      # A13 会清算手写夹具与真实协议的偏差
uv run python evals/scripts/record_fixtures.py --bad-key openai,deepseek,qwen,anthropic --apply
uv run pytest -m live --basetemp=.pytest_tmp -q
uv run python evals/scripts/bench_baseline.py --repeat 3  # 出延迟/成本基线表
```

```shell
# live 用例：只跑 deepseek，其余三家自动 skip
uv run pytest -m live -k deepseek --basetemp=.pytest_tmp -q

# 录 deepseek 夹具：先 dry-run 看会写哪些，确认后加 --apply
uv run python evals/scripts/record_fixtures.py --providers deepseek
uv run python evals/scripts/record_fixtures.py --providers deepseek --apply
uv run pytest --basetemp=.pytest_tmp -q

# 录一次真实鉴权失败体（用无效 key，零成本，验证 401→502）
uv run python evals/scripts/record_fixtures.py --bad-key deepseek --apply

# 延迟/成本基线（只跑 deepseek）
uv run python evals/scripts/bench_baseline.py --models deepseek-flash --repeat 3
```

#### 想"不带 model"也能跑？改一行 yaml
```shell
# packages/agent_framework/src/agent_framework/models.yaml
routes:
  default: { primary: deepseek-flash, fallback: [deepseek-flash] }
  cheap:  { primary: deepseek-flash, fallback: [deepseek-flash] }
```



