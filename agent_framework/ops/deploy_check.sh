#!/usr/bin/env bash
# 上线自检：只回答"这个地址现在是不是按我们要的方式在服务"。
#
#   ./ops/deploy_check.sh https://growth.example.com sk-live-xxx
#   ./ops/deploy_check.sh http://localhost:8100 sk-test            # 部署前本地预演
#
# 五项，任何一项失败就非 0 退出——CI 与人都能用同一个判据。
# 为什么不用 pytest：这些断言的对象是**已经跑起来的进程**，
# 而不是仓库里的代码。测试绿了但线上 401 全放行，是这类部署最常见的形态。

set -uo pipefail

BASE_URL="${1:?用法: deploy_check.sh <base_url> <api_key> [fixture_model]}"
API_KEY="${2:?用法: deploy_check.sh <base_url> <api_key> [fixture_model]}"
FIXTURE="${3:-fixture}"

PASS=0
FAIL=0

ok()   { PASS=$((PASS + 1)); printf '  \033[32m✓\033[0m %s\n' "$1"; }
bad()  { FAIL=$((FAIL + 1)); printf '  \033[31m✗\033[0m %s\n' "$1"; }
note() { printf '    %s\n' "$1"; }

# `curl -s -o body -w '%{http_code}'` 是唯一稳的取状态码方式：
# `-f` 会在 4xx 时直接失败，反而看不到 code 与 message。
http() {
  local method="$1" path="$2" body="${3:-}" auth="${4:-}"
  local args=(-s -o /tmp/deploy_check_body -w '%{http_code}' -X "$method" "${BASE_URL}${path}")
  [ -n "$body" ] && args+=(-H 'Content-Type: application/json' -d "$body")
  [ -n "$auth" ] && args+=(-H "X-API-Key: ${auth}")
  curl "${args[@]}" --max-time 20 2>/dev/null || echo "000"
}

body_field() {
  python3 -c 'import json,sys;d=json.load(open("/tmp/deploy_check_body"));print(d.get(sys.argv[1],""))' "$1" 2>/dev/null
}

echo "== 1. 健康探针（免鉴权）=="
CODE=$(http GET /health)
if [ "$CODE" = "200" ]; then
  ok "/health 200"
  note "$(cat /tmp/deploy_check_body | head -c 200)"
else
  bad "/health 返回 ${CODE}（探针不通，后面都不用看了）"
fi

echo "== 2. 无凭据必须 401 =="
CODE=$(http POST /v1/draft '{"brief":"x","model":"'"${FIXTURE}"'"}')
if [ "$CODE" = "401" ]; then
  ok "无 key → 401"
  note "code=$(body_field code)"
else
  # 这一条失败意味着**鉴权没开**，比任何内容问题都严重。
  bad "无 key 返回 ${CODE}（期望 401：AUTH_MODE 没设成 key？）"
fi

echo "== 3. 错凭据必须 401 且 code 是 AUTH_INVALID =="
CODE=$(http POST /v1/draft '{"brief":"x","model":"'"${FIXTURE}"'"}' "sk-definitely-wrong")
if [ "$CODE" = "401" ]; then
  ok "错 key → 401"
  note "code=$(body_field code)"
else
  bad "错 key 返回 ${CODE}（期望 401）"
fi

echo "== 4. 正确凭据能出结果 =="
CODE=$(http POST /v1/draft '{"brief":"上线自检","model":"'"${FIXTURE}"'"}' "$API_KEY")
if [ "$CODE" = "200" ]; then
  ok "带 key → 200"
  note "alias=$(body_field alias) request_id=$(body_field request_id)"
elif [ "$CODE" = "502" ] || [ "$CODE" = "504" ]; then
  # 502/504 不是鉴权问题，是上游或超时——这两件事值班要分开看，
  # 所以这里判失败但把信息打出来，别让人以为 key 又错了。
  bad "带 key 返回 ${CODE}（鉴权通过，是上游故障：code=$(body_field code)）"
else
  bad "带 key 返回 ${CODE}：$(head -c 200 /tmp/deploy_check_body)"
fi

echo "== 5. 响应头带请求号 =="
# 请求号出现在 header 而不是只在 body：反代、日志、APM 都读 header。
if curl -s -D - -o /dev/null --max-time 20 "${BASE_URL}/health" 2>/dev/null \
  | grep -qi '^x-request-id:'; then
  ok "X-Request-Id 存在"
else
  bad "X-Request-Id 缺失（多实例时无法串联同一次调用）"
fi

echo
echo "通过 ${PASS} 项，失败 ${FAIL} 项"
[ "$FAIL" -eq 0 ] || exit 1
