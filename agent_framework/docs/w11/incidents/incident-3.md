# INC-3 · `AUTH_MODE=key` 静默没生效：环境变量读的是一个不存在的名字

- **发现时间**：W10 D5（`ops/deploy_check.sh` 第一版实跑）
- **严重度**：P0（鉴权看起来"配好了"，实际全部匿名放行）
- **影响面**：上线后的所有端点

## 1. 现象

按 `ops/.env.example` 写好 `.env.local`：

```
AUTH_MODE=key
API_KEYS=sk-live-...=alice
```

起服务，不带任何 key 请求 `/v1/tasks`：

```
HTTP/1.1 200 OK
```

**应该是 401。** 没有任何报错、没有 warning，`Settings().auth_mode` 打印出来是 `"off"`。

## 2. 根因

`pydantic-settings` 的 env 名**取自字段名**（`case_insensitive` + 大写化），
而不是取自任何 alias。字段当时叫 `auth`，于是它读的是 `AUTH`；
`.env` 里写的是 `AUTH_MODE` → 读不到 → **静默使用默认值 `off`**。

这是 pydantic-settings 默认行为里最危险的一点：**打错名字不报错**。
`extra="ignore"`（本项目为了容忍前端配置而开的）让它更安静。

一句话：**配置项的名字与实际读取的 key 有两个真相，而默认值把错的那个盖住了。**

## 3. 怎么被发现的：门禁脚本，不是人

`ops/deploy_check.sh` 第一版跑出 **3/5**，第 2、3 项红：

```
[2/5] 未带 key 请求受保护端点 → 期望 401，实际 200   ✗
[3/5] 带错误 key 请求         → 期望 401，实际 200   ✗
```

这两项的判据来自 `ERROR_HTTP` 表（`AUTH_REQUIRED` → 401），而不是脚本里另写一个期望值——
**如果脚本自己写"期望 401"，它和代码就会一起错。**

## 4. 修复

1. 字段改名 `auth_mode`（env 名随之变成 `AUTH_MODE`）；
2. `AUTH_MODE=key` 但 `API_KEYS` 为空时**启动即拒绝**：
   ```python
   raise ValueError("AUTH_MODE=key 但 API_KEYS 为空；启动被拒绝")
   ```
   这比"起得来但全部放行"安全得多——起不来会被人立刻发现，静默放行不会；
3. `.env.example` 与实际字段名一起改，`deploy_check.sh` 复跑 **5/5**。

## 5. 回归用例

| 用例 | 断言 |
|---|---|
| `test_auth.py::test_enabling_auth_without_keys_is_refused` | 开了鉴权却没 key → 抛错，不是默认放行 |
| `test_auth.py::test_missing_key_is_auth_required` | 无 key → 401 + `AUTH_REQUIRED` |
| `test_auth.py::test_endpoint_requires_key_when_enabled` | 受保护端点真的要求 key |
| `test_auth.py::test_health_stays_public` | 探针端点保持公开（否则 K8s 探活全红） |

## 6. 教训

1. **配置项的"名字"也要有测试。** 单测里直接构造 `Settings(auth_mode="key")` 永远过——
   它绕过了 env 读取这一步，而 bug 恰好在那一层。能抓住它的只有"真的用 `.env` 起一个进程"。
2. **默认值必须是最安全的那个。** 这个字段默认 `off` 是对的（开发方便），
   但"显式开了却没生效"必须变成启动失败，而不是继续按默认值跑。
3. **安全开关的验证不能靠读代码。** 读代码时你会相信 `AUTH_MODE` 被读到了；
   只有一次真实的 401/200 能证明。
