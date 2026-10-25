# legacy / Temporal 示例（仅参考）

这里保存重构前的 FastAPI + Temporal 示例代码，**不参与 workspace 构建、不被打包、不会被 pytest 收集**。

用途：W2 长任务（深度调研等分钟级流程）需要重新引入 Temporal 时，可直接参考以下文件的写法：

| 文件 | 参考价值 |
|---|---|
| `workflows.py` | `@workflow.defn` 定义与 activity 调度 |
| `activities.py` | `@activity.defn` 幂等/重试边界 |
| `worker.py` | Worker 启动、task queue 绑定 |
| `web.py` | 旧 `/v1/run` 启动工作流、SSE 假流式写法（**反面参考**：真实流式必须走直连） |

注意：

- 依赖 `temporalio` 已从 workspace 移除，W2 需要时再加回 `packages/agent_framework`。
- 原 `compose.yaml` 仍保留在仓库根目录，W2 迁移至 `ops/`。
- 架构决策：**流式对话走直连，分钟级长任务才走工作流**（见 `docs/architecture.md` 第 7 节）。
