# W12 执行清单（框架提取、文档与面试 · P0，收官周）

> **角色约定**：沿用 `docs/w1/w1-tasks.md` 第 3 行——架构、骨架、验收标准由我给，代码由你写，我不代写。
> **终局目标（ROADEMAP 开头）**：12 周结束交付**一个**生产级仓库 `growth-agent`，具备生产可靠性 / 评测设计 / 多智能体协调三大稀缺能力的**可验证证据**。评审不看功能数量，看证据。
> 本周不堆新功能，只做三件事：**提取框架 → 补齐文档与证据包 → 对着 JD 讲清楚**。

## 承接面：W1–W11 已交付（你要打包的全部资产）

| 资产 | 位置 | W12 的动作 |
|---|---|---|
| 框架包 `agent_framework`（llm/session/resilience/retrieval/knowledge/memory/tracing…） | `packages/` | 提取公共接口、依赖注入、脚手架 |
| 产品 `growth_agent`（调研→SEO→文案 pipeline + MCP 工具） | `apps/` | 作为 dogfood 证据，证明框架被真实使用 |
| `evals/history` 12 周数据点 + golden + 门禁 | `evals/` | 画 12 周指标趋势图（证据包第 2 件） |
| 3 篇根因案例（W4）+ ≥5 篇线上 incident（W11） | `docs/w4/`、`docs/w11/` | 面试"最难的三个问题"素材 |
| 单/多 Agent 对比数据（W9） | `docs/w9/single-vs-multi-ab.md` | 稀缺能力③的证据 |
| 压测/混沌/安全红队三份报告（W2/W11/W10） | `evals/history/`、`docs/w11/`、`docs/w10/` | 证据包第 5 件 |
| ADR-001~012 + 各周 ADR | `docs/architecture.md` + `docs/wN/` | 拆成独立文件，≥5 篇（DoD 文档项） |
| DoD 九条自评（W11） | `docs/w11/dod-self-assessment.md` | 逐条补齐缺口或标注 |
| 真实上线 + runbook（W10/W11） | `ops/` | 证据包第 7 件 |

> W12 检查点（ROADEMAP）：**第三方场景可用脚手架 1 天内跑通**；能对着两份 JD 十条职责逐条讲清架构与量化结果。
> ⚠️ W1 `w1-tasks.md` T1.2 提醒：本项目位于父仓库 `PycharmProjects/LLMs` 内（monorepo）。**做作品集前建议 W12 单独建一个干净仓库，只保留本目录历史**——本周 Day2 处理。

**本周预算**：5 天 × 2–3h ≈ 12–15h。

---

## 0. 本周骨架总览

| Day | 新增模块（落点） | 承接资产 | 推进的 DoD/证据包 |
|---|---|---|---|
| D1 | 提取框架公共接口 + 依赖注入 | 全部框架包 | 工程（可复用） |
| D2 | `agent create` 脚手架 + 单测补全 + 本地打包 + 独立仓库 | 框架包 | 检查点"1 天跑通"、证据包1 |
| D3 | 架构图 + ADR≥5 独立文件 + README（指标/截图/demo） | 12 周记录 | DoD 文档项、证据包8 |
| D4 | 前沿调研笔记 + 可落地评估 | W9 A2A 笔记 | 文档、面试深度 |
| D5 | 模拟面试：按两份 JD 十条职责逐条讲 | 全部证据 | 收官验收 |

**唯一主线**：把"我做了一个项目"变成"我能证明这个项目生产级，且能对着岗位讲清楚每一步的量化取舍"。**证据 > 功能，量化 > 形容**。

---

## Gate 0 · 开工前（30 min）

- [ ] 复核 W11 的 DoD 九条自评表，列出仍标注"缺口"的项（本周能补的补，补不了如实写）
- [ ] 导出 `evals/history` 全部周数据点，确认 12 周曲线连续（断了要说明）
- [ ] 重读两份 JD（`ROADEMAP.md` 的来源 `text.md` 等），列出**十条职责**清单
- [ ] `make eval` 全绿、门禁生效；确认线上服务仍可访问
- [ ] 决定作品集仓库方案：独立干净仓库 vs 保留 monorepo 历史

---

## Day 1（3h）· 提取框架公共接口 + 依赖注入

**目标**：从产品中提取**只被真实使用过的抽象**（ROADEMAP §8 反过度设计纪律），加依赖注入。

### 你要做的事
- [ ] 盘点 `packages/agent_framework/` 里**被 `apps/` 真实 import 的符号**，列清单
- [ ] 删除/合并"从没被产品调用过"的抽象（宁可 W12 再提取，也不要留通用空壳）
- [ ] 定义框架公开入口 `__init__.py`（稳定的公共 API 面）
- [ ] 依赖注入：gateway/store/retriever 等通过构造参数注入，不再模块级全局
- [ ] `docs/w12/framework-api.md`：公共接口清单 + 依赖图
- [ ] 单测：公共 API 稳定性、DI 后可用

### 引导框架

```python
# packages/agent_framework/src/agent_framework/__init__.py —— 只暴露稳定公共面
from .bootstrap import build_gateway
from .llm.client import LLMGateway, ChatCall, ChatOutcome
from .llm.types import Message, ToolSpec, ToolCall, Usage
from .config import Settings, load_catalog
# ... 只列被 apps/ 真实使用 + 愿意承诺稳定的

__all__ = [...]
```

**算法步骤（今天最容易写错的地方）**
1. **用 import 扫描找"真实使用过"的符号**：`grep -rn "from agent_framework" apps/`——只提取这些，其余标记 internal
2. 依赖注入：把 `create_app(gateway=None)` 这类模式（W1 已有）推广到 store/retriever；**别用模块级全局单例**（W11 灰度/测试都吃过这亏）
3. 公共 API 一旦进 `__all__` 就要承诺稳定；不确定的先留 internal，别为"看起来完整"乱承诺
4. 提取后 `apps/` 只依赖公共入口（ADR-004 单向依赖），`grep` 验证无 `llm.providers` 泄漏
5. **每删一个抽象都要确认没被用**（否则 W12 提取把线上功能删没了）

**自检**
- [ ] `grep -rn "from agent_framework" apps/` 只命中 `__init__.py` 暴露的公共入口
- [ ] DI 改造后 `make eval` 全绿、端到端可跑
- [ ] `pytest` 覆盖公共 API 的稳定性断言

### 🔍 交给我的锚点
framework-api.md、import 扫描结果、DI 改造 diff。我会看：**是否只提取真实使用过的**、DI 后有无全局单例、公共 API 边界是否克制。

---

## Day 2（3h）· `agent create` 脚手架 + 单测补全 + 打包 + 独立仓库

**目标**：第三方场景用脚手架 **1 天内跑通**（ROADEMAP 检查点）；建立干净的作品集仓库。

### 你要做的事
- [ ] `packages/agent_framework/src/agent_framework/cli.py`：`agent create <name>` 生成最小可跑项目
- [ ] 脚手架模板：最小 `models.yaml` + 一个 `create_app` + 一个示例端点 + README
- [ ] 单测补全：核心模块覆盖率 ≥70%（DoD 工程项）
- [ ] 本地打包：`uv build` / `pip install` 验证可安装
- [ ] **独立干净仓库**：从 monorepo 拆出，只保留本目录 12 周 commit 历史
- [ ] `docs/w12/scaffold-guide.md`：脚手架 1 天跑通演示

### 引导框架

```python
# cli.py 骨架
def create_project(name: str, target: Path) -> None:
    """生成最小可跑项目：能 import 框架、能起 /health、能发一条 chat。"""
    ...
```

**算法步骤（今天最容易写错的地方）**
1. 脚手架生成的**必须真能跑**（装依赖 → 起服务 → 发一条 chat），不是空目录——检查点是"1 天内跑通"不是"生成文件"
2. 覆盖率 ≥70%：先测核心路径（client 编排、错误映射、RAG 溯源、guardrail），别为凑数测 getter/setter
3. `uv build` 产物能被**另一个干净 venv** `pip install` 并 import——验证打包不是自嗨
4. 独立仓库：用 `git filter-repo`/subtree 保留本目录历史，**提交前确认没把 `.env.local`/密钥带过去**（W1 Gate 0 的老规矩）
5. 脚手架演示录一段命令输出（W12 D3 demo 素材）

**自检**
- [ ] 全新目录跑 `agent create demo` → `uv sync` → `/health` → 一条 chat 成功
- [ ] 覆盖率报告 ≥70%（核心模块）
- [ ] 干净 venv 能 `pip install` 打包产物
- [ ] 新仓库 `git log` 有连续 12 周 commit，且 `git status` 无密钥文件

### 🔍 交给我的锚点
脚手架演示输出、覆盖率报告、新仓库链接。我会看：脚手架是否真能跑、覆盖率是否核心路径、新仓库有没有带出密钥。

---

## Day 3（3h）· 架构图 + ADR≥5 + README + demo 视频

**目标**：补齐 DoD 文档项（README + 架构图 + ADR≥5 + runbook + 12 周趋势图），产出**证据包第 8 件**。

### 你要做的事
- [ ] `docs/architecture-diagram.md`（或图片）：五层架构 + 依赖方向（ADR-004）
- [ ] 把 ADR-001~012 拆成 `docs/adr/ADR-0NN.md` 独立文件（≥5 篇，DoD 要求）
- [ ] README 重写：一句话目标 + 12 周指标 + 截图 + 快速开始 + 证据包索引
- [ ] **12 周指标趋势图**：成功率 / p95 / 单任务成本 / judge 分（从 `evals/history` 生成）
- [ ] 5 分钟 demo 视频：主题 → 调研 → SEO → 3 平台文案 + trace 回放
- [ ] `docs/w12/evidence-index.md`：证据包 8 件的索引链接

### 引导框架

**证据包 8 件索引（逐条给链接）**
| # | 证据 | 位置 |
|---|---|---|
| 1 | GitHub 仓库（单产品+框架+评测+ops，12 周 commit） | 独立仓库 |
| 2 | 12 周指标趋势图 | `docs/w12/trends.png` + history json |
| 3 | `evals/history` 原始数据 | `evals/history/` |
| 4 | 3 篇根因归因案例 | `docs/w4/cases/` |
| 5 | 压测+混沌+安全红队三报告 | 各周文档 |
| 6 | 单/多 Agent 对比实验 | `docs/w9/single-vs-multi-ab.md` |
| 7 | 真实上线记录（≥5 条修复）+ runbook | `docs/w11/incidents/`、`ops/runbook.md` |
| 8 | 架构图 + ADR + demo 视频 | `docs/architecture-diagram.md`、`docs/adr/`、视频 |

**算法步骤**
1. 趋势图**从 history json 自动生成**（脚本落 `evals/scripts/plot_trends.py`），别手画假数据——曲线断了如实标注原因
2. ADR 拆分保持编号连续（001~012 + 各周新增），每篇有"决策/理由/代价/状态"
3. README 的指标**用真实数字**（引用 history），不写"显著提升"这种没有证据的形容词
4. demo 视频重点演示"全程有 trace"和"多智能体调研"两大稀缺能力，不是功能流水账
5. `evidence-index.md` 让评审/面试官能一键定位每件证据

**自检**
- [ ] `plot_trends.py` 从 history 生成四张趋势图，数据点连续
- [ ] `docs/adr/` ≥5 篇独立文件，编号无重复
- [ ] README 每个数字都能在 history 找到出处
- [ ] evidence-index 8 条链接全部可达

### 🔍 交给我的锚点
趋势图、README、ADR 目录、evidence-index。我会看：趋势图是否来自真实数据、README 数字是否可溯源、8 件证据是否齐全可达。

---

## Day 4（2h）· 前沿调研笔记 + 可落地评估

**目标**：写前沿笔记（MCP/A2A/AG-UI、多智能体、Computer Use、上下文工程），并给出"可落地评估"——展示判断力而非追热点。

### 你要做的事
- [ ] `docs/w12/frontier-notes.md`：各前沿方向的现状 + 与本项目的关系
- [ ] 每个方向给"是否值得本项目落地 + 理由 + 成本估算"
- [ ] 与 ROADEMAP §7 的 P3 决策呼应（哪些深入档解锁了/没解锁，为什么）
- [ ] Computer Use 按 §7 默认档：**只写笔记，不实现**

### 引导框架

**每个方向的评估模板**
```
## 方向（如 A2A / AG-UI / Computer Use / 上下文工程）
- 现状：一句话 + 关键规范/版本
- 与本项目的关系：已在 W? 做过默认档 / 尚未涉及
- 是否值得落地：是/否 + 理由（对齐目标岗位 JD）
- 成本估算：人天 + 依赖风险（引用 ROADEMAP §7 ROI 排序）
```

**算法步骤**
1. 结论优先，别贴一堆链接不写判断（呼应 W1 Day1 矩阵"必须写结论"的要求）
2. 明确回答"为什么本项目 12 周内没深入 X"——用 §7 的 ROI/解锁规则支撑，不是"没时间"
3. 前沿调研要能支撑面试追问："你了解 XX 吗？为什么没做？"
4. Computer Use 明确标注"通用岗性价比低，只进笔记"（§7 默认档）

**自检**
- [ ] 每个方向都有"是否值得落地"的明确结论
- [ ] 与 §7 的 P3 决策一致（解锁了什么/为什么没解锁）
- [ ] 能口头回答"为什么没做 X"

### 🔍 交给我的锚点
frontier-notes.md。我会看：有没有明确判断（不是资料堆砌）、与 §7 决策是否一致、能否支撑面试追问。

---

## Day 5（2.5h）· 模拟面试：按两份 JD 十条职责逐条讲

**目标**：对着两份 JD 的**十条职责**，每条准备 10 分钟讲述（架构 + 量化结果 + 取舍），完成收官验收。

### 你要做的事
- [ ] `docs/w12/interview-prep.md`：十条职责逐条的讲述提纲
- [ ] 每条挂对应的证据包链接 + 一个量化数字 + 一个"最难的问题/取舍"
- [ ] 自测：随机抽一条，10 分钟讲清（录音/对着人讲一遍）
- [ ] 最终 DoD 九条自评复核（对照 W11 表，补最后的缺口或标注）
- [ ] 周五合流仪式：最后一次合 main + 写 `evals/history/YYYY-Www.json` + `docs/log.md` 收官条目

### 引导框架

**每条职责的讲述提纲模板**
```
## JD 职责 N：<原文>
- 我的对应实现：哪个模块/哪周
- 量化结果：一个真实数字（history/趋势图/报告）
- 架构取舍：为什么这样做（引用 ADR），替代方案是什么
- 最难的问题：一个 incident 或反直觉发现（如 W1 日志静默丢弃、W9 多 Agent 预算每跳重置）
- 可演示：demo 视频的哪一段 / 哪条命令
```

**算法步骤**
1. 十条职责**逐条**覆盖两份 JD（Agent 平台/框架 JD + 增长营销 Agent JD），别只挑好讲的
2. 每条必须有**一个数字**（成功率/p95/成本/judge/Recall 提升/拦截数）——没有数字就回去补度量
3. "最难的问题"优先用 W1–W11 的真实 incident（日志静默丢弃、注入间接注入、多 Agent 预算重置等），比编造的更有说服力
4. 收官自评：DoD 九条最终状态（达标/部分/缺口）如实写进 `docs/log.md`
5. 合流：这是第 12 个数据点，曲线闭合；`docs/log.md` 写一段 12 周总结

### 🔍 Gate 12 · 收官验收（全绿才算 12 周完成）

| # | 验收项 | 通过标准 |
|---|---|---|
| 1 | 框架可用 | 公共 API 只含真实使用的抽象，DI 无全局单例 |
| 2 | 脚手架 | 第三方 1 天内跑通（实跑验证过） |
| 3 | 覆盖率 | 核心模块 ≥70% |
| 4 | 独立仓库 | 干净历史、无密钥、12 周 commit 连续 |
| 5 | 文档 | 架构图 + ADR≥5 + README + runbook + 趋势图 |
| 6 | 证据包 | 8 件齐全且索引可达 |
| 7 | 趋势图 | 12 周数据点连续，来自真实 history |
| 8 | DoD 自评 | 九条逐条达标/缺口 + 证据链接 |
| 9 | 面试准备 | 十条职责逐条有数字 + 取舍 + 最难问题 |
| 10 | 合流 | 第 12 个数据点闭合，`docs/log.md` 收官 |

---

## 本周最容易返工的 6 处

1. **提取了没被产品用过的抽象** → 违反 §8 反过度设计纪律，面试被问"这个谁在用"答不上
2. **脚手架生成但跑不通** → 检查点直接挂；必须实跑验证
3. **趋势图手画假数据 / 曲线断了不说明** → 证据包第 2 件失效，评审一眼看穿
4. **README 写"显著提升"没有数字** → 违反"证据 > 形容"，面试减分
5. **新仓库带出 `.env.local`/密钥** → 安全事故，W1 Gate 0 老规矩
6. **面试准备只讲功能不讲取舍** → "最难的三个问题"和 ADR 才是区分度所在

---

## 产出物位置

| 产出 | 位置 |
|---|---|
| 框架 API/脚手架 | `src/agent_framework/{__init__.py,cli.py}`、`docs/w12/{framework-api,scaffold-guide}.md` |
| 文档/证据 | `docs/architecture-diagram.md`、`docs/adr/ADR-0NN.md`、README、`docs/w12/{evidence-index,trends.png}` |
| 趋势脚本 | `evals/scripts/plot_trends.py` |
| 前沿/面试 | `docs/w12/{frontier-notes,interview-prep}.md` |
| DoD 复核 | `docs/w11/dod-self-assessment.md`（最终版） |
| 收官记录 | `evals/history/YYYY-Www.json`（第 12 点）、`docs/log.md`（收官条目） |

---

## P3 挂载点（W12 末 = 最终决策点，见 ROADEMAP §7）

- 若 W11 末解锁了**前端深入档**，本周完成并放进 demo；否则默认档单页已在
- **Computer Use / 自研向量库 / LoRA 微调 / 高并发规模**：除非明确投对应岗位（基础设施/算法/通用 Agent/高并发），否则**只写笔记不实现**——ROI 排序里排在最后
- 三个决策点（W4/W8/W12 末）的最终结果写进 `docs/log.md`：解锁了什么、为什么、余量如何
- **收官心法**：12 周后你手里的不是"一个能跑的 demo"，是"一个能用 12 周数据证明生产级、且能对着岗位逐条讲清取舍的项目"——这才是雇主公认稀缺能力的兑现
