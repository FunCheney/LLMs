# LLMs — 大模型学习仓库

一个按学习路线组织的 **大模型（LLM）全栈学习仓库**，覆盖从 Python 基础工具、机器学习、深度学习，到手写 GPT 模型、RAG 系统、AI Agent，最终落地为一个生产级 Agent 框架的完整进阶路径。

## 🗺️ 学习路线

```
00 基础工具 → 01 机器学习 → 02 深度学习 → 03 强化学习
        ↓
05 手搓 LLM（构建 GPT from scratch）
        ↓
06 RAG 检索增强生成 ──┐
                      ├──→ 07 AI Agent → agent_framework（生产级 Agent 项目）
                      ┘
```

## 📁 仓库结构

| 目录 | 内容 | 形式 |
|---|---|---|
| [`00_basic_tools/`](./00_basic_tools/) | Python / NumPy / Pandas / Matplotlib / PyTorch / Transformers 基础 | Notebook + 代码 |
| [`01_basic_machine_learning/`](./01_basic_machine_learning/) | 线性回归、梯度下降 | 代码 + Notebook |
| [`02_basic_deep_learning/`](./02_basic_deep_learning/) | 深度学习（神经网络、NLP） | 笔记 |
| [`03_baisc_reinforcement_learning/`](./03_baisc_reinforcement_learning/) | 强化学习基础 | 笔记 |
| [`05_LLM/`](./05_LLM/) | **手搓大模型**：分词、注意力机制、GPT 实现、预训练与微调 | 代码 + 配套文章 |
| [`06_rag/`](./06_rag/) | RAG 系统：文档解析、切分、向量化、检索、重排、生成 + `rag_mvp` 项目 | 代码 + 项目 |
| [`07_agent/`](./07_agent/) | AI Agent：Function Calling、MCP、ReAct、LangChain / LangGraph / LlamaIndex | 代码 + 笔记 |
| [`agent_framework/`](./agent_framework/) | 生产级 Agent 框架项目 `growth-agent`（可评测、可观测、可降级） | uv 管理的工程化项目 |

---

## 00 基础工具 `00_basic_tools/`

| 子目录 | 说明 |
|---|---|
| `01_python/` | Python 基础与进阶（列表、列表优化、async 异步） |
| `02_numpy/` | ndarray、NumPy API、ndarray 性能优化 |
| `03_pandas/` | Pandas 数据处理 |
| `04_matplotlib/` | Matplotlib 可视化 |
| `05_pytorch/` | PyTorch 官方教程实践：数据加载 → Tensor → 神经网络 → 自动求导 → 优化器 → 模型保存与加载 |
| `05_pytorch/transformers/` | 手写注意力机制（attention）、多头注意力 |

## 01 机器学习 `01_basic_machine_learning/`

- `01_linear_regression/` — 从零实现线性回归
- `02_gradient_descent/` — 梯度下降原理与实现

## 05 手搓大模型 `05_LLM/`

参考 *Build a Large Language Model (From Scratch)*，从零构建一个 GPT 模型，每个章节配有配套文章（见 [`05_LLM/README.md`](./05_LLM/README.md)）：

| 章节 | 内容 |
|---|---|
| `00_setup` | 环境准备 |
| `01_ch1` | 手写 Tensor、神经网络、Transformer 组件、数据加载 |
| `02_ch2` | 分词器与词嵌入、位置编码 |
| `03_ch3` | 注意力机制：简单自注意力 → 可训练权重 → 因果注意力 → 多头注意力 |
| `04_ch4` | 实现 GPT 模型与文本生成 |
| `05_ch5` | 预训练：训练策略、生成控制、加载开源权重 |
| `06_ch6` | 分类微调 |
| `07_ch7` | 指令微调 |

```bash
cd 05_LLM
pip install -r requirements.txt
```

## 06 RAG `06_rag/`

按「准备数据 → 知识更新 → 知识检索」三大环节组织，路线见 [`06_rag/README.md`](./06_rag/README.md)：

- **数据准备**：PDF / 表格 / 在线数据；多文档格式解析（`02_data_handle/`）
- **知识入库**：文件切分（句子 / 字符数 / 递归）、向量化与嵌入、向量数据库（`03_vector_embedding/`）
- **知识检索**：query → 检索 → 重排序 → 增强 → 生成（`04_search/`）
- **演进阶段**：朴素 RAG → 高级 RAG → 模块化 RAG
- **`rag_mvp/`** — 完整可运行的 RAG MVP 项目（含测试与第三方依赖）

```bash
cd 06_rag
pip install -r requirements.txt
python 00_check_env.py   # 环境检查
python 01_check_model.py # 模型检查
```

## 07 AI Agent `07_agent/`

从理论、设计、实践三个方面研究 AI Agent，学习路线见 [`roadmap.md`](./07_agent/roadmap.md)：

| 模块 | 说明 |
|---|---|
| `0-basic-knowledge/` | Agent 基础：Function Calling、上下文管理、MCP、Plan-and-Solve、Reflection、ReAct、搜索 Agent |
| `1-basic-agent/` · `2-tool-agent/` | 最简 Agent 与工具调用 Agent（单工具 / 多工具） |
| `3-langchain_agent/` · `3-langchain_example/` | LangChain：hello world、模型 IO、Ollama、Prompt、Parser、LCEL、多 Agent 结构 |
| `3-langgraph/` | LangGraph：基础用法、API、email-assistant 实战、进阶模式 |
| `4-llama-index/` | LlamaIndex 实践 |

## agent_framework — 生产级 Agent 项目

Capstone 项目 [`growth-agent`](./agent_framework/README.md)：从「主题」到「深度调研 → SEO 文章 → 多平台文案」的全链路 Agent，目标是**可评测、可观测、可降级**。

```text
agent_framework/
├── packages/agent_framework/   # 框架库：llm / knowledge / memory / skills / planner / eval / resilience
├── apps/growth_agent/          # 产品应用（依赖框架，反向不成立）
├── evals/                      # golden 数据集 / 评测脚本
├── ops/                        # Dockerfile / compose / runbook
├── docs/                       # 架构文档 / ADR / 周任务
└── legacy/temporal_demo/       # 旧 Temporal demo（仅参考）
```

```bash
cd agent_framework
uv sync          # 安装全部依赖
uv run pytest    # 冒烟测试（离线）
uv run ruff check . && uv run mypy
```

---

## 🔧 环境

- Python 3.12+（`agent_framework` 使用 [uv](https://docs.astral.sh/uv/) 管理依赖）
- 各子目录自带 `requirements.txt`，按需安装

## 📌 待补充

- [ ] `02_basic_deep_learning/` 深度学习内容
- [ ] `03_baisc_reinforcement_learning/` 强化学习内容
- [ ] `01_basic_machine_learning/` 补充分类、聚类、无监督学习
