1. RequestIDMiddleware 是如何工作的，运行时机是什么
2. app.include_router(chat_router)，app.include_router(task_router) 这两个 router 是什么时候被调度到的，以及如何被调度到的。
chat_router 和 task_router 的关系是什么。
3. build_limiter 为什么要放在 llmGateway 中。llmGateway 提供了哪些功能，他里面保存的属性都是干什么用的
4. @router.post("/v1/chat")
    async def chat(body: ChatRequest, request: Request) 中的 body 和 request 是如何被赋值的 
5. build_session_store 构建的 session 是在什么时候用到的
6. submit_task 提交到队列中的任务是如何执行的
7. task_queue 处理使用 InMemoryTaskQueue，还有其他的实现吗？如果有如何接入，如果没有如何实现并接入
7. 重写查询不能再 06_rag/rag_mvp/src/rag_mvp/pre_query 中实现吗？
8. 给w5和w6中用到的类的字段添加注释
9. 梳理一个整体的架构图出来
10. InMemoryTaskQueue 为什么没有实现 agent_framework/packages/agent_framework/src/agent_framework/queue/base.py 类
11. memory 系统是怎么在 agent 框架中起作用的
12. RAG 系统是怎么在 agent 框架中起作用的
13. 如何加载并使用三方的 mcp 服务, 比如：我要在本项目中使用 GitHub 的 mcp 服务
14. 多轮对话是怎么设计的
15. 如何解决幻觉问题的
16. 上下文是如何管理的
17. token 成本如何计算的
18. functionCall 功能是如何实现的，如何区分是自己的工具，还是 mcp 服务提供的工具，最终在执行工具调用的时候是如何区分这个工具应该去调用 mcp 服务，还是本地执行
19. MCPClient（如何连接 server 的，鉴权握手式如何做的）
20. mcp server 需要提供的 resource 和 prompt 应该如何在项目中使用
21. 能不能输出一个 mcp 和 skill 在本项目中的设计方案 
22. 在本项目中 mcp 工具是如何被注入到上下文中的


### 场景问题
1. RAG 召回不准怎么办
2. Agent 为什么会死循环，如何解决的
3. 工具吊桶失败怎么处理
4. 上下文太长怎么优化
5. 如果我有多个 mcpServer，如何确定当前会话使用到哪些 mcp 工具的。


### 实战题
每个题按照
需求分析-->技术方案-->核心流程--> 异常处理 --> 优化方向，最好能够结合本项目的实现
1. 给你一个业务，设计 Agent 方案
2. RAG 效果不好，怎么排查和优化
3. Agent 工具调用失败，超时，死循环，怎么做异常兜底
