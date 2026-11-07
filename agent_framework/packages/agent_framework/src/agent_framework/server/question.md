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