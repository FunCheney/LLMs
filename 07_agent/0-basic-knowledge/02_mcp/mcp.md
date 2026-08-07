### MCP 是什么

MCP（Model Context Protocol）是 工具与 AI 客户端之间的开放协议：

1. 独立的 MCP Server 暴露标准接口：list_tools、call_tool、read_resource 等 
2. MCP Client（如 Cursor、Claude Desktop）连接 Server，拿到工具列表 
3. Client 把这些工具转成模型能用的 Function Calling schema 
4. 模型选中某个 tool 后，Client 通过 MCP 协议转发给 Server 执行

特点：

* 工具独立进程/服务，可复用、可共享 
* 标准传输方式：stdio、HTTP/SSE 
* 统一发现机制：不用每个应用重写接入逻辑 
* 还支持 Resources（读文件/文档）、Prompts 等，不止函数


### 如何实现一个 MCP server

#### 1. 必须实现
这些是任何 MCP Server 都要有的，SDK 大多帮你封装好了，你只需配置：

| 功能               |作用| 是否自己要写        |
|------------------|--|---------------|
| initialize 握手    |	告诉 Client：协议版本、server 名称、支持哪些能力（capabilities）| SDK 自动，你只填元数据 |
| transport（传输层）   |	stdio 或 HTTP/SSE，决定 Client 怎么连你| SDK 提供，你选一种   |
| capabilities 声明  |	声明你支持 tools / resources / prompts 中的哪些| 你配置           |
| 错误处理    |	用 JSON-RPC 标准错误码返回失败| 部分自己写         |

握手阶段 Client 会问「你会啥」，你声明的 capabilities 决定后面它会不会调你对应的接口。

#### 2. 核心能力
##### 1. Tools（工具）—— 最核心，几乎必做
让模型「动手做事」。你要实现两个方法：

|方法	| 作用                                     |
|---|----------------------------|
|list_tools()| 返回工具清单：name + description + inputSchema |
|call_tool(name, arguments)| 收到调用，执行业务，返回结果|                                      

inputSchema 的质量直接决定模型调用是否准确（这是你最该花心思的地方）。

```json
{
  name: "searchFlights",
  description: "Search for available flights",
  inputSchema: {
    type: "object",
    properties: {
      origin: { type: "string", description: "Departure city" },
      destination: { type: "string", description: "Arrival city" },
      date: { type: "string", format: "date", description: "Travel date" }
    },
    required: ["origin", "destination", "date"]
  }
}
```

##### 2. Resources（资源）—— 按需

让模型/用户「读数据」，类似只读文件系统。适合暴露文档、配置、日志等。

|方法	| 作用 |
|---|---|
|list_resources()|列出可读资源（URI 列表）|
|read_resource(uri)|返回某个资源内容|
区别：Tool 是动作（可能有副作用），Resource 是只读数据引用。

##### 3. Prompts（提示模板）—— 按需
给用户提供预设的提示词模板（在 Cursor 里表现为可选的 prompt）。

|方法	| 作用 |
|---|---|
|list_prompts()|列出模板|
|get_prompt(name, args)|返回填充好的消息|

最小可用的 Server 只做 Tools 就够了。

#### 3. 传输方式

|方式	| 场景 | 部署 |
|-|---|---|
|stdio|本地单用户、自己电脑上跑|Cursor 启动子进程，最简单|
|HTTP / Streamable HTTP|远程、团队共享、多用户|部署成服务，走 URL|
|SSE|远程、需要服务端推送|部署成服务|





