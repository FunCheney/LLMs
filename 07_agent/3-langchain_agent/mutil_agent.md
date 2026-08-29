## 多智能体
多智能体 / A2A 更常见的落点，是把一个大任务拆给多个专业角色协作，而不是单纯为了“多几个 Agent”。例如：

* 软件工程团队式协作：规划 Agent 负责拆需求，代码 Agent 负责实现，测试 / Review Agent 负责运行验证和指出风险，最后由协调者汇总结果。
* 并行代码任务：当多个改动彼此独立时，可以让不同子 Agent 分别处理不同模块，再由 Supervisor 汇总 diff、测试结果和冲突风险。
* 数据分析代理团队：数据理解 Agent 解释指标口径，SQL / Python Agent 查询和计算，图表 Agent 生成可视化，业务解释 Agent 输出结论和建议。


### 多智能体常见的模式

* subAgent（子代理）:
* handoffs （交接）:
* skill（技能）:
* router（路由）:
* custom work（自定义工作流）: 


### 多智能体常见的结构形态

结合官方模式和 LangGraph 语境，可以把多智能体常见形态收敛成下面几类：

* 单智能体（Single Agent）：一个 Agent 负责整条任务链路，适合简单任务和教程起步。 
* Supervisor（主管型）：一个中心主管负责决定调用哪个子 Agent，适合统一入口和集中调度。 
* Handoff（交接型）：当前 Agent 可以把控制权交给别的 Agent，更适合角色切换和会话延续。 
* Router（路由型）：先判断任务属于哪类，再把任务交给某个专门 Agent。 
* Network / Peer-to-peer（网络型）：多个 Agent 更平等地交换信息，没有单一主管，适合研究、协作式问题求解。 
* Hierarchical（层级型）：多层主管和子主管分层拆任务，适合复杂组织结构。
