Reducer 任然属于 State 的设计。

* Graph 负责描述图的设计
* State 负责描述共享数据
* Schema 负责买书 State 中有哪些字段
* Reducer 负责描述这些字段何时合并更新

### 定义

Reducer 是归约函数，节点一般只返回“局部状态更新”，那这些更新到底怎么和旧状态合并。

对于 State 的某个字段而言，Reducer 可以看成一个是“旧值 + 新增更新 -> 合并新值的函数”。不同字段可以有不同的 Reducer；
如果某个字段没有显式指定 Reducer，LangGraph 默认就按覆盖更新处理，也就是节点返回的新值直接替换这个字段原来的旧值。

