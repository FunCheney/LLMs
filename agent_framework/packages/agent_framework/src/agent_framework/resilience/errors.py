"""入口层过载错误（re-export）。

为什么类定义不在这里：`llm/errors.py` 的 `ERROR_HTTP` 是"异常 → 对外状态码/code"的
唯一真相源，它必须能看见 `ServerBusyError`。若把类定义放本模块，`llm/errors.py`
就要 import 本模块，而本模块又要 import `LLMError` —— 循环导入在"先 import 谁"
不同时行为不一致，是最难查的一类故障。

所以：**类在 llm/errors.py，语义在本模块**。W2 之后新增入口层异常沿用同一条规则。
"""

from ..llm.errors import ServerBusyError

__all__ = ["ServerBusyError"]
