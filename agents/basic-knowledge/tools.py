# pip install duckduckgo-search
# pip install google-search-results
from typing import Dict, Any, List

from ddgs import DDGS

def search(query: str) -> str:
    """
     一个网页搜索引擎工具。
    它使用 DuckDuckGo 来搜索并返回排名前3的结果摘要。
    :param query:
    :return:
    """
    print(f"正在执行真实网页搜索：{query}")

    try:
        with DDGS()as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]

        if not results:
            return f"对不起，没找到 {query} 的信息"

        result_string = []
        for i, result in enumerate(results):
            result_string.append(f"[{i + 1}] {result['title']} \n {result['body']}")

        return "\n\n".join(result_string)
    except Exception as e:
        print(e)
        return f"搜索发生错误：{e}"


class ToolExecutor:
    """
    工具执行器
    """
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(self, name: str, desc: str, func: callable):

        if name in self.tools:
            print(f"警告: 工具 {name} 已经存在，将被覆盖")

        self.tools[name] = {"desc": desc, "func": func}

        print(f"工具 {name} 已被注册, func={func}")

    def getTool(self, name: str) -> callable:

        return self.tools.get(name).get("func", None)

    def getAvailableTools(self) -> str:
        return "\n".join([
            f"- {name}: {info['desc']}"
            for name, info in self.tools.items()
        ])


# --- 工具初始化与使用示例 ---
if __name__ == '__main__':
    # 1. 初始化工具执行器
    toolExecutor = ToolExecutor()

    # 2. 注册我们的实战搜索工具
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    toolExecutor.registerTool("Search", search_description, search)

    # 3. 打印可用的工具
    print("\n--- 可用的工具 ---")
    print(toolExecutor.getAvailableTools())

    # 4. 智能体的Action调用，这次我们问一个实时性的问题
    print("\n--- 执行 Action: Search['英伟达最新的GPU型号是什么'] ---")
    tool_name = "Search"
    tool_input = "英伟达最新的GPU型号是什么"

    tool_function = toolExecutor.getTool(tool_name)
    if tool_function:
        observation = tool_function(tool_input)
        print("--- 观察 (Observation) ---")
        print(observation)
    else:
        print(f"错误：未找到名为 '{tool_name}' 的工具。")