from agents.HelloAgentsLLM import HelloAgentsLLM
from agents.tools import ToolExecutor
import re
from agents.tools import search

# ReAct 提示词模板
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下：
{tools}

请严格按照以下格式进行回应：

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一：
- `{{tool_name}}[{{tool_input}}]`：调用一个可用工具。
- `Finish[最终答案]`：当你认为已经获得最终答案时。

现在，请开始解决以下问题：
Question: {question}
History: {history}
"""

class RecAtAgent:

    def __init__(self, llm: HelloAgentsLLM, tool_exec: ToolExecutor, max_steps: int=5):
        self.llm = llm
        self.tool_exec = tool_exec
        self.max_steps = max_steps
        self.history = []

    def run(self, question: str):

        self.history = []
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"----- 第 {current_step} 执行-----")

            # 1. 格式化提示词
            tools_desc = self.tool_exec.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str,
            )

            # 2. 调用 LLM 进行思考
            messages = [{"role": "user", "content": prompt}]

            response_txt = self.llm.think(messages)
            # 3. 解析 LLM 的输出
            thought, action = self._parse_output(response_txt)

            if thought:
                print(f'LLM thought: {thought}')

            if not action:
                print(f'LLM not action break')
                break

            # 4. 执行 action
            if action.startswith("Finish"):
                final_answer = re.match(r"Finish\[(.*)]", action).group(1)
                print(f'LLM final answer: {final_answer}')
                return final_answer

            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                continue

            print(f'LLM output action: {tool_name} {tool_input}')

            tool_func = self.tool_exec.getTool(tool_name)
            if not tool_func:
                observation = f"错误未找到名为【{tool_name}】的工具"
            else:
                # 调用真实工具
                observation = tool_func(tool_input)

            print(f'LLM observation: {observation}')

            self.history.append(f"Action: {action}")
            self.history.append(f"Observation:{observation}")

        return None


    def _parse_output(self, text: str):
        """
        解析 LLM 的输出，提取 thought 和 action
        :param text:
        :return:
        """
        thought_match = re.search(r"Thought: (.*)", text)
        action_match = re.search(r"Action: (.*)", text)

        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """解析Action字符串，提取工具名称和输入。"""
        match = re.match(r"(\w+)\[(.*)\]", action_text)
        if match:
            return match.group(1), match.group(2)
        return None, None



if __name__ == '__main__':
    BASE_URL = "https://api-inference.modelscope.cn/v1/"
    MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
    llm = HelloAgentsLLM(MODEL_ID, BASE_URL, None, 60)
    tool_executor = ToolExecutor()
    search_desc = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.registerTool("Search", search_desc, search)
    agent = RecAtAgent(llm, tool_executor)
    question = "华为最新的手机是哪一款？它的主要卖点是什么？"
    agent.run(question)




















