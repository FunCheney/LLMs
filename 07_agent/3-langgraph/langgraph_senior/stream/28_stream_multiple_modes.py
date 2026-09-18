"""
【案例】多模式流式传输：同一图依次演示 values、updates、列表组合 [values, updates]、以及 debug 模式。

知识点速览：
- stream_mode 为列表时，每次迭代得到 (mode, chunk) 元组，便于前端按类型分别处理。
- `values` 看“全貌”，`updates` 看“增量”；`debug` 输出更细，适合调试，不适合直接当业务输出。
- 这个案例的核心价值是帮你建立“同一张图可以同时暴露多种观察视角”，而不是背住某个模式名。
- 节点函数返回的字典仍按 State 的 Reducer 合并；本例字段未显式 Annotated，默认就是覆盖更新。
"""
from typing import TypedDict

from langgraph.constants import START, END
from langgraph.graph import StateGraph


class DiliState(TypedDict):
    question: str
    answer: str
    confidence: float # 置信度分数
    steps: list

def think(state: DiliState):
    """ 思考节点，模拟多步推理，写入 steps"""
    question = state['question']
    if "天气" in question:
        answer = "天气晴朗"
        confidence = 0.9
    elif "时间" in question:
        answer = "现在是上午 10 点"
        confidence = 0.8
    else:
        answer = "这是一个很好的问题"
        confidence = 0.5

    return {
        "answer": answer,
        "confidence": confidence,
    }

def respond(state: DiliState) -> DiliState:
    """回应节点：根据关键词生成答案与置信度。"""
    question = state["question"]
    if "天气" in question:
        answer = "今天天气晴朗"
        confidence = 0.9
    elif "时间" in question:
        answer = "现在是上午10点"
        confidence = 0.8
    else:
        answer = "这是一个很好的问题"
        confidence = 0.7

    return {
        "answer": answer,
        "confidence": confidence,
    }

def reflect(state: DiliState):
    """ 反思节点：在 steps 上追加校验与结论"""
    answer = state['answer']
    confidence = state['confidence']
    steps = state.get('steps', [])

    steps.append(f"验证答案：{answer}")
    steps.append(f"置信度评估：{confidence}")

    if confidence > 0.8:
        conclusion = "高置信度答案"
    elif confidence > 0.5:
        conclusion = "中等置信度答案"
    else:
        conclusion = "低置信度答案"

    steps.append(f"结论: {conclusion}")

    return {"steps": steps}


def main():
    builder = StateGraph(DiliState)
    builder.add_node("think", think)
    builder.add_node("respond", respond)
    builder.add_node("reflect", reflect)

    builder.add_edge(START, "think")
    builder.add_edge("think", "respond")
    builder.add_edge("respond", "reflect")
    builder.add_edge("reflect", END)

    graph = builder.compile()

    print("=== LangGraph 多模式流式传输演示 ===\n")

    input_state = {
        "question": "今天天气怎么样?",
        "answer": "",
        "confidence": 0.0,
        "steps": [],
    }

    print("--- 1. 使用 stream_mode='values' 模式 ---")
    print("显示每一步执行后的完整状态:")
    for chunk in graph.stream(input_state, stream_mode="values"):
        print(f"  {chunk}")

    print("\n" + "=" * 60 + "\n")

    print("--- 2. 使用 stream_mode='updates' 模式 ---")
    print("只显示每一步的状态更新:")
    for chunk in graph.stream(input_state, stream_mode="updates"):
        print(f"  {chunk}")

    print("\n" + "=" * 60 + "\n")

    print("--- 3. 同时使用 stream_mode=[values, updates] 多种流模式 ---")
    print("同时显示完整状态和状态更新:")
    for mode, chunk in graph.stream(input_state, stream_mode=["values", "updates"]):
        print(f"  [{mode}]: {chunk}")

    print("\n" + "=" * 60 + "\n")

    print("--- 4. 使用 debug 模式 ---")
    print("显示详细的调试信息:")
    try:
        for chunk in graph.stream(input_state, stream_mode="debug"):
            print(f"  {chunk}")
    except Exception as e:
        print(f"  Debug模式可能需要特殊配置: {e}")


if __name__ == "__main__":
    main()

