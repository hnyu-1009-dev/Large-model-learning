"""
LangGraph 最小可运行示例。

这个文件对应文档里的“快速入门”和 Graph API 的第一部分，重点说明：

1. State 是什么。
2. Node 是什么。
3. Edge 是什么。
4. compile() 和 invoke() 分别做什么。
5. 图为什么适合描述有状态工作流。

建议你先完整读完这个文件，再运行一次。
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class HelloState(TypedDict):
    """
    图中的共享状态。

    这就是 LangGraph 里最核心的“数据载体”：
    每个节点都读取当前状态的一部分，并返回一个“局部更新”。
    LangGraph 会把这些局部更新合并回总状态。
    """

    user_input: str
    normalized_input: str
    answer: str


def normalize_user_input(state: HelloState) -> HelloState:
    """
    第一个节点：清洗输入。

    节点函数的基本形态非常简单：
    - 入参：当前状态
    - 返回：一个 dict，表示这一步对状态的更新

    注意：
    这里没有必要把整个状态都重新返回，只返回需要更新的字段即可。
    """

    text = state["user_input"].strip().lower()
    return {"normalized_input": text}


def reply_user(state: HelloState) -> HelloState:
    """
    第二个节点：根据上一步结果生成回复。

    这一步展示了“状态在节点之间流动”的感觉：
    - 它依赖上一个节点写入的 normalized_input
    - 再把 answer 写回共享状态
    """

    normalized_input = state["normalized_input"]

    if "langgraph" in normalized_input:
        answer = "LangGraph is a graph-based orchestration framework for stateful agents."
    else:
        answer = f"You said: {state['user_input']}"

    return {"answer": answer}


def build_graph():
    """
    构建图的标准步骤：

    1. 用 StateGraph 声明图使用哪种状态结构。
    2. 注册节点。
    3. 注册边，决定执行顺序。
    4. compile() 得到可执行图对象。
    """

    builder = StateGraph(HelloState)

    # 注册节点时，名称只是图里的“路由标识”。
    builder.add_node("normalize_user_input", normalize_user_input)
    builder.add_node("reply_user", reply_user)

    # START 和 END 是 LangGraph 内置的两个特殊节点：
    # - START 表示图的入口
    # - END 表示图的终点
    builder.add_edge(START, "normalize_user_input")
    builder.add_edge("normalize_user_input", "reply_user")
    builder.add_edge("reply_user", END)

    return builder.compile()


def show_graph_structure(graph) -> None:
    """
    draw_mermaid() 返回 Mermaid 文本。

    文档里提到可以可视化图，这里先用最容易理解的方式：
    直接打印 Mermaid 源码，让你知道图内部到底长什么样。
    """

    mermaid = graph.get_graph().draw_mermaid()
    print("=== Mermaid 图结构 ===")
    print(mermaid)
    print()


def main() -> None:
    graph = build_graph()

    show_graph_structure(graph)

    # invoke() 会从 START 开始执行，直到走到 END。
    # 输入必须满足 HelloState 的要求。
    result = graph.invoke(
        {
            "user_input": "  Tell me what LangGraph is.  ",
            "normalized_input": "",
            "answer": "",
        }
    )

    print("=== 图执行结果 ===")
    print(result)
    print()

    # 重点观察：
    # - user_input 是初始输入
    # - normalized_input 由第一个节点生成
    # - answer 由第二个节点生成
    print("=== 学习要点 ===")
    print("1. 节点返回的是状态更新，而不是整个世界。")
    print("2. 边描述控制流，状态描述数据流。")
    print("3. compile() 之后才得到真正可运行的图对象。")


if __name__ == "__main__":
    main()
