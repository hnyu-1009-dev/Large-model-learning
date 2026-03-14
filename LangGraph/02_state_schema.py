from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


# 定义输入状态模型
class InputState(TypedDict):
    question: str


# 定义输出状态模式
class OutputState(TypedDict):
    answer: str


# 定义整体状态模式，结合输入和输出
class OverallState(InputState, OutputState):
    pass


# 定义处理节点
def answer_node(state: InputState):
    """
    处理输入并生成答案的节点

    Args:
     state: 输入状态

    Returns:
     dict: 包含答案的字典
    """
    print(f"执行 answer_node 节点:")
    print(f" 输入: {state}")

    # 示例答案
    answer = "再见" if "bye" in state["question"].lower() else "你好"
    result = {"answer": answer, "question": state["question"]}

    print(f" 输出: {result}")
    return result


# 定义整体状态（这是在节点间共享的公共状态）
class OverallStatePrivate(TypedDict):
    a: str


# node_1 的输出包含不属于整体状态的私有数据
class Node1Output(TypedDict):
    private_data: str


# 私有数据仅在 node_1 和 node_2 之间共享
def node_1(state: OverallStatePrivate) -> Node1Output:
    """
    第一个节点，生成私有数据

    Args:
     state: 整体状态

    Returns:
     dict: 包含私有数据的字典
    """
    output = {"private_data": "由 node_1 设置"}
    print(f"进入 node_1 节点:")
    print(f" 输入: {state}")
    print(f" 返回: {output}")
    return output


# 节点2的输入只请求node_1之后可用的私有数据
class Node2Input(TypedDict):
    private_data: str


def node_2(state: Node2Input) -> OverallStatePrivate:
    """
    第二个节点，可以访问node_1的私有数据

    Args:
     state: 包含私有数据的输入状态

    Returns:
     dict: 更新的整体状态
    """
    output = {"a": "由 node_2 设置"}
    print(f"进入 node_2 节点:")
    print(f" 输入: {state}")
    print(f" 返回: {output}")
    return output


# 节点3只能访问整体状态（无法访问node_1的私有数据）
def node_3(state: OverallStatePrivate) -> OverallStatePrivate:
    """
    第三个节点，只能访问整体状态

    Args:
     state: 整体状态

    Returns:
     dict: 更新的整体状态
    """
    output = {"a": "由 node_3 设置"}
    print(f"进入 node_3 节点:")
    print(f" 输入: {state}")
    print(f" 返回: {output}")
    return output


def demo_input_output_schema():
    """演示输入输出模式"""
    print("=== 演示输入输出模式 ===")

    # 使用指定的输入和输出模式构建图
    builder = StateGraph(
        OverallState, input_schema=InputState, output_schema=OutputState
    )
    builder.add_node("answer_node", answer_node)  # 添加答案节点
    builder.add_edge(START, "answer_node")  # 定义起始边
    builder.add_edge("answer_node", END)  # 定义结束边
    graph = builder.compile()  # 编译图

    # 使用输入调用图并打印结果
    result = graph.invoke({"question": "你好"})
    print(f"图调用结果: {result}")

    print()
