from typing import TypedDict
from langgraph.graph import StateGraph
from langgraph.constants import START, END


# 状态的定义
class OverallState(TypedDict):
    user_input: str


# 节点的定义：任何能进行逻辑操作的结构
def node_1(state: OverallState) -> OverallState:
    user_input = state["user_input"]
    print(user_input)
    # return state["user_input"]


# 注册图
# 传入主状态信息
builder = StateGraph(OverallState)

# 注册结点
builder.add_node("node_1", node_1)

# 注册边
builder.add_edge(START, "node_1")
builder.add_edge("node_1", END)


# 编译图
graph = builder.compile()

# 调用图
graph.invoke(
    {
        "user_input":"hello"
    }
)
