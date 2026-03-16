"""
Reducer 专题。

文档中提到：状态由 schema + reducer 共同定义。
如果你只写 schema，不理解 reducer，就很容易误判状态合并结果。

这个文件覆盖：

1. 默认覆盖行为。
2. operator.add 追加列表 / 字符串。
3. operator.mul 用于累乘。
4. 自定义 reducer。
5. add_messages 与 RemoveMessage。
"""

from typing import Annotated, TypedDict
import operator

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class OverwriteState(TypedDict):
    """
    没有 Annotated reducer 时，LangGraph 默认使用“覆盖”策略。
    """

    counter: int


def set_counter_to_one(state: OverwriteState) -> dict:
    return {"counter": 1}


def set_counter_to_hundred(state: OverwriteState) -> dict:
    return {"counter": 100}


class AddState(TypedDict):
    """
    operator.add 会把左右两边做加法：
    - 对 list 来说是拼接
    - 对 str 来说是连接
    """

    items: Annotated[list[str], operator.add]
    text: Annotated[str, operator.add]


def append_part_a(state: AddState) -> dict:
    return {"items": ["A"], "text": "Lang"}


def append_part_b(state: AddState) -> dict:
    return {"items": ["B"], "text": "Graph"}


class MultiplyState(TypedDict):
    """
    operator.mul 的例子更少见，但能帮助你直观理解：
    reducer 本质上只是“旧值 + 新值怎样合并”的规则。
    """

    product: Annotated[int, operator.mul]


def multiply_by_two(state: MultiplyState) -> dict:
    return {"product": 2}


def multiply_by_three(state: MultiplyState) -> dict:
    return {"product": 3}


def deduplicate_keep_order(left: list[str], right: list[str]) -> list[str]:
    """
    自定义 reducer。

    这类 reducer 很常见：
    - 既要保留历史
    - 又不希望重复
    """

    merged: list[str] = []
    for item in [*(left or []), *(right or [])]:
        if item not in merged:
            merged.append(item)
    return merged


class CustomReducerState(TypedDict):
    tags: Annotated[list[str], deduplicate_keep_order]


def emit_base_tags(state: CustomReducerState) -> dict:
    return {"tags": ["graph", "state", "graph"]}


def emit_more_tags(state: CustomReducerState) -> dict:
    return {"tags": ["agent", "state", "memory"]}


class ChatState(TypedDict):
    """
    add_messages 是消息场景最重要的 reducer。

    它的作用不是简单 append，而是按消息 ID 做更智能的合并。
    所以在聊天系统里应优先用它，而不是直接 operator.add。
    """

    messages: Annotated[list, add_messages]


def add_ai_reply(state: ChatState) -> dict:
    return {"messages": [AIMessage(content="This is a reply from the assistant.", id="ai-1")]}


def revise_ai_reply(state: ChatState) -> dict:
    """
    因为消息 ID 相同，所以这一步不是“新增一条消息”，而是“替换原消息”。
    """

    return {"messages": [AIMessage(content="This is the revised assistant reply.", id="ai-1")]}


def delete_first_human_message(state: ChatState) -> dict:
    """
    RemoveMessage 用于删除已有消息。
    """

    first_message_id = state["messages"][0].id
    return {"messages": [RemoveMessage(id=first_message_id)]}


def demo_default_overwrite() -> None:
    print("=== demo_default_overwrite ===")

    builder = StateGraph(OverwriteState)
    builder.add_node("set_counter_to_one", set_counter_to_one)
    builder.add_node("set_counter_to_hundred", set_counter_to_hundred)
    builder.add_edge(START, "set_counter_to_one")
    builder.add_edge("set_counter_to_one", "set_counter_to_hundred")
    builder.add_edge("set_counter_to_hundred", END)

    graph = builder.compile()
    result = graph.invoke({"counter": 0})

    print(result)
    print("最终值是 100，因为后一个节点覆盖了前一个节点写入的 counter。\n")


def demo_operator_add() -> None:
    print("=== demo_operator_add ===")

    builder = StateGraph(AddState)
    builder.add_node("append_part_a", append_part_a)
    builder.add_node("append_part_b", append_part_b)
    builder.add_edge(START, "append_part_a")
    builder.add_edge("append_part_a", "append_part_b")
    builder.add_edge("append_part_b", END)

    graph = builder.compile()
    result = graph.invoke({"items": [], "text": ""})

    print(result)
    print("items 被追加，text 被拼接。\n")


def demo_operator_mul() -> None:
    print("=== demo_operator_mul ===")

    builder = StateGraph(MultiplyState)
    builder.add_node("multiply_by_two", multiply_by_two)
    builder.add_node("multiply_by_three", multiply_by_three)
    builder.add_edge(START, "multiply_by_two")
    builder.add_edge("multiply_by_two", "multiply_by_three")
    builder.add_edge("multiply_by_three", END)

    graph = builder.compile()

    # 对乘法 reducer 来说，初始值通常应当是 1。
    result = graph.invoke({"product": 1})
    print(result)
    print("结果是 6，因为合并过程等价于 1 * 2 * 3。\n")


def demo_custom_reducer() -> None:
    print("=== demo_custom_reducer ===")

    builder = StateGraph(CustomReducerState)
    builder.add_node("emit_base_tags", emit_base_tags)
    builder.add_node("emit_more_tags", emit_more_tags)
    builder.add_edge(START, "emit_base_tags")
    builder.add_edge("emit_base_tags", "emit_more_tags")
    builder.add_edge("emit_more_tags", END)

    graph = builder.compile()
    result = graph.invoke({"tags": []})

    print(result)
    print("自定义 reducer 保留顺序并去重，这比简单 append 更适合标签类状态。\n")


def demo_add_messages() -> None:
    print("=== demo_add_messages ===")

    builder = StateGraph(ChatState)
    builder.add_node("add_ai_reply", add_ai_reply)
    builder.add_node("revise_ai_reply", revise_ai_reply)
    builder.add_node("delete_first_human_message", delete_first_human_message)
    builder.add_edge(START, "add_ai_reply")
    builder.add_edge("add_ai_reply", "revise_ai_reply")
    builder.add_edge("revise_ai_reply", "delete_first_human_message")
    builder.add_edge("delete_first_human_message", END)

    graph = builder.compile()
    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content="Hello", id="human-1"),
            ]
        }
    )

    print(result)
    print("你会看到：旧的人类消息被删除，AI 消息按相同 ID 被替换而不是重复追加。\n")


def main() -> None:
    demo_default_overwrite()
    demo_operator_add()
    demo_operator_mul()
    demo_custom_reducer()
    demo_add_messages()


if __name__ == "__main__":
    main()
