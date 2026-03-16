"""
State schema 专题。

这个文件集中讲清文档中最容易混淆的几个概念：

1. state_schema：图内部完整状态。
2. input_schema：图对外暴露的输入接口。
3. output_schema：图对外暴露的输出接口。
4. 私有状态：只在部分节点间传递，不进入全局状态。
5. dataclass 默认值：当你希望状态字段自带默认值时怎么写。
"""

from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class InputState(TypedDict):
    """图的外部输入接口。"""

    question: str


class OutputState(TypedDict):
    """图的外部输出接口。"""

    answer: str


class OverallState(InputState, OutputState):
    """
    图内部的完整状态。

    内部节点可以读写这个 schema 里的全部字段。
    但调用者未必需要同时输入所有字段，也未必希望看到所有字段。
    所以才会有 input_schema / output_schema。
    """

    cleaned_question: str


def clean_question(state: InputState) -> dict:
    """
    节点只声明自己真正依赖的字段。

    这里只需要 question，因此把入参类型写成 InputState 就够了。
    """

    return {"cleaned_question": state["question"].strip().lower()}


def answer_question(state: OverallState) -> dict:
    """
    第二个节点依赖更完整的状态，因此使用 OverallState。
    """

    cleaned = state["cleaned_question"]

    if "bye" in cleaned:
        answer = "Goodbye from LangGraph."
    elif "state" in cleaned:
        answer = "State is the shared data that flows across the graph."
    else:
        answer = f"Echo: {state['question']}"

    return {"answer": answer}


class PublicState(TypedDict):
    """
    公共状态。

    只有这个 schema 里的字段，才会在整个图里长期存在。
    """

    visible_value: str


class PrivatePayload(TypedDict):
    """
    私有状态。

    这个字段不属于公共状态，只在特定节点之间短暂传递。
    """

    secret_value: str


def produce_private_value(state: PublicState) -> PrivatePayload:
    """
    第一个节点生成一个“只给后继节点消费”的临时字段。
    """

    return {"secret_value": f"secret derived from {state['visible_value']}"}


class PrivateInput(TypedDict):
    """只消费私有字段的输入 schema。"""

    secret_value: str


def consume_private_value(state: PrivateInput) -> PublicState:
    """
    第二个节点把私有字段重新加工后写回公共状态。

    注意：
    secret_value 并不会自动加入全局状态，它只是被这个节点消费。
    """

    return {"visible_value": f"publicized -> {state['secret_value']}"}


def observe_public_state(state: PublicState) -> PublicState:
    """
    这个节点只能看到公共状态，看不到 secret_value。
    """

    return {"visible_value": f"{state['visible_value']} | observed by node_3"}


@dataclass
class DefaultState:
    """
    当你想给状态字段提供默认值时，dataclass 比 TypedDict 更合适。

    这里用 field(default_factory=...) 避免列表默认值被多个实例共享。
    """

    topic: str = "langgraph"
    tags: list[str] = field(default_factory=lambda: ["graph", "state"])
    note: str = ""


def enrich_default_state(state: DefaultState) -> dict:
    """
    dataclass 状态照样返回 dict 更新即可。
    """

    return {
        "tags": state.tags + ["dataclass-default"],
        "note": f"learning topic -> {state.topic}",
    }


def demo_input_output_schema() -> None:
    """
    演示 input_schema / output_schema 的用法。

    外部只输入 question，最终只看到 answer。
    但图内部额外维护了 cleaned_question。
    """

    print("=== demo_input_output_schema ===")

    builder = StateGraph(
        OverallState,
        input_schema=InputState,
        output_schema=OutputState,
    )
    builder.add_node("clean_question", clean_question)
    builder.add_node("answer_question", answer_question)
    builder.add_edge(START, "clean_question")
    builder.add_edge("clean_question", "answer_question")
    builder.add_edge("answer_question", END)

    graph = builder.compile()
    result = graph.invoke({"question": "  What is state in LangGraph?  "})

    print(result)
    print("这里只返回 answer，因为 output_schema 限制了对外输出。\n")


def demo_private_state() -> None:
    """
    演示私有状态只在指定节点之间传递。
    """

    print("=== demo_private_state ===")

    builder = StateGraph(PublicState)
    builder.add_sequence(
        [
            ("produce_private_value", produce_private_value),
            ("consume_private_value", consume_private_value),
            ("observe_public_state", observe_public_state),
        ]
    )
    builder.add_edge(START, "produce_private_value")

    graph = builder.compile()
    result = graph.invoke({"visible_value": "seed"})

    print(result)
    print("注意最终状态里没有 secret_value，因为它从来不是公共状态的一部分。\n")


def demo_dataclass_defaults() -> None:
    """
    演示 dataclass 默认值。
    """

    print("=== demo_dataclass_defaults ===")

    builder = StateGraph(DefaultState)
    builder.add_node("enrich_default_state", enrich_default_state)
    builder.add_edge(START, "enrich_default_state")
    builder.add_edge("enrich_default_state", END)

    graph = builder.compile()

    # 即使只传入部分字段，其余字段也会使用 dataclass 默认值。
    result = graph.invoke({"topic": "reducers"})
    print(result)
    print()


def main() -> None:
    demo_input_output_schema()
    demo_private_state()
    demo_dataclass_defaults()


if __name__ == "__main__":
    main()
