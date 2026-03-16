"""
综合示例：一个有短期记忆、长期记忆、人工审批的迷你聊天机器人。

这个例子不是为了做一个真正强大的机器人，
而是把前面几个最关键的能力组合起来：

1. Messages 风格的消息管理
2. runtime.context 传递运行时配置
3. runtime.store 作为长期记忆
4. checkpointer 作为短期记忆和中断恢复基础
5. Command 控制流跳转
6. interrupt 让人工审批进入工作流
"""

from dataclasses import dataclass
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, interrupt


class TutorState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    profile: str
    pending_memory: str
    tool_result: str


@dataclass
class TutorContext:
    assistant_name: str


def load_user_profile(state: TutorState, runtime: Runtime[TutorContext]) -> dict:
    namespace = ("users", state["user_id"], "profile")
    item = runtime.store.get(namespace, "latest")
    profile = item.value["text"] if item else "no profile yet"
    return {"profile": profile}


def route_user_intent(state: TutorState, runtime: Runtime[TutorContext]) -> Command | dict:
    """
    这里不用真实 LLM，而是用规则模拟意图识别。

    这样更利于学习 LangGraph 本身，而不会把注意力分散到模型调用细节。
    """

    last_human_message = next(
        message for message in reversed(state["messages"]) if isinstance(message, HumanMessage)
    )
    text = last_human_message.content.strip()
    lowered = text.lower()

    if lowered.startswith("remember:"):
        memory_text = text.split(":", 1)[1].strip()
        return Command(update={"pending_memory": memory_text}, goto="approve_memory_write")

    if "what do you remember" in lowered:
        return {
            "messages": [
                AIMessage(
                    content=(
                        f"{runtime.context.assistant_name} remembers: {state['profile']}"
                    )
                )
            ]
        }

    return {
        "messages": [
            AIMessage(
                content=(
                    f"{runtime.context.assistant_name}: "
                    "say 'remember: ...' to store a fact, "
                    "or ask 'what do you remember'."
                )
            )
        ]
    }


def approve_memory_write(state: TutorState) -> Command:
    decision = interrupt(
        {
            "question": f"Approve saving memory '{state['pending_memory']}'?",
            "pending_memory": state["pending_memory"],
        }
    )

    if decision["approved"]:
        return Command(goto="save_memory")

    return Command(update={"tool_result": "memory write rejected"}, goto="finalize_tool_result")


def save_memory(state: TutorState, runtime: Runtime[TutorContext]) -> dict:
    namespace = ("users", state["user_id"], "profile")
    runtime.store.put(namespace, "latest", {"text": state["pending_memory"]})
    return {"tool_result": f"saved memory: {state['pending_memory']}"}


def finalize_tool_result(state: TutorState, runtime: Runtime[TutorContext]) -> dict:
    return {
        "messages": [AIMessage(content=f"{runtime.context.assistant_name}: {state['tool_result']}")],
        "pending_memory": "",
        "tool_result": "",
    }


def build_tutor_graph():
    builder = StateGraph(TutorState, context_schema=TutorContext)
    builder.add_node("load_user_profile", load_user_profile)
    builder.add_node("route_user_intent", route_user_intent)
    builder.add_node("approve_memory_write", approve_memory_write)
    builder.add_node("save_memory", save_memory)
    builder.add_node("finalize_tool_result", finalize_tool_result)

    builder.add_edge(START, "load_user_profile")
    builder.add_edge("load_user_profile", "route_user_intent")
    builder.add_edge("save_memory", "finalize_tool_result")
    builder.add_edge("route_user_intent", END)
    builder.add_edge("finalize_tool_result", END)

    return builder.compile(
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
    )


def main() -> None:
    graph = build_tutor_graph()
    context = TutorContext(assistant_name="TutorBot")
    config = {"configurable": {"thread_id": "tutor-thread"}}

    print("=== turn 1: normal chat ===")
    turn_1 = graph.invoke(
        {
            "messages": [HumanMessage(content="hello", id="human-1")],
            "user_id": "alice",
            "profile": "",
            "pending_memory": "",
            "tool_result": "",
        },
        config=config,
        context=context,
    )
    print(turn_1)
    print()

    print("=== turn 2: request memory write, graph gets interrupted ===")
    turn_2 = graph.invoke(
        {
            "messages": [HumanMessage(content="remember: likes LangGraph", id="human-2")],
            "user_id": "alice",
            "profile": "",
            "pending_memory": "",
            "tool_result": "",
        },
        config=config,
        context=context,
    )
    print(turn_2)
    print("interrupts:", graph.get_state(config).interrupts)
    print()

    print("=== turn 2 resume: approve the memory write ===")
    resumed = graph.invoke(
        Command(resume={"approved": True}),
        config=config,
        context=context,
    )
    print(resumed)
    print()

    print("=== turn 3: ask what the bot remembers ===")
    turn_3 = graph.invoke(
        {
            "messages": [HumanMessage(content="what do you remember", id="human-3")],
            "user_id": "alice",
            "profile": "",
            "pending_memory": "",
            "tool_result": "",
        },
        config=config,
        context=context,
    )
    print(turn_3)
    print()

    print("=== 学习总结 ===")
    print("1. 同一个 thread_id 让 messages 形成短期记忆。")
    print("2. runtime.store 让 profile 形成长期记忆。")
    print("3. interrupt + resume 让人工审批接入图工作流。")
    print("4. Command 让节点在更新状态时顺便跳转控制流。")


if __name__ == "__main__":
    main()
