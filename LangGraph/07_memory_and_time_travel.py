"""
记忆与时间旅行。

这个文件把文档里的 Memory + Time Travel 放在一起讲，原因是：
两者都建立在“检查点可回看、状态可恢复”这个基础之上。
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, trim_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore


class ConversationState(TypedDict):
    messages: Annotated[list, add_messages]


def reply_with_history(state: ConversationState) -> dict:
    message_count = len(state["messages"])
    return {
        "messages": [
            AIMessage(content=f"I can see {message_count} message(s) in this thread.")
        ]
    }


class CounterState(TypedDict):
    count: int


def increment(state: CounterState) -> dict:
    return {"count": state["count"] + 1}


class PreferenceState(TypedDict):
    user_id: str
    preference: str
    remembered_preferences: list[str]


def save_preference(state: PreferenceState, runtime: Runtime[None]) -> dict:
    namespace = ("users", state["user_id"], "preferences")
    key = f"pref-{state['preference']}"
    runtime.store.put(namespace, key, {"kind": "preference", "text": state["preference"]})
    return {}


def load_preferences(state: PreferenceState, runtime: Runtime[None]) -> dict:
    namespace = ("users", state["user_id"], "preferences")
    items = runtime.store.search(namespace, filter={"kind": "preference"})
    return {"remembered_preferences": [item.value["text"] for item in items]}


class ManagedChatState(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str


def trim_message_history(state: ManagedChatState) -> dict:
    trimmed_messages = trim_messages(
        state["messages"],
        max_tokens=2,
        token_counter=lambda messages: len(messages),
        strategy="last",
    )
    return {
        "messages": [RemoveMessage(id=msg.id) for msg in state["messages"]] + trimmed_messages,
    }


def delete_first_message(state: ManagedChatState) -> dict:
    first_message_id = state["messages"][0].id
    return {"messages": [RemoveMessage(id=first_message_id)]}


def summarize_old_messages(state: ManagedChatState) -> dict:
    old_messages = state["messages"][:-1]
    summary = " | ".join(getattr(message, "content", "") for message in old_messages)
    removals = [RemoveMessage(id=message.id) for message in old_messages]
    return {"summary": summary, "messages": removals}


def demo_short_term_memory() -> None:
    print("=== demo_short_term_memory ===")

    builder = StateGraph(ConversationState)
    builder.add_node("reply_with_history", reply_with_history)
    builder.add_edge(START, "reply_with_history")
    builder.add_edge("reply_with_history", END)

    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "memory-thread"}}

    first_turn = graph.invoke(
        {"messages": [HumanMessage(content="hello", id="human-1")]},
        config=config,
    )
    second_turn = graph.invoke(
        {"messages": [HumanMessage(content="remember me", id="human-2")]},
        config=config,
    )

    print("first turn:", first_turn)
    print("second turn:", second_turn)
    print()


def demo_time_travel() -> None:
    print("=== demo_time_travel ===")

    builder = StateGraph(CounterState)
    builder.add_node("increment_1", increment)
    builder.add_node("increment_2", increment)
    builder.add_edge(START, "increment_1")
    builder.add_edge("increment_1", "increment_2")
    builder.add_edge("increment_2", END)

    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "time-travel-thread"}}

    final_result = graph.invoke({"count": 0}, config=config)
    history = list(graph.get_state_history(config))
    checkpoint_before_increment_2 = history[1].config

    fork_config = graph.update_state(
        checkpoint_before_increment_2,
        {"count": 10},
        as_node="increment_1",
    )
    fork_result = graph.invoke(None, config=fork_config)

    print("original final result:", final_result)
    print("fork result from old checkpoint:", fork_result)
    print()


def demo_long_term_memory() -> None:
    print("=== demo_long_term_memory ===")

    builder = StateGraph(PreferenceState)
    builder.add_node("save_preference", save_preference)
    builder.add_node("load_preferences", load_preferences)
    builder.add_edge(START, "save_preference")
    builder.add_edge("save_preference", "load_preferences")
    builder.add_edge("load_preferences", END)

    store = InMemoryStore()
    graph = builder.compile(store=store)

    result_1 = graph.invoke(
        {"user_id": "alice", "preference": "python", "remembered_preferences": []}
    )
    result_2 = graph.invoke(
        {"user_id": "alice", "preference": "graphs", "remembered_preferences": []}
    )

    print("after first memory:", result_1)
    print("after second memory:", result_2)
    print()


def demo_manage_short_term_memory() -> None:
    print("=== demo_manage_short_term_memory ===")

    initial_state = {
        "messages": [
            HumanMessage(content="message-1", id="m1"),
            AIMessage(content="message-2", id="m2"),
            HumanMessage(content="message-3", id="m3"),
        ],
        "summary": "",
    }

    trim_builder = StateGraph(ManagedChatState)
    trim_builder.add_node("trim_message_history", trim_message_history)
    trim_builder.add_edge(START, "trim_message_history")
    trim_builder.add_edge("trim_message_history", END)
    trim_graph = trim_builder.compile()
    print("trimmed:", trim_graph.invoke(initial_state))

    delete_builder = StateGraph(ManagedChatState)
    delete_builder.add_node("delete_first_message", delete_first_message)
    delete_builder.add_edge(START, "delete_first_message")
    delete_builder.add_edge("delete_first_message", END)
    delete_graph = delete_builder.compile()
    print("deleted first message:", delete_graph.invoke(initial_state))

    summarize_builder = StateGraph(ManagedChatState)
    summarize_builder.add_node("summarize_old_messages", summarize_old_messages)
    summarize_builder.add_edge(START, "summarize_old_messages")
    summarize_builder.add_edge("summarize_old_messages", END)
    summarize_graph = summarize_builder.compile()
    print("summarized:", summarize_graph.invoke(initial_state))
    print()


def main() -> None:
    demo_short_term_memory()
    demo_time_travel()
    demo_long_term_memory()
    demo_manage_short_term_memory()


if __name__ == "__main__":
    main()
