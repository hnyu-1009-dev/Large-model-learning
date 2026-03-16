"""
持久化、流式输出、中断恢复。

这个文件对应文档中的高级特性部分，重点覆盖：

1. Threads 和 Checkpoints
2. InMemorySaver / SqliteSaver
3. stream() 的常见模式
4. 自定义流数据
5. interrupt() 与 Command(resume=...)
"""

from pathlib import Path
from typing import Annotated, TypedDict
import operator

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt


class CounterState(TypedDict):
    count: int


def increment(state: CounterState) -> dict:
    return {"count": state["count"] + 1}


def build_counter_graph(checkpointer):
    builder = StateGraph(CounterState)
    builder.add_node("increment_1", increment)
    builder.add_node("increment_2", increment)
    builder.add_edge(START, "increment_1")
    builder.add_edge("increment_1", "increment_2")
    builder.add_edge("increment_2", END)
    return builder.compile(checkpointer=checkpointer)


class StreamState(TypedDict):
    task_name: str
    log: Annotated[list[str], operator.add]
    done: bool


def stream_worker(state: StreamState, runtime: Runtime[None]) -> dict:
    runtime.stream_writer({"event": "progress", "message": f"start {state['task_name']}"})
    runtime.stream_writer({"event": "progress", "message": f"finish {state['task_name']}"})
    return {"log": [f"processed {state['task_name']}"], "done": True}


class ApprovalState(TypedDict):
    task_name: str
    approved: bool
    note: str


def approval_gate(state: ApprovalState) -> dict:
    decision = interrupt(
        {
            "question": f"Do you approve task: {state['task_name']}?",
            "current_state": state,
        }
    )
    return {"approved": decision["approved"], "note": decision["note"]}


def finalize_approval(state: ApprovalState) -> dict:
    if state["approved"]:
        return {"note": f"approved -> {state['note']}"}
    return {"note": f"rejected -> {state['note']}"}


def demo_in_memory_checkpointer() -> None:
    print("=== demo_in_memory_checkpointer ===")

    graph = build_counter_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "counter-thread"}}

    result = graph.invoke({"count": 0}, config=config)
    current_state = graph.get_state(config)
    history = list(graph.get_state_history(config))

    print("final result:", result)
    print("current state:", current_state.values)
    print("history checkpoints:", len(history))
    print("latest checkpoint config:", history[0].config)
    print()


def demo_sqlite_checkpointer() -> None:
    print("=== demo_sqlite_checkpointer ===")

    db_path = Path(__file__).with_name("langgraph_demo_checkpoints.sqlite")

    with SqliteSaver.from_conn_string(str(db_path)) as saver:
        graph = build_counter_graph(saver)
        config = {"configurable": {"thread_id": "sqlite-counter-thread"}}
        result = graph.invoke({"count": 10}, config=config)

    print("sqlite result:", result)
    print(f"sqlite db created at: {db_path}")
    print()


def demo_stream_modes() -> None:
    print("=== demo_stream_modes ===")

    builder = StateGraph(StreamState)
    builder.add_node("stream_worker", stream_worker)
    builder.add_edge(START, "stream_worker")
    builder.add_edge("stream_worker", END)
    graph = builder.compile()

    print("-- values mode --")
    for chunk in graph.stream(
        {"task_name": "demo", "log": [], "done": False},
        stream_mode="values",
    ):
        print(chunk)

    print("-- updates + custom mode --")
    for chunk in graph.stream(
        {"task_name": "demo", "log": [], "done": False},
        stream_mode=["updates", "custom"],
    ):
        print(chunk)

    print()


def demo_interrupt_resume() -> None:
    print("=== demo_interrupt_resume ===")

    builder = StateGraph(ApprovalState)
    builder.add_node("approval_gate", approval_gate)
    builder.add_node("finalize_approval", finalize_approval)
    builder.add_edge(START, "approval_gate")
    builder.add_edge("approval_gate", "finalize_approval")
    builder.add_edge("finalize_approval", END)

    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "approval-thread"}}

    first_result = graph.invoke(
        {"task_name": "delete stale rows", "approved": False, "note": "waiting"},
        config=config,
    )
    print("first invoke:", first_result)
    print("stored interrupts:", graph.get_state(config).interrupts)

    resumed_result = graph.invoke(
        Command(resume={"approved": True, "note": "approved by reviewer"}),
        config=config,
    )
    print("after resume:", resumed_result)
    print()


def print_durable_execution_notes() -> None:
    print("=== durable execution notes ===")
    print("1. 尽量让节点幂等：重复执行同一步时结果应可接受。")
    print("2. 把随机数、时间、外部副作用显式包起来，不要散落在节点里。")
    print("3. 用检查点保存已完成步骤，避免恢复时重做昂贵工作。")
    print("4. 恢复执行时，追求的是一致性，不只是程序不报错。")
    print()


def main() -> None:
    demo_in_memory_checkpointer()
    demo_sqlite_checkpointer()
    demo_stream_modes()
    demo_interrupt_resume()
    print_durable_execution_notes()


if __name__ == "__main__":
    main()
