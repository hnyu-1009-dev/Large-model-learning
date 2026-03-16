"""
节点、边、Send、Command、Runtime Context。

这个文件覆盖 Graph API 中最重要的控制流能力：

1. 普通边。
2. 条件边。
3. 条件入口。
4. 循环。
5. Send 实现 map-reduce 风格分发。
6. Command 在“更新状态 + 决定下一步”时的优势。
7. runtime.context 如何传入“不是状态的一部分”的运行时配置。
"""

from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict
import operator

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, Send


class RouteState(TypedDict):
    score: int
    route: str
    result: str


def evaluate_score(state: RouteState) -> dict:
    return {}


def choose_route(state: RouteState) -> str:
    """
    条件边函数只负责“决定路由”，自己不更新状态。
    """

    return "advanced" if state["score"] >= 80 else "basic"


def basic_node(state: RouteState) -> dict:
    return {"result": "basic learning path", "route": "basic"}


def advanced_node(state: RouteState) -> dict:
    return {"result": "advanced learning path", "route": "advanced"}


class EntryState(TypedDict):
    mode: str
    message: str


def choose_entry(state: EntryState) -> str:
    """
    条件入口本质上是在 START 之后立刻做一次动态分流。
    """

    return "fast" if state["mode"] == "fast" else "safe"


def fast_entry(state: EntryState) -> dict:
    return {"message": "entered from fast path"}


def safe_entry(state: EntryState) -> dict:
    return {"message": "entered from safe path"}


class LoopState(TypedDict):
    remaining_steps: int
    log: Annotated[list[str], operator.add]


def do_one_step(state: LoopState) -> dict:
    """
    这个节点每次执行一次，把 remaining_steps 减 1。
    """

    current = state["remaining_steps"]
    return {
        "remaining_steps": current - 1,
        "log": [f"processed step {current}"],
    }


def continue_or_stop(state: LoopState) -> str:
    """
    只要 remaining_steps 还大于 0，就继续循环。
    """

    return "do_one_step" if state["remaining_steps"] > 0 else END


class JokeState(TypedDict):
    topics: list[str]
    jokes: Annotated[list[str], operator.add]


class JokeTask(TypedDict):
    topic: str


def prepare_topics(state: JokeState) -> dict:
    """
    Send 前面通常会有一个“准备输入”的节点。
    """

    return {}


def map_topics_to_send(state: JokeState) -> list[Send]:
    """
    Send 允许一个节点把同一轮工作分发成多个并行任务。

    每个 Send 都指定：
    - 发送给哪个节点
    - 该节点要收到什么局部输入
    """

    return [Send("write_joke", {"topic": topic}) for topic in state["topics"]]


def write_joke(state: JokeTask) -> dict:
    return {"jokes": [f"joke about {state['topic']}"]}


class CommandState(TypedDict):
    order_amount: int
    route: str
    note: str


def command_router(state: CommandState) -> Command[Literal["vip_handler", "normal_handler"]]:
    """
    当你需要“同时改状态 + 同时决定 goto”时，Command 比条件边更自然。
    """

    if state["order_amount"] >= 1000:
        return Command(
            update={
                "route": "vip_handler",
                "note": "high-value order routed to VIP handler",
            },
            goto="vip_handler",
        )

    return Command(
        update={
            "route": "normal_handler",
            "note": "normal order routed to standard handler",
        },
        goto="normal_handler",
    )


def vip_handler(state: CommandState) -> dict:
    return {"note": f"{state['note']} | VIP discount prepared"}


def normal_handler(state: CommandState) -> dict:
    return {"note": f"{state['note']} | standard queue"}


class ContextState(TypedDict):
    question: str
    answer: str


@dataclass
class ContextSchema:
    """
    context_schema 里的内容不是图状态的一部分。

    它适合放：
    - 模型名
    - 环境配置
    - 数据库连接
    - 功能开关
    """

    model_name: str
    tone: str


def answer_with_context(state: ContextState, runtime: Runtime[ContextSchema]) -> dict:
    context = runtime.context
    return {
        "answer": (
            f"[model={context.model_name} tone={context.tone}] "
            f"answer to: {state['question']}"
        )
    }


def demo_conditional_edges() -> None:
    print("=== demo_conditional_edges ===")

    builder = StateGraph(RouteState)
    builder.add_node("evaluate_score", evaluate_score)
    builder.add_node("basic_node", basic_node)
    builder.add_node("advanced_node", advanced_node)
    builder.add_edge(START, "evaluate_score")
    builder.add_conditional_edges(
        "evaluate_score",
        choose_route,
        {
            "basic": "basic_node",
            "advanced": "advanced_node",
        },
    )
    builder.add_edge("basic_node", END)
    builder.add_edge("advanced_node", END)

    graph = builder.compile()
    print(graph.invoke({"score": 55, "route": "", "result": ""}))
    print(graph.invoke({"score": 95, "route": "", "result": ""}))
    print()


def demo_conditional_entry_and_loop() -> None:
    print("=== demo_conditional_entry_and_loop ===")

    entry_builder = StateGraph(EntryState)
    entry_builder.add_node("fast_entry", fast_entry)
    entry_builder.add_node("safe_entry", safe_entry)
    entry_builder.add_conditional_edges(
        START,
        choose_entry,
        {
            "fast": "fast_entry",
            "safe": "safe_entry",
        },
    )
    entry_builder.add_edge("fast_entry", END)
    entry_builder.add_edge("safe_entry", END)
    entry_graph = entry_builder.compile()

    print(entry_graph.invoke({"mode": "fast", "message": ""}))
    print(entry_graph.invoke({"mode": "safe", "message": ""}))

    loop_builder = StateGraph(LoopState)
    loop_builder.add_node("do_one_step", do_one_step)
    loop_builder.add_conditional_edges("do_one_step", continue_or_stop)
    loop_builder.add_edge(START, "do_one_step")
    loop_graph = loop_builder.compile()

    print(loop_graph.invoke({"remaining_steps": 3, "log": []}))
    print()


def demo_send() -> None:
    print("=== demo_send ===")

    builder = StateGraph(JokeState)
    builder.add_node("prepare_topics", prepare_topics)
    builder.add_node("write_joke", write_joke)
    builder.add_edge(START, "prepare_topics")
    builder.add_conditional_edges("prepare_topics", map_topics_to_send, ["write_joke"])
    builder.add_edge("write_joke", END)

    graph = builder.compile()
    result = graph.invoke({"topics": ["cats", "dogs", "graphs"], "jokes": []})

    print(result)
    print("Send 很适合 fan-out / map-reduce 类任务。\n")


def demo_command() -> None:
    print("=== demo_command ===")

    builder = StateGraph(CommandState)
    builder.add_node("command_router", command_router)
    builder.add_node("vip_handler", vip_handler)
    builder.add_node("normal_handler", normal_handler)
    builder.add_edge(START, "command_router")
    builder.add_edge("vip_handler", END)
    builder.add_edge("normal_handler", END)

    graph = builder.compile()
    print(graph.invoke({"order_amount": 80, "route": "", "note": ""}))
    print(graph.invoke({"order_amount": 5000, "route": "", "note": ""}))
    print()


def demo_runtime_context() -> None:
    print("=== demo_runtime_context ===")

    builder = StateGraph(ContextState, context_schema=ContextSchema)
    builder.add_node("answer_with_context", answer_with_context)
    builder.add_edge(START, "answer_with_context")
    builder.add_edge("answer_with_context", END)

    graph = builder.compile()
    result = graph.invoke(
        {"question": "What is runtime context?", "answer": ""},
        context=ContextSchema(model_name="mock-llm", tone="teacher"),
    )

    print(result)
    print("这里的 model_name / tone 没有进入状态，却能被节点访问。\n")


def main() -> None:
    demo_conditional_edges()
    demo_conditional_entry_and_loop()
    demo_send()
    demo_command()
    demo_runtime_context()


if __name__ == "__main__":
    main()
