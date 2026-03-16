"""
Functional API。

Graph API 适合你显式搭建节点和边；
Functional API 适合你把工作流写成“任务 + 入口点”的形式。
"""

import time

from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.graph import END, START, StateGraph
from langgraph.types import CachePolicy, Command, RetryPolicy, interrupt


@task
def double_number(x: int) -> int:
    return x * 2


@entrypoint(checkpointer=InMemorySaver())
def basic_workflow(x: int) -> int:
    return double_number(x).result()


def simulate_io(topic: str, delay: float = 0.2) -> str:
    time.sleep(delay)
    return f"paragraph about {topic}"


@task
def generate_paragraph(topic: str) -> str:
    return simulate_io(topic)


@entrypoint(checkpointer=InMemorySaver())
def parallel_workflow(topics: list[str]) -> str:
    futures = [generate_paragraph(topic) for topic in topics]
    paragraphs = [future.result() for future in futures]
    return "\n".join(paragraphs)


@entrypoint(checkpointer=InMemorySaver())
def running_total_workflow(number: int, *, previous: int | None = None):
    running_total = (previous or 0) + number
    return entrypoint.final(
        value={"latest": number, "running_total": running_total},
        save=running_total,
    )


@task
def draft_article(topic: str) -> str:
    return f"draft article about {topic}"


@entrypoint(checkpointer=InMemorySaver())
def review_workflow(topic: str) -> str:
    draft = draft_article(topic).result()
    decision = interrupt({"draft": draft, "question": "Edit this draft before publishing?"})
    return decision["final_text"]


TASK_RETRY_CALLS = {"count": 0}


@task(
    retry_policy=RetryPolicy(
        max_attempts=3,
        initial_interval=0.01,
        max_interval=0.02,
        jitter=False,
        retry_on=ValueError,
    )
)
def flaky_task(x: int) -> int:
    TASK_RETRY_CALLS["count"] += 1
    if TASK_RETRY_CALLS["count"] < 2:
        raise ValueError("retry me")
    return x * 10


@task(cache_policy=CachePolicy())
def cached_task(x: int) -> int:
    print(f"cached_task executes with x={x}")
    return x + 1


@entrypoint(cache=InMemoryCache())
def cached_and_retried_workflow(x: int) -> dict:
    return {
        "flaky": flaky_task(x).result(),
        "cached_first": cached_task(x).result(),
        "cached_second": cached_task(x).result(),
    }


def double_state(state):
    return {"value": state["value"] * 2}


def add_five_state(state):
    return {"value": state["value"] + 5}


def build_graph_api_demo():
    builder = StateGraph(dict)
    builder.add_node("double_state", double_state)
    builder.add_node("add_five_state", add_five_state)
    builder.add_edge(START, "double_state")
    builder.add_edge("double_state", "add_five_state")
    builder.add_edge("add_five_state", END)
    return builder.compile()


GRAPH_API_DEMO = build_graph_api_demo()


@task
def process_number_by_graph(x: int) -> int:
    result = GRAPH_API_DEMO.invoke({"value": x})
    return result["value"]


@entrypoint(checkpointer=InMemorySaver())
def mixed_api_workflow(numbers: list[int]) -> dict:
    futures = [process_number_by_graph(number) for number in numbers]
    outputs = [future.result() for future in futures]
    return {"inputs": numbers, "outputs": outputs, "sum": sum(outputs)}


def demo_basic_functional_api() -> None:
    print("=== demo_basic_functional_api ===")
    print(basic_workflow.invoke(6, config={"configurable": {"thread_id": "basic-workflow"}}))
    print()


def demo_parallel_workflow() -> None:
    print("=== demo_parallel_workflow ===")
    print(
        parallel_workflow.invoke(
            ["langgraph", "state", "agent"],
            config={"configurable": {"thread_id": "parallel-workflow-thread"}},
        )
    )
    print()


def demo_entrypoint_final() -> None:
    print("=== demo_entrypoint_final ===")

    config = {"configurable": {"thread_id": "running-total-thread"}}
    print(running_total_workflow.invoke(3, config=config))
    print(running_total_workflow.invoke(4, config=config))
    print(running_total_workflow.invoke(5, config=config))
    print()


def demo_interrupt_resume() -> None:
    print("=== demo_interrupt_resume ===")

    config = {"configurable": {"thread_id": "review-thread"}}
    first = review_workflow.invoke("LangGraph", config=config)
    print("first invoke:", first)
    print("interrupts:", review_workflow.get_state(config).interrupts)
    second = review_workflow.invoke(
        Command(resume={"final_text": "edited article about LangGraph"}),
        config=config,
    )
    print("after resume:", second)
    print()


def demo_retry_and_cache() -> None:
    print("=== demo_retry_and_cache ===")

    TASK_RETRY_CALLS["count"] = 0
    print(cached_and_retried_workflow.invoke(8))
    print(f"flaky_task real calls = {TASK_RETRY_CALLS['count']}")
    print()


def demo_mixed_api() -> None:
    print("=== demo_mixed_api ===")
    print(
        mixed_api_workflow.invoke(
            [1, 2, 3],
            config={"configurable": {"thread_id": "mixed-api-thread"}},
        )
    )
    print()


def main() -> None:
    demo_basic_functional_api()
    demo_parallel_workflow()
    demo_entrypoint_final()
    demo_interrupt_resume()
    demo_retry_and_cache()
    demo_mixed_api()


if __name__ == "__main__":
    main()
