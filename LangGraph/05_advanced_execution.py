"""
执行层高级特性。

这个文件对应文档里和节点执行策略有关的内容：

1. Node caching
2. Retry policy
3. Deferred node
4. Async graph

这些能力的共同目标是：
让图在真实生产环境里更稳、更省、更可控。
"""

import asyncio
from typing import Annotated, TypedDict
import operator

from langgraph.cache.memory import InMemoryCache
from langgraph.graph import END, START, StateGraph
from langgraph.types import CachePolicy, RetryPolicy


class CacheState(TypedDict):
    x: int
    y: int


CACHE_CALLS = {"count": 0}


def expensive_node(state: CacheState) -> dict:
    """
    用全局计数器模拟“昂贵节点被执行了几次”。
    """

    CACHE_CALLS["count"] += 1
    return {"y": state["x"] * 10}


class RetryState(TypedDict):
    attempts: int
    ok: bool


RETRY_CALLS = {"count": 0}


def flaky_node(state: RetryState) -> dict:
    """
    第一次故意失败，第二次成功。
    """

    RETRY_CALLS["count"] += 1
    if RETRY_CALLS["count"] < 2:
        raise ValueError("temporary error")
    return {"attempts": RETRY_CALLS["count"], "ok": True}


class DeferredState(TypedDict):
    log: Annotated[list[str], operator.add]


def main_work(state: DeferredState) -> dict:
    return {"log": ["main work finished"]}


def cleanup_work(state: DeferredState) -> dict:
    """
    defer=True 的典型用途：
    - 清理资源
    - 收尾日志
    - 延迟上报
    """

    return {"log": ["cleanup executed after normal path"]}


class AsyncState(TypedDict):
    value: int
    doubled: int


async def async_double(state: AsyncState) -> dict:
    """
    异步节点适合 IO 密集型工作，例如：
    - 调 API
    - 查数据库
    - 并发访问多个远程服务
    """

    await asyncio.sleep(0.05)
    return {"doubled": state["value"] * 2}


def demo_node_cache() -> None:
    print("=== demo_node_cache ===")

    CACHE_CALLS["count"] = 0

    builder = StateGraph(CacheState)
    builder.add_node("expensive_node", expensive_node, cache_policy=CachePolicy())
    builder.add_edge(START, "expensive_node")
    builder.add_edge("expensive_node", END)

    graph = builder.compile(cache=InMemoryCache())
    print(graph.invoke({"x": 2, "y": 0}))
    print(graph.invoke({"x": 2, "y": 0}))
    print(f"expensive_node real calls = {CACHE_CALLS['count']}")
    print("相同输入第二次命中缓存，因此节点不会再次执行。\n")


def demo_retry_policy() -> None:
    print("=== demo_retry_policy ===")

    RETRY_CALLS["count"] = 0

    builder = StateGraph(RetryState)
    builder.add_node(
        "flaky_node",
        flaky_node,
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_interval=0.01,
            max_interval=0.02,
            jitter=False,
            retry_on=ValueError,
        ),
    )
    builder.add_edge(START, "flaky_node")
    builder.add_edge("flaky_node", END)

    graph = builder.compile()
    print(graph.invoke({"attempts": 0, "ok": False}))
    print(f"flaky_node real calls = {RETRY_CALLS['count']}")
    print()


def demo_deferred_node() -> None:
    print("=== demo_deferred_node ===")

    builder = StateGraph(DeferredState)
    builder.add_node("main_work", main_work)
    builder.add_node("cleanup_work", cleanup_work, defer=True)
    builder.add_edge(START, "main_work")
    builder.add_edge("main_work", END)
    builder.add_edge("main_work", "cleanup_work")

    graph = builder.compile()
    print(graph.invoke({"log": []}))
    print("defer 节点适合做最终清理，不影响主路径建模。\n")


def demo_async_graph() -> None:
    print("=== demo_async_graph ===")

    builder = StateGraph(AsyncState)
    builder.add_node("async_double", async_double)
    builder.add_edge(START, "async_double")
    builder.add_edge("async_double", END)

    graph = builder.compile()
    result = asyncio.run(graph.ainvoke({"value": 21, "doubled": 0}))

    print(result)
    print("当节点是 async def 时，可以使用 ainvoke() / astream()。\n")


def main() -> None:
    demo_node_cache()
    demo_retry_policy()
    demo_deferred_node()
    demo_async_graph()


if __name__ == "__main__":
    main()
