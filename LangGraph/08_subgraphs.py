"""
子图专题。

子图的核心价值是“分层建模”：
你可以先把一个局部流程做成独立图，再把它当作更大图中的一个节点使用。
"""

from typing import Annotated, TypedDict
import operator

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


class ResearchState(TypedDict):
    topic: str
    notes: Annotated[list[str], operator.add]
    summary: str


def collect_fact_a(state: ResearchState) -> dict:
    return {"notes": [f"{state['topic']} fact A"]}


def collect_fact_b(state: ResearchState) -> dict:
    return {"notes": [f"{state['topic']} fact B"]}


def summarize_research(state: ResearchState) -> dict:
    return {"summary": " | ".join(state["notes"])}


def build_research_subgraph():
    builder = StateGraph(ResearchState)
    builder.add_node("collect_fact_a", collect_fact_a)
    builder.add_node("collect_fact_b", collect_fact_b)
    builder.add_node("summarize_research", summarize_research)
    builder.add_edge(START, "collect_fact_a")
    builder.add_edge("collect_fact_a", "collect_fact_b")
    builder.add_edge("collect_fact_b", "summarize_research")
    builder.add_edge("summarize_research", END)
    return builder.compile(name="research_subgraph")


class ParentCallState(TypedDict):
    topic: str
    report: str


def run_subgraph_inside_node(state: ParentCallState) -> dict:
    subgraph = build_research_subgraph()
    subgraph_result = subgraph.invoke({"topic": state["topic"], "notes": [], "summary": ""})
    return {"report": subgraph_result["summary"]}


class ParentNodeState(TypedDict):
    topic: str
    notes: Annotated[list[str], operator.add]
    summary: str
    final_report: str


def finalize_report(state: ParentNodeState) -> dict:
    return {"final_report": f"final report -> {state['summary']}"}


class ParentNavigateState(TypedDict):
    messages: Annotated[list[str], operator.add]
    task_status: str
    subtask_result: str


class ChildNavigateState(TypedDict):
    messages: Annotated[list[str], operator.add]
    task_status: str
    subtask_result: str
    child_data: str


def main_controller(state: ParentNavigateState) -> Command:
    return Command(
        update={"messages": ["start child task"], "task_status": "subtask_started"},
        goto="subgraph_node",
    )


def data_processor(state: ChildNavigateState) -> Command:
    return Command(
        update={
            "messages": ["child task finished"],
            "task_status": "subtask_completed",
            "subtask_result": "processed data",
        },
        goto="task_finisher",
        graph=Command.PARENT,
    )


def task_finisher(state: ParentNavigateState) -> dict:
    return {"messages": ["parent task finished"], "task_status": "completed"}


def build_navigation_subgraph():
    builder = StateGraph(ChildNavigateState)
    builder.add_node("data_processor", data_processor)
    builder.add_edge(START, "data_processor")
    builder.add_edge("data_processor", END)
    return builder.compile(name="navigation_subgraph")


def demo_call_subgraph_inside_node() -> None:
    print("=== demo_call_subgraph_inside_node ===")

    builder = StateGraph(ParentCallState)
    builder.add_node("run_subgraph_inside_node", run_subgraph_inside_node)
    builder.add_edge(START, "run_subgraph_inside_node")
    builder.add_edge("run_subgraph_inside_node", END)

    graph = builder.compile()
    print(graph.invoke({"topic": "LangGraph", "report": ""}))
    print()


def demo_add_subgraph_as_node() -> None:
    print("=== demo_add_subgraph_as_node ===")

    research_subgraph = build_research_subgraph()

    builder = StateGraph(ParentNodeState)
    builder.add_node("research_subgraph", research_subgraph)
    builder.add_node("finalize_report", finalize_report)
    builder.add_edge(START, "research_subgraph")
    builder.add_edge("research_subgraph", "finalize_report")
    builder.add_edge("finalize_report", END)

    graph = builder.compile()
    print(
        graph.invoke(
            {
                "topic": "LangGraph",
                "notes": [],
                "summary": "",
                "final_report": "",
            }
        )
    )
    print()


def demo_parent_navigation() -> None:
    print("=== demo_parent_navigation ===")

    builder = StateGraph(ParentNavigateState)
    builder.add_node("main_controller", main_controller)
    builder.add_node("task_finisher", task_finisher)
    builder.add_node("subgraph_node", build_navigation_subgraph())
    builder.add_edge(START, "main_controller")
    builder.add_edge("main_controller", "subgraph_node")
    builder.add_edge("task_finisher", END)

    graph = builder.compile()
    print(
        graph.invoke(
            {
                "messages": ["user starts task"],
                "task_status": "init",
                "subtask_result": "",
            }
        )
    )
    print()


def demo_stream_subgraph_updates() -> None:
    print("=== demo_stream_subgraph_updates ===")

    research_subgraph = build_research_subgraph()

    builder = StateGraph(ParentNodeState)
    builder.add_node("research_subgraph", research_subgraph)
    builder.add_node("finalize_report", finalize_report)
    builder.add_edge(START, "research_subgraph")
    builder.add_edge("research_subgraph", "finalize_report")
    builder.add_edge("finalize_report", END)

    graph = builder.compile()

    for chunk in graph.stream(
        {
            "topic": "subgraph streaming",
            "notes": [],
            "summary": "",
            "final_report": "",
        },
        stream_mode="updates",
        subgraphs=True,
    ):
        print(chunk)

    print()


def main() -> None:
    demo_call_subgraph_inside_node()
    demo_add_subgraph_as_node()
    demo_parent_navigation()
    demo_stream_subgraph_updates()


if __name__ == "__main__":
    main()
