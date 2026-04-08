from typing import Awaitable, Callable, Literal, TypedDict

from langchain.agents import create_agent
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool


class GraphState(TypedDict):
    question: str
    route: Literal["data_mapping", "general"]
    answer: str


def classify_route(question: str) -> Literal["data_mapping", "general"]:
    text = question.lower()
    mapping_keywords = ("data mapping", "data_mapping", "mapping", "schema")
    if any(keyword in text for keyword in mapping_keywords):
        return "data_mapping"
    return "general"


def build_nodes(
    llm: Runnable, data_mapping_tools: list[BaseTool], general_tools: list[BaseTool]
) -> tuple[
    Callable[[GraphState], Awaitable[GraphState]],
    Callable[[GraphState], Awaitable[GraphState]],
    Callable[[GraphState], Awaitable[GraphState]],
]:
    # Two agents, each with a different MCP tool subset.
    data_mapping_agent = create_agent(llm, tools=data_mapping_tools)
    general_agent = create_agent(llm, tools=general_tools)

    async def classify_node(state: GraphState) -> GraphState:
        return {**state, "route": classify_route(state["question"])}

    async def data_mapping_node(state: GraphState) -> GraphState:
        result = await data_mapping_agent.ainvoke(
            {"messages": [{"role": "user", "content": state["question"]}]}
        )
        return {**state, "answer": result["messages"][-1].content}

    async def general_node(state: GraphState) -> GraphState:
        result = await general_agent.ainvoke(
            {"messages": [{"role": "user", "content": state["question"]}]}
        )
        return {**state, "answer": result["messages"][-1].content}

    return classify_node, data_mapping_node, general_node
