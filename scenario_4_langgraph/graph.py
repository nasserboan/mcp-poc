from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from scenario_4_langgraph.nodes import GraphState, build_nodes


def build_graph(llm: Runnable, data_mapping_tools: list[BaseTool], general_tools: list[BaseTool]):
    classify_node, data_mapping_node, general_node = build_nodes(
        llm, data_mapping_tools, general_tools
    )

    graph = StateGraph(GraphState)
    graph.add_node("classify", classify_node)
    graph.add_node("data_mapping_path", data_mapping_node)
    graph.add_node("general_path", general_node)

    def route_by_state(state: GraphState) -> str:
        return state["route"]

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route_by_state,
        {
            "data_mapping": "data_mapping_path",
            "general": "general_path",
        },
    )
    graph.add_edge("data_mapping_path", END)
    graph.add_edge("general_path", END)

    return graph.compile()
