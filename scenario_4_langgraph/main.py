import asyncio

import mlflow
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

from scenario_4_langgraph.graph import build_graph
from scenario_4_langgraph.nodes import classify_route
from shared.llm import get_llm
from shared.questions import SCENARIO_QUESTIONS

load_dotenv()
mlflow.set_experiment("scenario_4_langgraph_fixed")
mlflow.langchain.autolog()

MCP_SERVER_URL = "http://localhost:8000/sse"


async def run_all(questions: list[str]) -> list[str]:

    ## connect to the MCP server and get the tools
    client = MultiServerMCPClient(
        {"poc-server": {"url": MCP_SERVER_URL, "transport": "sse"}}
    )
    tools = await client.get_tools()

    ## split the tools into data_mapping and general
    data_mapping_tools = [tool for tool in tools if tool.name == "data_mapping"]
    general_tools = [tool for tool in tools if tool.name != "data_mapping"]

    ## build the graph
    llm = get_llm()
    app = build_graph(llm, data_mapping_tools, general_tools)

    ## get the answers
    answers: list[str] = []
    for question in questions:
        result = await app.ainvoke(
            {"question": question, "route": classify_route(question), "answer": ""}
        )
        answers.append(result["answer"])
    return answers


if __name__ == "__main__":
    answers = asyncio.run(run_all(SCENARIO_QUESTIONS))
    for q, a in zip(SCENARIO_QUESTIONS, answers):
        print(f"Q: {q}")
        print(f"A: {a}")
        print("=" * 60)
