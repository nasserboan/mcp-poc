"""
Scenario 3 — MCP Client

A LangGraph agent that connects to the MCP server at runtime.
The agent has NO hardcoded tools — it discovers them from the server via MCP.

Graph: START → agent → (tool_calls?) → tools → agent → ... → END
       (identical to scenario 2, but tools come from the MCP server)

Requires the MCP server (server.py) to be running before executing this script.
"""
import asyncio
import mlflow
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from shared.llm import get_llm

load_dotenv()
mlflow.set_experiment("scenario_3_mcp")
mlflow.langchain.autolog()

MCP_SERVER_URL = "http://localhost:8000/sse"


async def run_all(questions: list[str]) -> list[str]:
    client = MultiServerMCPClient(
        {"poc-server": {"url": MCP_SERVER_URL, "transport": "sse"}}
    )
    tools = await client.get_tools()
    llm = get_llm()
    agent = create_agent(llm, tools=tools)
    answers = []
    for question in questions:
        result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
        answers.append(result["messages"][-1].content)
    return answers


if __name__ == "__main__":
    questions = [
        "What does the literature say about LLM agent tool use?",  # → rag_search
        "What is the exchange rate from USD to BRL today?",         # → get_exchange_rate
        "What is 1337 * 42?",                                       # → calculate
    ]
    answers = asyncio.run(run_all(questions))
    for q, a in zip(questions, answers):
        print(f"Q: {q}")
        print(f"A: {a}")
        print("=" * 60)
