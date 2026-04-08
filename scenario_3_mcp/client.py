import asyncio
import mlflow
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from shared.llm import get_llm
from shared.questions import SCENARIO_QUESTIONS

load_dotenv()
mlflow.set_experiment("scenario_3_mcp")
mlflow.langchain.autolog()

MCP_SERVER_URL = "http://localhost:8000/sse"


async def run_all(questions: list[str]) -> list[str]:
    client = MultiServerMCPClient(
        {"poc-server": {"url": MCP_SERVER_URL, "transport": "sse"}}
    )
    tools = await client.get_tools()
    print("=" * 60)
    for tool in tools:
        print(f"Tool name: {tool.name}")
        print(f"Tool description: {tool.description}")
        print(f"Tool args_schema: {tool.args_schema}")
        print("=" * 60)
    llm = get_llm()
    agent = create_agent(llm, tools=tools)
    answers = []
    for question in questions:
        result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
        answers.append(result["messages"][-1].content)
    return answers


if __name__ == "__main__":
    answers = asyncio.run(run_all(SCENARIO_QUESTIONS))
    for q, a in zip(SCENARIO_QUESTIONS, answers):
        print(f"Q: {q}")
        print(f"A: {a}")
        print("=" * 60)
