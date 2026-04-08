import ast
import os
import sys

import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP

from shared.rag import get_retriever

load_dotenv()

mcp = FastMCP("MCP PoC Server")


@mcp.tool()
def rag_search(query: str) -> str:
    """Search arXiv papers about LLM agents. Use for research questions about LLM agents."""
    retriever = get_retriever()
    docs = retriever.invoke(query)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


@mcp.tool()
def get_exchange_rate(base: str, target: str) -> str:
    """Get the current exchange rate between two currencies. Example: base='USD', target='BRL'."""
    response = httpx.get(f"https://open.er-api.com/v6/latest/{base}")
    data = response.json()
    rate = data["rates"].get(target.upper())
    if rate is None:
        return f"Currency '{target}' not found."
    return f"1 {base.upper()} = {rate} {target.upper()}"


@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely. Example: '2 + 2 * 10'."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = eval(compile(tree, "<string>", "eval"))  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


if __name__ == "__main__":
    print("Starting MCP server on http://localhost:8000")
    print("Tools available: rag_search, get_exchange_rate, calculate")
    mcp.run(transport="sse", host="0.0.0.0", port=8000)
