import ast
import json

import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import BaseModel, Field

from shared.rag import get_retriever

load_dotenv()

mcp = FastMCP("MCP PoC Server")


class DataMappingInput(BaseModel):
    """Payload for the `data` argument when calling this tool."""

    source: str = Field(
        description="Required. Logical source id (e.g. payload name or input schema)."
    )
    target: str = Field(
        description="Required. Logical target id (e.g. API body name or output schema)."
    )
    version: int = Field(
        default=1,
        description="Optional (defaults to 1). Mapping contract version between source and target.",
    )


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


@mcp.tool()
def data_mapping(data: DataMappingInput) -> str:
    """Dict→dict mapping example: echoes on the server whatever you send.

    Pass a single `data` argument (JSON object) with:
    - `source` (string, required): logical source of the data.
    - `target` (string, required): logical target (e.g. API body shape).
    - `version` (integer, optional): defaults to 1 if omitted.

    Example: {"source": "order_payload", "target": "api_body", "version": 1}
    """
    payload = data.model_dump()
    print(payload)
    return json.dumps(payload, ensure_ascii=False)


if __name__ == "__main__":
    print("Starting MCP server on http://localhost:8000")
    print("Tools available: rag_search, get_exchange_rate, calculate, data_mapping")
    mcp.run(transport="sse", host="0.0.0.0", port=8000)
