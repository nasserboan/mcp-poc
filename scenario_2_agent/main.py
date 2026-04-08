"""
Scenario 2 — LangGraph Agent

The LLM has the RAG tool available but decides whether to use it.
Graph: START → agent → (tool_calls?) → tools → agent → ... → END

The agent calls RAG only when it judges the tool is needed.
Compare MLflow traces between a research question and a simple question.
"""
import mlflow
from dotenv import load_dotenv
from langchain.agents import create_agent

from shared.llm import get_llm
from shared.rag import rag_search

load_dotenv()
mlflow.set_experiment("scenario_2_agent")
mlflow.langchain.autolog()


def run(question: str) -> str:
    llm = get_llm()
    agent = create_agent(llm, tools=[rag_search])
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content


if __name__ == "__main__":
    questions = [
        "What does the literature say about LLM agent memory?",  # Should trigger RAG
        "What is the capital of France?",                         # Should NOT trigger RAG
    ]
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {run(q)}")
        print("=" * 60)
