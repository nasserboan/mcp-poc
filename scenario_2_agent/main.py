import mlflow
from dotenv import load_dotenv
from langchain.agents import create_agent

from shared.llm import get_llm
from shared.questions import SCENARIO_QUESTIONS
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
    for q in SCENARIO_QUESTIONS:
        print(f"Q: {q}")
        print(f"A: {run(q)}")
        print("=" * 60)
