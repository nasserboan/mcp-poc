"""
Scenario 1 — LangChain Chain

The RAG retriever is hardwired into the chain.
It ALWAYS runs, regardless of the question.
The LLM has no choice — it receives the retrieved context every time.
"""
import mlflow
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from shared.llm import get_llm
from shared.rag import get_retriever

load_dotenv()
mlflow.set_experiment("scenario_1_chain")
mlflow.langchain.autolog()


def run(question: str) -> str:
    retriever = get_retriever()
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(
        "Answer the question using the context below.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(question)


if __name__ == "__main__":
    questions = [
        "What are the main challenges in building LLM agents?",
        "What is the capital of France?",  # RAG still runs even for this
    ]
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {run(q)}")
        print("=" * 60)
