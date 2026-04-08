from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings

load_dotenv()

CHROMA_DIR = "./chroma_db"


def get_retriever():
    embeddings = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": 3})


@tool
def rag_search(query: str) -> str:
    """Search arXiv papers about LLM agents. Use this when asked about research, papers, or literature on LLM agents."""
    retriever = get_retriever()
    docs = retriever.invoke(query)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)
