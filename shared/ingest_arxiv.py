import arxiv
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHROMA_DIR = "./chroma_db"
QUERY = "LLM agents tool use reasoning"
MAX_RESULTS = 5


def ingest():
    print(f"Fetching {MAX_RESULTS} papers from arXiv...")
    client = arxiv.Client()
    search = arxiv.Search(query=QUERY, max_results=MAX_RESULTS)
    results = list(client.results(search))

    docs = [
        Document(
            page_content=f"{paper.title}\n\n{paper.summary}",
            metadata={"title": paper.title, "url": paper.entry_id},
        )
        for paper in results
    ]

    print(f"Ingesting into ChromaDB at {CHROMA_DIR}...")
    embeddings = OpenAIEmbeddings()
    Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DIR)
    print(f"Done. Ingested {len(docs)} papers:")
    for doc in docs:
        print(f"  - {doc.metadata['title']}")


if __name__ == "__main__":
    ingest()
