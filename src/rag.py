from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.config import CHROMA_DIR, COLLECTION, LLM_MODEL, TOP_K, require_groq_key
from src.ingest import get_embeddings

NOT_FOUND = "I cannot find the answer in the document."


def get_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def format_docs(docs) -> str:
    return "\n\n".join(
        f"[page {doc.metadata.get('page_label', doc.metadata.get('page', '?'))}]\n"
        f"{doc.page_content}"
        for doc in docs
    )


def ask(question: str) -> dict:
    require_groq_key()
    store = get_store()
    docs = store.similarity_search(question, k=TOP_K)
    if not docs:
        return {"answer": NOT_FOUND, "sources": []}

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Answer using only the context below. "
            f"If the context does not contain the answer, reply exactly: {NOT_FOUND} "
            "Do not use outside knowledge.\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ])
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=1,
        reasoning_effort="low",
        model_kwargs={"include_reasoning": False},
    )      
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": format_docs(docs),
        "question": question,
    }).strip()

    sources = [
        {
            "page_label": doc.metadata.get("page_label", doc.metadata.get("page")),
            "source": doc.metadata.get("source"),
        }
        for doc in docs
    ]
    return {"answer": answer, "sources": sources}
