from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION,
    EMBEDDING_MODEL,
)


def load_pdf(path: str) -> list[Document]:
    return PyPDFLoader(path).load()


def split_pages(pages: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(pages)


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def ingest_pdf(path: str) -> int:
    pages = load_pdf(path)
    chunks = [chunk for chunk in split_pages(pages) if chunk.page_content.strip()]
    if not chunks:
        raise RuntimeError(
            "No text extracted. Use a PDF with a real text layer, not a scan."
        )

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    store.reset_collection()
    store.add_documents(chunks)
    return len(chunks)
