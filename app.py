import sys

from src.ingest import ingest_pdf
from src.rag import ask


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python app.py path\\to\\file.pdf")
        raise SystemExit(1)

    pdf_path = sys.argv[1]
    print(f"Reading and indexing {pdf_path}")
    print("The first run downloads the local embedding model. That can take a few minutes.")
    count = ingest_pdf(pdf_path)
    print(f"Stored {count} chunks. Ask a question, or type exit.")

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        result = ask(question)
        print("\n" + result["answer"])
        if result["sources"]:
            pages = ", ".join(str(item["page_label"]) for item in result["sources"])
            print(f"Pages searched: {pages}")


if __name__ == "__main__":
    main()