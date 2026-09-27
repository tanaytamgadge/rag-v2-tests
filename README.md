# Document Q&A with grounded RAG

Ask questions about a PDF. Answers come only from that file. If the file does not contain the answer, the app says so.

## What this shows

- PDF loading, chunking, local embeddings, and Chroma retrieval
- A Groq model (`openai/gpt-oss-120b`) that sees only the retrieved passages
- A refusal when the context does not support an answer

## Stack

Python, LangChain, PyPDF, `BAAI/bge-small-en-v1.5` (local, no embedding API), Chroma, Groq.

## Run

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Put a Groq API key in `.env`:

```text
GROQ_API_KEY=your_key_here
```

Index any text-based PDF and start asking questions:

```powershell
python app.py path\to\your.pdf
```

The first run downloads the embedding model, so indexing can take a few minutes. Type a question and press Enter. Type `exit` to stop.

Try one question whose answer is in the PDF, then one that is not. The second reply should be:

```text
I cannot find the answer in the document.
```

Indexing another PDF replaces the previous index.

## Pipeline

1. `PyPDFLoader` reads one document per page.
2. `RecursiveCharacterTextSplitter` cuts pages into 800-character chunks with 120 characters of overlap. Page metadata stays on every chunk.
3. `BAAI/bge-small-en-v1.5` embeds each chunk locally.
4. Chroma stores the vectors on disk.
5. The question is embedded with the same model. The 4 nearest chunks go to Groq.
6. The prompt forbids outside knowledge and requires an explicit refusal.

Retrieved page labels are printed with the answer, so a wrong reply can be traced to retrieval or to the model.

## Design choices

- Embeddings run locally. The only secret is the Groq key, and `.env` is gitignored.
- Ingest and query live in separate modules. The answer step does not open the PDF.
- Re-indexing resets the Chroma collection first, so the same file is not stored twice.
- `reasoning_effort` is set to `low` and reasoning text is omitted from the reply.

## Limits

This is a single-document command-line app. A scanned PDF with no text layer produces no chunks. There is no reranking and no saved evaluation set. Grounding is enforced by the prompt.