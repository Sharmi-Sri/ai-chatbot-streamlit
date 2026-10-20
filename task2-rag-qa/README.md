# Task 2: Document Q&A Assistant with RAG & Vector Embeddings

A Retrieval-Augmented Generation (RAG) pipeline that answers questions **strictly from loaded PDF/text documents**, with source page citations.

## How it works

```
PDF/TXT -> PyPDFLoader/TextLoader -> RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
        -> Gemini embeddings -> ChromaDB (persisted in ./chroma_db)
question -> (rewrite using chat history) -> top-4 similarity search
        -> Gemini answers only from retrieved context -> answer + page citations
```

| File | Purpose |
|---|---|
| `rag_pipeline.py` | RAG engine: load, chunk, embed, store, retrieve, answer, cite |
| `cli.py` | Console Q&A loop |
| `sample_handbook.pdf` | 4-page fictional handbook for testing |
| `requirements.txt` | Dependencies |
| `.env.example` | Template for your API key |

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (Mac/Linux: source venv/bin/activate)
python -m pip install -r requirements.txt
copy .env.example .env           # Mac/Linux: cp .env.example .env
```

Put your free Gemini key (https://aistudio.google.com/apikey) in `.env`. If the default model is unavailable, set `GEMINI_CHAT_MODEL` in `.env` (for example `GEMINI_CHAT_MODEL=gemini-3.8-flash`).

On Windows, ChromaDB needs the Microsoft Visual C++ Redistributable (https://aka.ms/vs/17/release/vc_redist.x64.exe).

## Run

```bash
python cli.py sample_handbook.pdf
```

## Example session

```
You: How many days of annual leave do employees get?
AI: Every full-time employee receives 22 days of paid annual leave per calendar year [sample_handbook.pdf p.2].
Sources: sample_handbook.pdf, page 2

You: What is the capital of France?
AI: I couldn't find that in the loaded documents.
```

## Design notes

- **Strictly grounded:** the prompt forbids outside knowledge; when the context lacks the answer, the assistant says so.
- **Page citations:** each chunk keeps `source` and 1-indexed `page` metadata; only pages the answer cites are listed.
- **Follow-up questions:** the latest question is rewritten into a standalone one using recent history before retrieval.
- **LCEL instead of `RetrievalQA`:** `RetrievalQA` is deprecated in current LangChain, so retrieval and the answer chain are composed directly, which also enables token streaming.
- **Idempotent ingestion:** chunk IDs are content hashes, so re-loading a file doesn't create duplicates.
- Scanned (image-only) PDFs need OCR first.
