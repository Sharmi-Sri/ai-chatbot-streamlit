"""Console version of the RAG Q&A assistant (Task 2 deliverable).

Usage:  python cli.py path/to/file.pdf [more files...]
        python cli.py            # reuse documents already stored in ChromaDB
"""
import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

from rag_pipeline import NOT_FOUND, RAGPipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask questions about your documents")
    ap.add_argument("files", nargs="*", help="PDF / TXT / MD files to ingest")
    args = ap.parse_args()

    rag = RAGPipeline()
    if args.files:
        pages, chunks = rag.ingest(args.files)
        print(f"Ingested {pages} pages -> {chunks} chunks.")
    if rag.count() == 0:
        sys.exit("No documents loaded. Pass at least one file.")

    print("Ask a question (type 'exit' to quit).\n")
    history: list[dict] = []
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        stream, sources = rag.ask(q, history)
        print("AI: ", end="", flush=True)
        answer = ""
        for token in stream:
            print(token, end="", flush=True)
            answer += token
        print()
        if sources and not answer.startswith(NOT_FOUND):
            print("Sources: " + "; ".join(rag.cited_only(answer, sources)))
        print()
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": answer}]


if __name__ == "__main__":
    main()
