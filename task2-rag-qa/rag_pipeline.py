"""Task 2: Document Q&A with RAG and vector embeddings.

Flow: load PDF/text -> recursive chunking -> Gemini embeddings -> ChromaDB
      -> retrieve top-k chunks -> answer strictly from context, with page citations.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Iterator, Sequence

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001")
NOT_FOUND = "I couldn't find that in the loaded documents."

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a document question-answering assistant. Answer ONLY using the "
            "context below. Do not use outside knowledge. If the context does not "
            "contain the answer, reply exactly: {not_found}\n"
            "Be concise and accurate. Each context block starts with its source tag "
            "like [file.pdf p.3]; mention these tags inline when you use a block.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's latest question as a single standalone question, "
            "using the chat history to resolve pronouns and references. "
            "If it is already standalone, return it unchanged. "
            "Output only the question.",
        ),
        ("human", "Chat history:\n{history}\n\nLatest question: {question}"),
    ]
)


def load_documents(path: str | Path) -> list[Document]:
    """Load a PDF or text file; pages are 1-indexed in metadata['page']."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        docs = PyPDFLoader(str(p)).load()
    elif suffix in {".txt", ".md"}:
        docs = TextLoader(str(p), encoding="utf-8").load()
    else:
        raise ValueError(f"Unsupported file type: {p.name} (use .pdf, .txt or .md)")
    for d in docs:
        d.metadata["source"] = p.name
        if "page" in d.metadata:
            d.metadata["page"] = int(d.metadata["page"]) + 1
    return docs


class RAGPipeline:
    def __init__(
        self,
        persist_dir: str = "chroma_db",
        collection: str = "docs",
        k: int = 4,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.persist_dir = persist_dir
        self.collection = collection
        self.k = k
        self.embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
        self.llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.store = self._make_store()

    def _make_store(self) -> Chroma:
        return Chroma(
            collection_name=self.collection,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )

    # ---------- ingestion ----------
    def ingest(self, paths: Sequence[str | Path]) -> tuple[int, int]:
        """Load, chunk, embed and store files. Returns (pages, chunks)."""
        pages = 0
        chunks: list[Document] = []
        for path in paths:
            docs = load_documents(path)
            pages += len(docs)
            chunks.extend(self.splitter.split_documents(docs))
        chunks = [c for c in chunks if c.page_content.strip()]
        if not chunks:
            raise ValueError("No extractable text found (scanned PDFs need OCR).")

        ids = [
            hashlib.sha1(
                f"{c.metadata.get('source')}|{c.metadata.get('page')}|{c.page_content}".encode()
            ).hexdigest()
            for c in chunks
        ]
        batch = 50  # keeps us inside free-tier embedding request limits
        for i in range(0, len(chunks), batch):
            self.store.add_documents(chunks[i : i + batch], ids=ids[i : i + batch])
        return pages, len(chunks)

    def count(self) -> int:
        return self.store._collection.count()

    def reset(self) -> None:
        self.store.delete_collection()
        self.store = self._make_store()

    # ---------- retrieval + generation ----------
    def retrieve(self, question: str) -> list[Document]:
        if self.count() == 0:
            return []
        return self.store.similarity_search(question, k=self.k)

    def condense_question(self, question: str, history: list[dict] | None) -> str:
        if not history:
            return question
        recent = history[-6:]
        text = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
        chain = CONDENSE_PROMPT | self.llm | StrOutputParser()
        return chain.invoke({"history": text, "question": question}).strip() or question

    @staticmethod
    def _format_context(docs: list[Document]) -> str:
        blocks = []
        for d in docs:
            page = d.metadata.get("page")
            tag = f"{d.metadata.get('source')} p.{page}" if page else d.metadata.get("source")
            blocks.append(f"[{tag}]\n{d.page_content}")
        return "\n\n".join(blocks)

    @staticmethod
    def sources(docs: list[Document]) -> list[str]:
        seen: dict[tuple, str] = {}
        for d in docs:
            src, page = d.metadata.get("source"), d.metadata.get("page")
            seen[(src, page or 0)] = f"{src}, page {page}" if page else f"{src}"
        return [seen[k] for k in sorted(seen, key=lambda x: (str(x[0]), x[1]))]

    @staticmethod
    def cited_only(answer: str, sources: list[str]) -> list[str]:
        """Keep only sources the answer cites via [file p.N] tags.

        Falls back to all retrieved sources if the answer has no usable tags.
        """
        cited = set()
        for block in re.findall(r"\[([^\]]*)\]", answer):
            for name, page in re.findall(r"([^,;\[\]]+?)\s+p\.(\d+)", block):
                cited.add(f"{name.strip()}, page {page}")
        picked = [s for s in sources if s in cited]
        return picked or sources

    def _stream(self, question: str, docs: list[Document]) -> Iterator[str]:
        if not docs:
            yield NOT_FOUND
            return
        chain = ANSWER_PROMPT | self.llm | StrOutputParser()
        yield from chain.stream(
            {
                "context": self._format_context(docs),
                "question": question,
                "not_found": NOT_FOUND,
            }
        )

    def ask(
        self, question: str, history: list[dict] | None = None
    ) -> tuple[Iterator[str], list[str]]:
        """Returns (token_stream, source_citations)."""
        standalone = self.condense_question(question, history)
        docs = self.retrieve(standalone)
        return self._stream(standalone, docs), self.sources(docs)
