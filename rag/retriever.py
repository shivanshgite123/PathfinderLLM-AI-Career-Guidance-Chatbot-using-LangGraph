"""
rag/retriever.py
─────────────────────────────────────────────────────────────
Builds (or loads from disk) a FAISS vector store and exposes a
retrieve() function used by the RAG node in the LangGraph workflow.
─────────────────────────────────────────────────────────────
"""

import os
from typing import List

from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from models.llm import get_embeddings
from rag.loader import load_documents

# ── Persistence path ──────────────────────────────────────────────────────────
_BASE_DIR   = os.path.dirname(__file__)
INDEX_PATH  = os.path.join(_BASE_DIR, "..", "data", "faiss_index")

# Singleton – built once per process
_vectorstore: FAISS | None = None


def _build_vectorstore() -> FAISS:
    """
    Build a FAISS vector store from documents in data/.
    If a persisted index already exists it is loaded instead of rebuilt.

    Returns:
        A ready-to-query FAISS vectorstore.
    """
    embeddings = get_embeddings()
    abs_index  = os.path.abspath(INDEX_PATH)

    if os.path.exists(abs_index):
        print(f"[retriever] Loading existing FAISS index from {abs_index}")
        return FAISS.load_local(
            abs_index,
            embeddings,
            allow_dangerous_deserialization=True,   # safe – we wrote this index
        )

    print("[retriever] Building new FAISS index …")
    docs = load_documents()

    if not docs:
        raise RuntimeError(
            "No documents found to index. "
            "Add .txt or .pdf files to the data/ directory."
        )

    vs = FAISS.from_documents(docs, embeddings)

    # Persist for faster restarts
    os.makedirs(abs_index, exist_ok=True)
    vs.save_local(abs_index)
    print(f"[retriever] FAISS index saved to {abs_index}")
    return vs


def get_vectorstore() -> FAISS:
    """Return the singleton vectorstore, initialising it on first call."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = _build_vectorstore()
    return _vectorstore


def retrieve(query: str, k: int = 4) -> str:
    """
    Retrieve the top-k most relevant document chunks for `query`.

    Args:
        query: Natural language search string.
        k:     Number of chunks to return (default 4).

    Returns:
        A single concatenated string of relevant context passages.
    """
    try:
        vs     = get_vectorstore()
        docs: List[Document] = vs.similarity_search(query, k=k)
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        return context or "No relevant context found."
    except Exception as exc:
        # Graceful degradation – the LLM will still answer from training data
        print(f"[retriever] WARNING – retrieval failed: {exc}")
        return "Career knowledge retrieval temporarily unavailable."


def reset_index() -> None:
    """
    Force-rebuild the FAISS index (useful after adding new documents).
    Deletes the persisted index and clears the in-memory singleton.
    """
    import shutil
    global _vectorstore
    abs_index = os.path.abspath(INDEX_PATH)
    if os.path.exists(abs_index):
        shutil.rmtree(abs_index)
        print(f"[retriever] Deleted existing index at {abs_index}")
    _vectorstore = None
    print("[retriever] Index reset – will rebuild on next retrieve() call.")
