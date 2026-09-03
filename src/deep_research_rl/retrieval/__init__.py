"""Uniform local and production retrieval backends with integrity checks."""

from __future__ import annotations

from pathlib import Path

from deep_research_rl.core.protocols import Retriever
from deep_research_rl.retrieval.bm25 import (
    BM25Retriever,
    build_bm25_index,
    load_bm25_index,
    verify_bm25_index,
)
from deep_research_rl.retrieval.faiss_bge import (
    AgentR1FaissBGEToolAdapter,
    BGEEncoder,
    FaissBGERetriever,
    build_faiss_bge_index,
    load_faiss_bge_index,
    verify_faiss_bge_index,
)
from deep_research_rl.retrieval.index import manifest_backend


def load_retriever(
    index_dir: str | Path,
    corpus_path: str | Path,
    *,
    top_k: int = 5,
    device: str = "cpu",
) -> Retriever:
    """Load the backend declared by a verified index manifest."""

    backend = manifest_backend(index_dir)
    if backend == "bm25":
        return load_bm25_index(index_dir, corpus_path, top_k=top_k)
    if backend == "faiss_bge":
        return load_faiss_bge_index(index_dir, corpus_path, top_k=top_k, device=device)
    raise ValueError(f"unsupported retrieval backend: {backend}")


def verify_index(index_dir: str | Path, corpus_path: str | Path) -> dict[str, object]:
    """Verify the backend declared by an index manifest without loading a BGE model."""

    backend = manifest_backend(index_dir)
    if backend == "bm25":
        return verify_bm25_index(index_dir, corpus_path)
    if backend == "faiss_bge":
        return verify_faiss_bge_index(index_dir, corpus_path)
    raise ValueError(f"unsupported retrieval backend: {backend}")


__all__ = [
    "AgentR1FaissBGEToolAdapter",
    "BGEEncoder",
    "BM25Retriever",
    "FaissBGERetriever",
    "build_bm25_index",
    "build_faiss_bge_index",
    "load_bm25_index",
    "load_faiss_bge_index",
    "load_retriever",
    "verify_bm25_index",
    "verify_faiss_bge_index",
    "verify_index",
]
