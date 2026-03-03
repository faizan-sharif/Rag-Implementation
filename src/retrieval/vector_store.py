"""
vector_store.py
===============
ChromaDB build/load + Hybrid BM25 + Dense retriever with RRF re-ranking.

Advanced retrieval pipeline:
  Step 1 - Dense retrieval   : ChromaDB MMR (semantic similarity + diversity)
  Step 2 - Sparse retrieval  : BM25 keyword match (catches exact codes like B011)
  Step 3 - RRF fusion        : Reciprocal Rank Fusion merges both result lists
  Step 4 - Type boosting     : table_row chunks get +score bonus (most precise)

This hybrid approach is far more accurate than dense-only for financial tables
where users query by exact head names, codes, and fiscal year column names.
"""

import os
import math
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from config.settings import (
    RETRIEVER_K, RETRIEVER_FETCH_K, RETRIEVER_LAMBDA, BM25_K
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Module-level BM25 index (rebuilt on each ingest, loaded from memory on query)
_bm25_index: BM25Okapi | None = None
_bm25_docs:  List[Document]   = []


def build_vector_store(
    chunks: List[Document],
    embeddings,
    persist_dir: str,
) -> Chroma:
    global _bm25_index, _bm25_docs

    logger.info(f"Building ChromaDB at: {persist_dir}")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    # Build BM25 index over same chunks
    _bm25_docs  = chunks
    _bm25_index = _build_bm25(chunks)
    logger.info(f"ChromaDB + BM25 index built ({len(chunks)} chunks).")
    return vectorstore


def load_vector_store(embeddings, persist_dir: str) -> Chroma | None:
    global _bm25_index, _bm25_docs

    if not os.path.exists(persist_dir):
        logger.warning(f"ChromaDB not found at: {persist_dir}")
        return None

    logger.info(f"Loading ChromaDB from: {persist_dir}")
    vs = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

    # Rebuild BM25 from persisted docs
    all_docs = vs.get(include=["documents", "metadatas"])
    if all_docs and all_docs.get("documents"):
        _bm25_docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(
                all_docs["documents"],
                all_docs["metadatas"] or [{}] * len(all_docs["documents"]),
            )
        ]
        _bm25_index = _build_bm25(_bm25_docs)
        logger.info(f"BM25 index rebuilt from {len(_bm25_docs)} stored chunks.")

    return vs


def get_hybrid_retriever(vectorstore: Chroma):
    """
    Returns a callable retriever using Hybrid RRF.
    Compatible with LangChain LCEL (accepts a string, returns List[Document]).
    """
    def retrieve(query: str) -> List[Document]:
        return hybrid_retrieve(query, vectorstore)
    return retrieve


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid retrieval internals
# ─────────────────────────────────────────────────────────────────────────────

def hybrid_retrieve(query: str, vectorstore: Chroma) -> List[Document]:
    # 1. Dense: MMR from ChromaDB
    dense_docs = vectorstore.max_marginal_relevance_search(
        query,
        k=RETRIEVER_K,
        fetch_k=RETRIEVER_FETCH_K,
        lambda_mult=RETRIEVER_LAMBDA,
    )

    # 2. Sparse: BM25
    sparse_docs = _bm25_search(query, k=BM25_K)

    # 3. RRF fusion
    fused = _reciprocal_rank_fusion(dense_docs, sparse_docs, k=60)

    # 4. Type boost: promote table_row chunks (most precise for cell lookup)
    boosted = _type_boost(fused)

    logger.info(
        f"Hybrid retrieval: {len(dense_docs)} dense + {len(sparse_docs)} BM25 "
        f"-> {len(boosted)} after RRF+boost"
    )
    return boosted[:RETRIEVER_K]


def _build_bm25(docs: List[Document]) -> BM25Okapi:
    tokenised = [doc.page_content.lower().split() for doc in docs]
    return BM25Okapi(tokenised)


def _bm25_search(query: str, k: int) -> List[Document]:
    if _bm25_index is None or not _bm25_docs:
        return []
    tokens = query.lower().split()
    scores = _bm25_index.get_scores(tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [_bm25_docs[i] for i in top_idx if scores[i] > 0]


def _reciprocal_rank_fusion(
    list1: List[Document],
    list2: List[Document],
    k: int = 60,
) -> List[Document]:
    """
    RRF score = sum(1 / (rank + k)) across retriever lists.
    Deduplication by page_content hash.
    """
    scores: dict[str, Tuple[float, Document]] = {}

    for rank, doc in enumerate(list1, start=1):
        key = doc.page_content[:200]
        prev_score = scores.get(key, (0.0, doc))[0]
        scores[key] = (prev_score + 1.0 / (rank + k), doc)

    for rank, doc in enumerate(list2, start=1):
        key = doc.page_content[:200]
        prev_score = scores.get(key, (0.0, doc))[0]
        scores[key] = (prev_score + 1.0 / (rank + k), doc)

    sorted_docs = sorted(scores.values(), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in sorted_docs]


def _type_boost(docs: List[Document]) -> List[Document]:
    """
    Reorder: table_row first, then table, then text.
    Within each group maintain RRF order.
    """
    rows   = [d for d in docs if d.metadata.get("type") == "table_row"]
    tables = [d for d in docs if d.metadata.get("type") == "table"]
    texts  = [d for d in docs if d.metadata.get("type") == "text"]
    return rows + tables + texts