"""
vector_store.py
===============
ChromaDB + Hybrid BM25 + Dense retriever with RRF re-ranking.

Fixes in this version:
  - BM25 index persisted to disk (JSON) so query sessions work correctly
  - Query expansion for common budget abbreviations (ADP, PSDP, etc.)
  - Bigram tokenization for better multi-word term matching
  - Increased candidate pool before final ranking
"""

import os
import json
import pickle
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from config.settings import (
    RETRIEVER_K, RETRIEVER_FETCH_K, RETRIEVER_LAMBDA, BM25_K, CHROMA_DIR
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

_bm25_index: BM25Okapi | None = None
_bm25_docs:  List[Document]   = []

BM25_CACHE_PATH = os.path.join(CHROMA_DIR, "bm25_index.pkl")

# ── Query expansion dictionary ─────────────────────────────────────────────
# Maps abbreviations/aliases -> expanded search terms added to query
QUERY_EXPANSIONS = {
    "adp":      ["annual development program", "development program", "adp"],
    "psdp":     ["public sector development program", "psdp"],
    "gdp":      ["gross domestic product", "gdp"],
    "pfc":      ["provincial finance commission", "pfc"],
    "g.o.p":    ["government of punjab", "gop"],
    "gop":      ["government of punjab"],
    "revenue":  ["revenue receipts", "revenue expenditure", "revenue"],
    "current":  ["current expenditure", "current revenue"],
    "capital":  ["capital expenditure", "capital receipts", "capital"],
    "pension":  ["pension", "pensionary charges", "superannuation"],
    "salary":   ["pay", "salary", "salaries", "employees"],
    "police":   ["police", "law enforcement"],
    "health":   ["health", "hospital services", "medical"],
    "education":["education", "schools", "higher education"],
    "transfer": ["transfer payment", "grants", "subsidies"],
}


def expand_query(query: str) -> str:
    """
    Expand query abbreviations to full terms for better BM25 matching.
    e.g. "ADP 2025-26" -> "ADP annual development program 2025-26"
    """
    tokens = query.lower().split()
    extra = []
    for token in tokens:
        clean = token.strip("().,?")
        if clean in QUERY_EXPANSIONS:
            extra.extend(QUERY_EXPANSIONS[clean])
    if extra:
        expanded = query + " " + " ".join(extra)
        logger.info(f"Query expanded: {query!r} -> {expanded!r}")
        return expanded
    return query


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

    # Build and persist BM25 index
    _bm25_docs  = chunks
    _bm25_index = _build_bm25(chunks)
    _save_bm25(chunks, persist_dir)
    logger.info(f"ChromaDB + BM25 index built and saved ({len(chunks)} chunks).")
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

    # Try loading persisted BM25 first (fast)
    if _load_bm25(persist_dir):
        logger.info(f"BM25 index loaded from disk ({len(_bm25_docs)} chunks).")
    else:
        # Fallback: rebuild from ChromaDB documents
        logger.info("BM25 cache not found, rebuilding from ChromaDB...")
        all_docs = vs.get(include=["documents", "metadatas"])
        if all_docs and all_docs.get("documents"):
            _bm25_docs = [
                Document(page_content=text, metadata=meta or {})
                for text, meta in zip(
                    all_docs["documents"],
                    all_docs["metadatas"] or [{}] * len(all_docs["documents"]),
                )
            ]
            _bm25_index = _build_bm25(_bm25_docs)
            _save_bm25(_bm25_docs, persist_dir)
            logger.info(f"BM25 rebuilt and saved ({len(_bm25_docs)} chunks).")

    return vs


def get_hybrid_retriever(vectorstore: Chroma):
    def retrieve(query: str) -> List[Document]:
        return hybrid_retrieve(query, vectorstore)
    return retrieve


# ─────────────────────────────────────────────────────────────────────────────

def hybrid_retrieve(query: str, vectorstore: Chroma) -> List[Document]:
    expanded_query = expand_query(query)

    # 1. Dense MMR retrieval
    dense_docs = vectorstore.max_marginal_relevance_search(
        expanded_query,
        k=RETRIEVER_K,
        fetch_k=RETRIEVER_FETCH_K,
        lambda_mult=RETRIEVER_LAMBDA,
    )

    # 2. BM25 sparse retrieval (on expanded query)
    sparse_docs = _bm25_search(expanded_query, k=BM25_K)

    # 3. Also run BM25 on original query (catches exact match)
    if expanded_query != query:
        sparse_docs_orig = _bm25_search(query, k=BM25_K // 2)
        # Merge without duplication
        seen = {d.page_content[:200] for d in sparse_docs}
        for d in sparse_docs_orig:
            if d.page_content[:200] not in seen:
                sparse_docs.append(d)

    # 4. RRF fusion
    fused = _reciprocal_rank_fusion(dense_docs, sparse_docs, k=60)

    # 5. Type boost
    boosted = _type_boost(fused)

    logger.info(
        f"Hybrid retrieval: {len(dense_docs)} dense + {len(sparse_docs)} BM25 "
        f"-> {len(boosted)} after RRF+boost (returning top {RETRIEVER_K})"
    )
    return boosted[:RETRIEVER_K]


def _build_bm25(docs: List[Document]) -> BM25Okapi:
    """Tokenize with unigrams + bigrams for better phrase matching."""
    tokenised = []
    for doc in docs:
        words = doc.page_content.lower().split()
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
        tokenised.append(words + bigrams)
    return BM25Okapi(tokenised)


def _bm25_search(query: str, k: int) -> List[Document]:
    if _bm25_index is None or not _bm25_docs:
        return []
    words = query.lower().split()
    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
    tokens = words + bigrams
    scores = _bm25_index.get_scores(tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [_bm25_docs[i] for i in top_idx if scores[i] > 0]


def _save_bm25(docs: List[Document], persist_dir: str) -> None:
    """Save BM25 docs to disk so query sessions can reload without re-ingesting."""
    try:
        cache_path = os.path.join(persist_dir, "bm25_cache.pkl")
        with open(cache_path, "wb") as f:
            pickle.dump(
                [{"content": d.page_content, "metadata": d.metadata} for d in docs],
                f
            )
        logger.info(f"BM25 cache saved to {cache_path}")
    except Exception as e:
        logger.warning(f"Could not save BM25 cache: {e}")


def _load_bm25(persist_dir: str) -> bool:
    """Load BM25 index from disk. Returns True if successful."""
    global _bm25_index, _bm25_docs
    cache_path = os.path.join(persist_dir, "bm25_cache.pkl")
    if not os.path.exists(cache_path):
        return False
    try:
        with open(cache_path, "rb") as f:
            raw = pickle.load(f)
        _bm25_docs = [
            Document(page_content=r["content"], metadata=r["metadata"])
            for r in raw
        ]
        _bm25_index = _build_bm25(_bm25_docs)
        return True
    except Exception as e:
        logger.warning(f"Could not load BM25 cache: {e}")
        return False


def _reciprocal_rank_fusion(
    list1: List[Document],
    list2: List[Document],
    k: int = 60,
) -> List[Document]:
    scores: dict[str, Tuple[float, Document]] = {}
    for rank, doc in enumerate(list1, start=1):
        key = doc.page_content[:200]
        prev = scores.get(key, (0.0, doc))[0]
        scores[key] = (prev + 1.0 / (rank + k), doc)
    for rank, doc in enumerate(list2, start=1):
        key = doc.page_content[:200]
        prev = scores.get(key, (0.0, doc))[0]
        scores[key] = (prev + 1.0 / (rank + k), doc)
    return [doc for _, doc in sorted(scores.values(), key=lambda x: x[0], reverse=True)]


def _type_boost(docs: List[Document]) -> List[Document]:
    rows   = [d for d in docs if d.metadata.get("type") == "table_row"]
    tables = [d for d in docs if d.metadata.get("type") == "table"]
    texts  = [d for d in docs if d.metadata.get("type") == "text"]
    return rows + tables + texts