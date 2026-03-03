"""
main.py
=======
Single entry point.
  python main.py ingest   ->  parse PDF, build ChromaDB + BM25
  python main.py query    ->  interactive Q&A session
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from config.settings import DOCUMENTS_DIR, CHROMA_DIR
from src.ingestion.document_loader import load_documents
from src.ingestion.text_splitter   import split_documents
from src.retrieval.embeddings      import get_embeddings
from src.retrieval.vector_store    import build_vector_store, load_vector_store
from src.chain.rag_chain           import RAGChain
from src.utils.logger              import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
def run_ingest() -> None:
    logger.info("=" * 55)
    logger.info("INGESTION PIPELINE STARTED")
    logger.info("=" * 55)

    docs = load_documents(DOCUMENTS_DIR)
    if not docs:
        logger.error("No documents found. Put PDF in documents/ folder.")
        return

    chunks = split_documents(docs)
    logger.info(f"Total chunks ready for embedding: {len(chunks)}")

    embeddings = get_embeddings()
    build_vector_store(chunks, embeddings, CHROMA_DIR)

    logger.info("=" * 55)
    logger.info("INGESTION COMPLETE — ChromaDB + BM25 index built.")
    logger.info("Run: python main.py query")
    logger.info("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
def run_query() -> None:
    embeddings  = get_embeddings()
    vectorstore = load_vector_store(embeddings, CHROMA_DIR)
    if vectorstore is None:
        logger.error("ChromaDB not found. Run: python main.py ingest")
        return

    chain = RAGChain(vectorstore)

    print()
    print("=" * 55)
    print("  Punjab Budget RAG — Financial Analyst Mode")
    print("  All figures in Rs. millions")
    print("  Type  exit  to quit")
    print("=" * 55)
    print()

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            break
        answer = chain.ask(question)
        print()
        print(answer)
        print()
        print("-" * 55)
        print()


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py ingest   # Parse PDF and build index")
        print("  python main.py query    # Start Q&A session")
        sys.exit(1)

    mode = sys.argv[1].lower()
    if mode == "ingest":
        run_ingest()
    elif mode == "query":
        run_query()
    else:
        print(f"Unknown mode: {mode!r}. Use ingest or query.")
        sys.exit(1)


if __name__ == "__main__":
    main()