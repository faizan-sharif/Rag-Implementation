"""
rag_chain.py
============
LCEL-based RAG chain — no deprecated RetrievalQA.
Python 3.11 + langchain 0.2.x compatible.

Pipeline:
  query -> hybrid_retriever -> format_docs -> prompt -> groq_llm -> str_parser
"""

from typing import List

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from src.llm.Gemini import get_llm
from src.llm.prompt_template import get_prompt
from src.retrieval.vector_store import get_hybrid_retriever
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RAGChain:
    """
    End-to-end RAG chain using:
      - Hybrid BM25 + Dense retrieval with RRF re-ranking
      - Table-row type boosting
      - 9-rule Financial Analyst prompt
      - Groq llama3-70b (deterministic, temp=0)
    """

    def __init__(self, vectorstore):
        self.llm       = get_llm()
        self.retriever = get_hybrid_retriever(vectorstore)
        self.prompt    = get_prompt()
        self._last_docs: List = []
        self.chain     = self._build_chain()
        logger.info("RAGChain ready (Hybrid BM25+Dense, LCEL).")

    # ─────────────────────────────────────────────────────────────────────
    def _build_chain(self):

        def retrieve_and_format(question: str) -> str:
            docs = self.retriever(question)
            self._last_docs = docs
            return self._format_context(docs)

        chain = (
            {
                "context":  RunnableLambda(retrieve_and_format),
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    # ─────────────────────────────────────────────────────────────────────
    def ask(self, question: str) -> str:
        logger.info(f"Query: {question}")
        self._last_docs = []

        answer = self.chain.invoke(question)

        sources = self._format_sources(self._last_docs)
        if sources:
            answer += f"\n\n*Source: {sources}*"

        return answer

    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _format_context(docs: List) -> str:
        """
        Present each chunk clearly labelled with its type and page.
        table_row chunks are prefixed with [ROW] so the LLM prioritises them.
        """
        parts = []
        for i, doc in enumerate(docs, 1):
            doc_type = doc.metadata.get("type", "text")
            page     = doc.metadata.get("page", "?")
            label    = "[ROW]" if doc_type == "table_row" else f"[{doc_type.upper()}]"
            parts.append(
                f"{label} Chunk {i} | Page {page}\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _format_sources(docs: List) -> str:
        if not docs:
            return ""
        pages   = sorted({str(d.metadata.get("page", "?")) for d in docs})
        sources = sorted({d.metadata.get("source", "unknown")  for d in docs})
        return f"{', '.join(sources)} | Page(s): {', '.join(pages)}"