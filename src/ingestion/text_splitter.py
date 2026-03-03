"""
text_splitter.py
================
Table-aware chunking strategy:
  - table       : keep whole (up to 3000 chars) - never cut mid-row
  - table_row   : never split (already atomic)
  - text        : standard recursive split
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from src.utils.logger import get_logger

logger = get_logger(__name__)

TABLE_MAX = CHUNK_SIZE * 2   # 3000 chars - keeps most budget tables intact


def split_documents(docs: List[Document]) -> List[Document]:
    table_row_docs = [d for d in docs if d.metadata.get("type") == "table_row"]
    table_docs     = [d for d in docs if d.metadata.get("type") == "table"]
    text_docs      = [d for d in docs if d.metadata.get("type") == "text"]

    result: List[Document] = []

    # table_row: atomic - never split
    result.extend(table_row_docs)

    # table: keep intact if fits, else split carefully at newlines
    table_splitter = RecursiveCharacterTextSplitter(
        chunk_size=TABLE_MAX,
        chunk_overlap=100,
        separators=["\n\n", "\n", " | "],
    )
    for doc in table_docs:
        if len(doc.page_content) <= TABLE_MAX:
            result.append(doc)
        else:
            splits = table_splitter.split_documents([doc])
            result.extend(splits)

    # text: standard split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    text_splits = text_splitter.split_documents(text_docs)
    result.extend(text_splits)

    logger.info(
        f"Chunks after split: {len(result)} total "
        f"({len(table_row_docs)} row + {len(table_docs)} table + {len(text_splits)} text)"
    )
    return result