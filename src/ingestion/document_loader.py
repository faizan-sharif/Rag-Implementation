"""
document_loader.py
==================
Advanced table-aware PDF loader using pdfplumber.

Key techniques:
  1. Per-page table extraction  -> markdown pipe format (LLM-readable)
  2. Header propagation         -> multi-page tables get repeated column headers
  3. Row context injection      -> each table row also stored as individual Document
     so needle-in-haystack lookups ("Police budget 2025-26") hit directly
  4. Raw text fallback          -> text between tables captured separately
  5. Rich metadata              -> page, type, table_index, row_index, source
"""

import os
import re
from typing import List

import pdfplumber
from langchain_core.documents import Document

from config.settings import SUPPORTED_EXTENSIONS
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Punjab Budget column aliases for normalisation
FISCAL_COL_ALIASES = {
    "accounts 2023-24":           "Accounts 2023-24",
    "account 2023-24":            "Accounts 2023-24",
    "budget estimates 2024-25":   "Budget Estimates 2024-25",
    "budget estimate 2024-25":    "Budget Estimates 2024-25",
    "revised estimates 2024-25":  "Revised Estimates 2024-25",
    "revised estimate 2024-25":   "Revised Estimates 2024-25",
    "budget estimates 2025-26":   "Budget Estimates 2025-26",
    "budget estimate 2025-26":    "Budget Estimates 2025-26",
}


def load_documents(documents_dir: str) -> List[Document]:
    if not os.path.exists(documents_dir):
        logger.error(f"Documents directory not found: {documents_dir}")
        return []

    all_docs: List[Document] = []
    for filename in sorted(os.listdir(documents_dir)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        filepath = os.path.join(documents_dir, filename)
        logger.info(f"Loading: {filename}")

        if ext == ".pdf":
            docs = load_pdf(filepath)
        else:
            docs = load_text_file(filepath)

        logger.info(f"  -> {len(docs)} chunks from {filename}")
        all_docs.extend(docs)

    logger.info(f"Total documents loaded: {len(all_docs)}")
    return all_docs


# ─────────────────────────────────────────────────────────────────────────────
# PDF loader
# ─────────────────────────────────────────────────────────────────────────────

def load_pdf(filepath: str) -> List[Document]:
    docs: List[Document] = []
    filename = os.path.basename(filepath)
    last_header: List[str] = []          # propagate headers across pages

    with pdfplumber.open(filepath) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):

            # ── 1. Extract tables ──────────────────────────────────────────
            tables = page.extract_tables(
                table_settings={
                    "vertical_strategy":   "lines",
                    "horizontal_strategy": "lines",
                    "snap_tolerance":       5,
                    "join_tolerance":       3,
                    "min_words_vertical":   1,
                    "min_words_horizontal": 1,
                }
            )

            if tables:
                for t_idx, raw_table in enumerate(tables):
                    cleaned = clean_table(raw_table)
                    if not cleaned:
                        continue

                    # Detect if first row is a header row
                    header, data_rows = detect_header(cleaned, last_header)
                    if header:
                        last_header = header  # propagate for next page

                    # ── 2. Whole-table document ────────────────────────────
                    md = build_markdown_table(header, data_rows)
                    if md:
                        docs.append(Document(
                            page_content=md,
                            metadata={
                                "source":      filename,
                                "page":        page_num,
                                "type":        "table",
                                "table_index": t_idx,
                                "row_count":   len(data_rows),
                            }
                        ))

                    # ── 3. Per-row documents (for precise cell lookup) ─────
                    if header:
                        for r_idx, row in enumerate(data_rows):
                            row_doc = build_row_document(
                                header, row, filename, page_num, t_idx, r_idx
                            )
                            if row_doc:
                                docs.append(row_doc)

            # ── 4. Raw text (non-table content) ───────────────────────────
            raw_text = page.extract_text()
            if raw_text:
                cleaned_text = clean_raw_text(raw_text)
                if len(cleaned_text) > 60:
                    docs.append(Document(
                        page_content=cleaned_text,
                        metadata={
                            "source": filename,
                            "page":   page_num,
                            "type":   "text",
                        }
                    ))

    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Table helpers
# ─────────────────────────────────────────────────────────────────────────────

def clean_table(raw_table: list) -> list:
    """Remove None cells, strip whitespace, drop fully-empty rows."""
    cleaned = []
    for row in raw_table:
        clean_row = [
            str(cell).replace("\n", " ").strip() if cell is not None else ""
            for cell in row
        ]
        if any(c for c in clean_row):
            cleaned.append(clean_row)
    return cleaned


def detect_header(rows: list, last_header: list) -> tuple:
    """
    Heuristic: first row is header if it contains known fiscal year strings
    or has no purely-numeric cells.
    Falls back to last_header for continuation pages.
    """
    if not rows:
        return last_header, rows

    first = rows[0]
    first_lower = [c.lower() for c in first]

    fiscal_keywords = ["accounts", "budget", "revised", "estimates", "head"]
    is_header = any(
        any(kw in cell for kw in fiscal_keywords)
        for cell in first_lower
    )
    # Also treat as header if no cell parses as a large number
    numeric_cells = sum(
        1 for c in first
        if re.sub(r"[,\s]", "", c).replace(".", "", 1).replace("-", "", 1).isdigit()
        and len(re.sub(r"[,\s]", "", c)) > 4
    )
    if numeric_cells == 0:
        is_header = True

    if is_header:
        # Normalise column names
        header = [normalise_col(c) for c in first]
        return header, rows[1:]
    else:
        return last_header, rows


def normalise_col(col: str) -> str:
    lower = col.lower().strip()
    return FISCAL_COL_ALIASES.get(lower, col.strip())


def build_markdown_table(header: list, rows: list) -> str:
    """Build a pipe-delimited markdown table string."""
    if not rows:
        return ""
    lines = []
    if header:
        lines.append(" | ".join(header))
        lines.append(" | ".join(["---"] * len(header)))
    for row in rows:
        padded = row + [""] * max(0, len(header) - len(row))
        lines.append(" | ".join(padded[:len(header)] if header else row))
    return "\n".join(lines)


def build_row_document(
    header: list, row: list, source: str,
    page: int, t_idx: int, r_idx: int
) -> Document | None:
    """
    Create a fine-grained Document for a single table row.
    Format: "Head of Account: X | Accounts 2023-24: Y | Budget Estimates 2024-25: Z | ..."
    This lets the retriever match "Police 2025-26" -> this exact row.
    """
    if not header or not row:
        return None

    padded = row + [""] * max(0, len(header) - len(row))
    pairs = []
    for col, val in zip(header, padded):
        if val.strip():
            pairs.append(f"{col}: {val.strip()}")

    if not pairs:
        return None

    content = " | ".join(pairs)

    # Skip rows that are pure separators or sub-headers
    if len(content) < 10:
        return None

    return Document(
        page_content=content,
        metadata={
            "source":      source,
            "page":        page,
            "type":        "table_row",
            "table_index": t_idx,
            "row_index":   r_idx,
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Text helpers
# ─────────────────────────────────────────────────────────────────────────────

def clean_raw_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_text_file(filepath: str) -> List[Document]:
    ext = os.path.splitext(filepath)[1].lower()
    filename = os.path.basename(filepath)
    try:
        if ext == ".docx":
            import docx2txt
            content = docx2txt.process(filepath)
        else:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        if content.strip():
            return [Document(
                page_content=content.strip(),
                metadata={"source": filename, "type": "text"},
            )]
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
    return []