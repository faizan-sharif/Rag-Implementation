"""
document_loader.py
==================
Advanced table-aware PDF loader using pdfplumber.

Fixes in this version:
  - Section header injection: each chunk knows what section it belongs to
  - Keyword tagging: ADP/development rows tagged in metadata for boosted retrieval
  - Natural language row description added alongside pipe format
  - More aggressive table extraction for merged/complex cells
  - Alias expansion stored in metadata
"""

import os
import re
from typing import List, Optional

import pdfplumber
from langchain_core.documents import Document

from config.settings import SUPPORTED_EXTENSIONS
from src.utils.logger import get_logger

logger = get_logger(__name__)

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

# Known section keywords to detect and propagate as context
SECTION_KEYWORDS = [
    "annual development program", "adp",
    "current expenditure", "capital expenditure",
    "revenue receipts", "revenue expenditure",
    "public sector development",
    "consolidated fund", "provincial consolidated fund",
    "general administration", "finance department",
    "health", "education", "police", "agriculture",
    "pension", "debt servicing",
]


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


def load_pdf(filepath: str) -> List[Document]:
    docs: List[Document] = []
    filename = os.path.basename(filepath)
    last_header: List[str] = []
    current_section: str = ""

    with pdfplumber.open(filepath) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):

            # Detect section from page text first
            raw_text = page.extract_text() or ""
            current_section = detect_section(raw_text, current_section)

            # Try multiple table extraction strategies
            tables = _extract_tables(page)

            if tables:
                for t_idx, raw_table in enumerate(tables):
                    cleaned = clean_table(raw_table)
                    if not cleaned:
                        continue

                    header, data_rows = detect_header(cleaned, last_header)
                    if header:
                        last_header = header

                    # Whole-table document
                    md = build_markdown_table(header, data_rows)
                    if md:
                        docs.append(Document(
                            page_content=md,
                            metadata={
                                "source":      filename,
                                "page":        page_num,
                                "type":        "table",
                                "table_index": t_idx,
                                "section":     current_section,
                                "row_count":   len(data_rows),
                            }
                        ))

                    # Per-row documents with section context injected
                    if header:
                        for r_idx, row in enumerate(data_rows):
                            row_doc = build_row_document(
                                header, row, filename, page_num,
                                t_idx, r_idx, current_section
                            )
                            if row_doc:
                                docs.append(row_doc)

            # Raw text
            if raw_text:
                cleaned_text = clean_raw_text(raw_text)
                if len(cleaned_text) > 60:
                    docs.append(Document(
                        page_content=cleaned_text,
                        metadata={
                            "source":  filename,
                            "page":    page_num,
                            "type":    "text",
                            "section": current_section,
                        }
                    ))

    return docs


def _extract_tables(page) -> list:
    """Try multiple extraction strategies, return best result."""
    # Strategy 1: line-based (best for ruled tables)
    tables = page.extract_tables({
        "vertical_strategy":   "lines",
        "horizontal_strategy": "lines",
        "snap_tolerance":       5,
        "join_tolerance":       3,
        "min_words_vertical":   1,
        "min_words_horizontal": 1,
    })
    if tables:
        return tables

    # Strategy 2: text-based (for tables without visible lines)
    tables = page.extract_tables({
        "vertical_strategy":   "text",
        "horizontal_strategy": "text",
        "snap_tolerance":       3,
    })
    if tables:
        return tables

    # Strategy 3: explicit lines only
    tables = page.extract_tables({
        "vertical_strategy":   "explicit",
        "horizontal_strategy": "lines",
        "explicit_vertical_lines": page.curves + page.edges,
    })
    return tables or []


def detect_section(text: str, current_section: str) -> str:
    """Detect the current budget section from page text."""
    text_lower = text.lower()
    for kw in SECTION_KEYWORDS:
        if kw in text_lower:
            return kw.title()
    return current_section


def clean_table(raw_table: list) -> list:
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
    if not rows:
        return last_header, rows

    first = rows[0]
    first_lower = [c.lower() for c in first]

    fiscal_keywords = ["accounts", "budget", "revised", "estimates", "head", "description", "particulars"]
    is_header = any(
        any(kw in cell for kw in fiscal_keywords)
        for cell in first_lower
    )

    numeric_cells = sum(
        1 for c in first
        if re.sub(r"[,\s]", "", c).replace(".", "", 1).replace("-", "", 1).lstrip("-").isdigit()
        and len(re.sub(r"[,\s]", "", c)) > 4
    )
    if numeric_cells == 0 and any(c.strip() for c in first):
        is_header = True

    if is_header:
        header = [normalise_col(c) for c in first]
        return header, rows[1:]
    else:
        return last_header, rows


def normalise_col(col: str) -> str:
    lower = col.lower().strip()
    return FISCAL_COL_ALIASES.get(lower, col.strip())


def build_markdown_table(header: list, rows: list) -> str:
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
    page: int, t_idx: int, r_idx: int,
    section: str = ""
) -> Optional[Document]:
    if not header or not row:
        return None

    padded = row + [""] * max(0, len(header) - len(row))
    pairs = []
    for col, val in zip(header, padded):
        if val.strip():
            pairs.append(f"{col}: {val.strip()}")

    if not pairs or len(" | ".join(pairs)) < 10:
        return None

    # Format 1: pipe-delimited (for exact lookup)
    pipe_content = " | ".join(pairs)

    # Format 2: natural language sentence (better semantic embedding)
    head_val = padded[0].strip() if padded else ""
    nl_parts = [f"The budget entry for '{head_val}'"]
    if section:
        nl_parts.append(f"under {section}")
    for col, val in zip(header[1:], padded[1:]):
        if val.strip() and col.strip():
            nl_parts.append(f"has {col} of Rs. {val.strip()} million")
    nl_sentence = " ".join(nl_parts) + "."

    # Combine both representations
    full_content = f"{pipe_content}\n{nl_sentence}"

    # Detect keywords for metadata tagging
    tags = []
    content_lower = full_content.lower()
    for kw in ["adp", "annual development", "development program",
               "current expenditure", "capital", "pension",
               "revenue", "salary", "grant"]:
        if kw in content_lower:
            tags.append(kw)

    return Document(
        page_content=full_content,
        metadata={
            "source":      source,
            "page":        page,
            "type":        "table_row",
            "table_index": t_idx,
            "row_index":   r_idx,
            "section":     section,
            "tags":        ",".join(tags),
        }
    )


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