# Rag System for tabular data

> AI-powered Q&A over the Punjab Government Annual Budget Statement 2025-26.
> Returns precise tabular answers in structured table format. All figures in **Rs. millions**.

---

## What This System Does

Ask natural language questions about the Tabular Data and get precise answers like:

```
Question: What is Total Revenue (A=B+C) for Budget Estimates 2025-26?

| Head of Account       | Column (Fiscal Year)     | Value (Rs. in million) |
|-----------------------|--------------------------|------------------------|
| Total Revenue (A=B+C) | Budget Estimates 2025-26 | 5,169,479.560          |

*Source: transactions.pdf | Page(s): 3, 4*
```

---

## Project Structure

```
D:\office work\Rag Implementation\
├── main.py                          # Entry point: ingest | query
├── requirements.txt                 # All dependencies pinned
├── .env                             # GOOGLE_API_KEY goes here
│
├── config\
│   ├── __init__.py
│   └── settings.py                  # All configuration constants
│
├── documents\
│   └── punjab_budget_2025_2026.pdf  # Your PDF goes here
│
├── chroma_db\                       # Auto-created after ingest
│   └── bm25_cache.pkl               # BM25 index (persisted)
│
└── src\
    ├── __init__.py
    ├── chain\
    │   ├── __init__.py
    │   └── rag_chain.py             # LCEL chain orchestrator
    ├── ingestion\
    │   ├── __init__.py
    │   ├── document_loader.py       # pdfplumber table extractor
    │   └── text_splitter.py        # Table-aware chunking
    ├── llm\
    │   ├── __init__.py
    │   ├── gemini_llm.py            # Google Gemini wrapper
    │   └── prompt_template.py      # 9-rule Financial Analyst prompt
    ├── retrieval\
    │   ├── __init__.py
    │   ├── embeddings.py            # all-mpnet-base-v2 embeddings
    │   └── vector_store.py         # Hybrid BM25 + Dense + RRF
    └── utils\
        ├── __init__.py
        └── logger.py               # Structured logging
```

---

## Quick Start

### 1. Get Gemini API Key (Free)
1. Go to: https://aistudio.google.com/app/apikey
2. Click **Create API Key**
3. Copy the key (starts with `AIza...`)

### 2. Add Key to .env
```
GOOGLE_API_KEY=AIza_your_real_key_here
```

### 3. Install Dependencies
```powershell
cd "D:\office work\Rag Implementation"
.\venv\Scripts\Activate
pip install -r requirements.txt
```

### 4. Place PDF
```
documents\punjab_budget_2025_2026.pdf
```

### 5. Ingest (First Time Only — takes 5–15 min)
```powershell
python main.py ingest
```

Expected output:
```
[INFO] Loading: punjab_budget_2025_2026.pdf
[INFO]   -> 4153 chunks from punjab_budget_2025_2026.pdf
[INFO] Chunks after split: 4314 total (3906 row + 122 table + 237 text)
[INFO] ChromaDB + BM25 index built and saved (4314 chunks).
[INFO] INGESTION COMPLETE
```

### 6. Query
```powershell
python main.py query
```

### Daily Use (ChromaDB already built)
```powershell
.\venv\Scripts\Activate
python main.py query
```

---

## Example Questions

| Question | Type |
|----------|------|
| What is Total Revenue (A=B+C) for Budget Estimates 2025-26? | Formula row |
| What is the Annual Development Program (ADP) for 2025-26? | ADP lookup |
| What is Current Expenditure for Revised Estimates 2024-25? | Column match |
| What were actual pension expenditures in Accounts 2023-24? | Historical |
| What is the Police department budget for Budget Estimates 2025-26? | Dept lookup |
| What is Total Capital Expenditure for Budget Estimates 2025-26? | Capital |
| What is the Health department allocation for 2025-26? | Dept lookup |
| What is Revenue Receipts (B) for Budget Estimates 2025-26? | Sub-head |

---

## Output Format

Every answer returns a strict markdown table:

```
| Head of Account | Column (Fiscal Year) | Value (Rs. in million) |
|----------------|---------------------|------------------------|
| [exact row]    | [exact column]      | [exact value]          |

*Source: filename.pdf | Page(s): X, Y*
```

If data is not found:
```
Data not found in retrieved context. The requested head of account
or fiscal year column was not present. Please re-ingest or rephrase.
```

If a formula check fails:
```
| Head of Account | Column (Fiscal Year) | Value (Rs. in million)      |
|----------------|---------------------|-----------------------------|
| Revenue (A=B+C) | Budget 2025-26     | Data inconsistency detected |
```

---

## Fiscal Year Columns

| Column Name | Meaning |
|-------------|---------|
| Accounts 2023-24 | Actual figures for FY 2023-24 |
| Budget Estimates 2024-25 | Approved budget for FY 2024-25 |
| Revised Estimates 2024-25 | Revised budget for FY 2024-25 |
| Budget Estimates 2025-26 | Proposed budget for FY 2025-26 |

---

## Advanced Techniques Used

| Layer | Technique | Benefit |
|-------|-----------|---------|
| **Ingestion** | pdfplumber (3 strategies) | Reads actual table grids, not raw text |
| **Ingestion** | Per-row Documents | Each budget row = individual searchable chunk |
| **Ingestion** | Natural language sentences | Better semantic embedding per row |
| **Ingestion** | Section header injection | ADP rows know they belong to ADP section |
| **Ingestion** | Header propagation | Multi-page tables keep column names |
| **Retrieval** | BM25 sparse (persisted) | Exact keyword match — survives session restart |
| **Retrieval** | Query expansion | ADP → "annual development program" automatically |
| **Retrieval** | Bigram tokenization | Better phrase matching in BM25 |
| **Retrieval** | Dense MMR (ChromaDB) | Semantic similarity with diversity |
| **Retrieval** | RRF fusion | Best of BM25 + dense combined |
| **Retrieval** | Type boosting | table_row chunks prioritized in results |
| **LLM** | Gemini 2.0 Flash | Fast, accurate, 1M token context |
| **LLM** | Temperature = 0.0 | Fully deterministic — no hallucination |
| **Prompt** | 9-rule strict analyst | Exact row/column match, formula verification |

---

## Configuration (config/settings.py)

```python
GEMINI_MODEL      = "gemini-2.0-flash"   # or "gemini-2.0-flash-lite"
LLM_TEMPERATURE   = 0.0                  # deterministic
CHUNK_SIZE        = 1500                 # chars per chunk
RETRIEVER_K       = 15                   # final chunks sent to LLM
RETRIEVER_FETCH_K = 60                   # candidates before MMR filter
BM25_K            = 15                   # BM25 candidates
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `GOOGLE_API_KEY not set` | Add real key to `.env` |
| `404 models/gemini-X not found` | Use `gemini-2.0-flash` in settings.py |
| `429 Quota exceeded` | Free tier exhausted — create new API key at aistudio.google.com |
| `Data not found` on valid question | Delete `chroma_db\` and re-ingest |
| `persist()` AttributeError | Already fixed — chromadb 0.5.3 auto-persists |
| `langchain_core.memory` error | Run: `pip install -r requirements.txt --force-reinstall` |
| Ingestion seems stuck | Normal — embedding 4000+ chunks takes 5–15 min on CPU |
| Telemetry warnings | Harmless — ChromaDB trying to send usage stats, ignore |

---

## Re-ingestion

Re-ingest only when:
- You replace the PDF with a new version
- You change `CHUNK_SIZE` or `CHUNK_OVERLAP` in settings
- You get persistent "Data not found" on queries that should work

```powershell
rmdir /s /q chroma_db
python main.py ingest
```

---

## Dependencies

```
langchain==0.3.13
langchain-core==0.3.28
langchain-google-genai==2.0.7
chromadb==0.5.3
sentence-transformers==3.0.1
pdfplumber==0.11.4
rank-bm25==0.2.2
google-generativeai==0.8.3
```

---

## Architecture

```
PDF
 │
 ▼
pdfplumber (3 extraction strategies)
 │
 ├── Table chunks    (markdown pipe format)
 ├── Table row chunks (pipe + natural language sentence)
 └── Text chunks     (raw page text)
 │
 ▼
ChromaDB (dense vectors) + BM25 cache (sparse index)
 │
 ▼
Query → expand_query() → BM25 + Dense MMR → RRF fusion → Type boost
 │
 ▼
Top 15 chunks → 9-rule Financial Analyst Prompt → Gemini 2.0 Flash
 │
 ▼
Strict markdown table answer
```
