import os

# Paths
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCUMENTS_DIR  = os.path.join(BASE_DIR, "documents")
CHROMA_DIR     = os.path.join(BASE_DIR, "chroma_db")

# Google Gemini LLM
# gemini-1.5-pro: best reasoning, 1M token context window
# gemini-1.5-flash: faster + cheaper, still excellent for structured data
GEMINI_MODEL      = "gemini-2.5-flash"
LLM_TEMPERATURE   = 0.0     # deterministic - critical for financial accuracy
LLM_MAX_TOKENS    = 2048

# Embeddings (local - no API cost)
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Chunking
CHUNK_SIZE    = 1500
CHUNK_OVERLAP = 150

# Retrieval
RETRIEVER_K       = 15
RETRIEVER_FETCH_K = 60
RETRIEVER_LAMBDA  = 0.7
BM25_K            = 15

SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md", ".docx"]