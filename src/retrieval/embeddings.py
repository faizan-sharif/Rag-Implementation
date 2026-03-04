from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import EMBEDDING_MODEL
from src.utils.logger import get_logger

logger = get_logger(__name__)

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    all-mpnet-base-v2:
      - Best general-purpose dense embedding for semantic search
      - 768-dim, strong on structured/financial text
      - Normalised -> cosine similarity works correctly
    """
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )