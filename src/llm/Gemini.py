"""
gemini_llm.py
=============
Google Gemini LLM wrapper via langchain-google-genai.
Replaces groq_llm.py — drop-in compatible with the LCEL chain.

Model choice:
  gemini-1.5-pro   -> best reasoning, handles long table context (1M tokens)
  gemini-1.5-flash -> 3x faster, same accuracy for structured Q&A
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import GEMINI_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_llm() -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key or api_key.startswith("AIza_your"):
        raise ValueError(
            "GOOGLE_API_KEY not set.\n"
            "  1. Go to https://aistudio.google.com/app/apikey\n"
            "  2. Click  Create API Key\n"
            "  3. Paste it into .env as: GOOGLE_API_KEY=AIza...\n"
        )
    logger.info(f"Initializing Google Gemini: {GEMINI_MODEL} (temp={LLM_TEMPERATURE})")
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=LLM_TEMPERATURE,
        max_output_tokens=LLM_MAX_TOKENS,
        google_api_key=api_key,
        convert_system_message_to_human=True,   # Gemini requires this
    )