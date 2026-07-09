import logging
import os
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("Gemini API Key not found.")

genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-2.5-flash"

FALLBACK_MESSAGE = "This information is not available in the uploaded dataset."

SYSTEM_PROMPT = """You are an AI assistant for the Climate Health Analysis project.

Answer ONLY using the supplied context.
Never use outside knowledge.
Never guess.
Never hallucinate.
Never fabricate countries, statistics, years, AQI, PM2.5, temperature, GDP, or health indices.

The retrieved context is the ONLY knowledge source.
If the retrieved context confidence is low, incomplete, or unrelated to the user's question,
never attempt to infer the answer.

If the answer is not available in the supplied context, reply EXACTLY:
This information is not available in the uploaded dataset.

Never change this sentence.
Keep answers concise.
Prefer bullet points.
Do not mention you are Gemini.
Do not mention AI."""

_MODEL: Optional[genai.GenerativeModel] = None


def load_model() -> genai.GenerativeModel:
    """Load and cache the Gemini generative model instance."""
    global _MODEL
    if _MODEL is None:
        _MODEL = genai.GenerativeModel(MODEL_NAME)
    return _MODEL


def build_prompt(question: str, context: str) -> str:
    """Build a single formatted prompt combining the system prompt, context, and question."""
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:"
    )


def generate_answer(question: str, context: str) -> str:
    """Generate a grounded natural language answer from Gemini using the retrieved context."""
    if not context or not context.strip():
        return FALLBACK_MESSAGE

    try:
        model = load_model()
        prompt = build_prompt(question, context)
        response = model.generate_content(prompt)

        if not response or not getattr(response, "text", None) or not response.text.strip():
            return FALLBACK_MESSAGE

        return response.text.strip()
    except Exception as exc:
        logger.error("generate_answer() failed: %s", exc)
        return FALLBACK_MESSAGE


if __name__ == "__main__":
    sample_context = (
        "Country: India, Year: 2023, Air Quality Index: 158, "
        "PM2.5: 62.4 ug/m3, Respiratory Disease Rate: 12.8 per 1000."
    )

    user_question = input("Ask a question: ").strip()
    answer = generate_answer(user_question, sample_context)
    print(answer)