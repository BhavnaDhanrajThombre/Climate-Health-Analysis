import logging
from typing import Any, Dict, Union

from data_query import execute_query
from llm import generate_answer
from query_router import route_query
from retriever import retrieve_context

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

FALLBACK_MESSAGE = "This information is not available in the uploaded dataset."


def _handle_analytical(question: str, router: Dict[str, Any]) -> Dict[str, Any]:
    """Route an analytical question to data_query and return its structured result."""
    try:
        logger.info("Analytical query detected for question: %s", question)
        result = execute_query(router)
        return result
    except Exception as exc:
        logger.error("Analytical query failed: %s", exc)
        return {"status": False, "operation": "", "column": "", "rows": 0, "data": [], "message": FALLBACK_MESSAGE}


def _handle_semantic(question: str) -> str:
    """Route a semantic question through the retriever and Gemini LLM."""
    try:
        logger.info("Semantic query detected for question: %s", question)
        context = retrieve_context(question)

        if not context or not context.strip():
            logger.warning("Retriever returned empty context for question: %s", question)
            return FALLBACK_MESSAGE

        answer = generate_answer(question, context)

        if not answer or not answer.strip():
            logger.warning("LLM returned empty answer for question: %s", question)
            return FALLBACK_MESSAGE

        return answer
    except Exception as exc:
        logger.error("Semantic query failed: %s", exc)
        return FALLBACK_MESSAGE


def chat(question: str) -> Union[str, Dict[str, Any]]:
    """Route a user question to the appropriate module and return the final response."""
    try:
        if not question or not question.strip():
            return FALLBACK_MESSAGE

        logger.info("Incoming question: %s", question)
        router = route_query(question)
        intent = router.get("intent", "unsupported")
        logger.info("Detected intent: %s", intent)

        if intent == "analytical":
            return _handle_analytical(question, router)

        if intent == "semantic":
            return _handle_semantic(question)

        logger.info("Unsupported intent for question: %s", question)
        return FALLBACK_MESSAGE
    except Exception as exc:
        logger.error("chat() failed: %s", exc)
        return FALLBACK_MESSAGE


if __name__ == "__main__":
    while True:
        user_question = input("Ask a question (or 'exit'): ").strip()
        if user_question.lower() == "exit":
            break
        response = chat(user_question)
        print(response)