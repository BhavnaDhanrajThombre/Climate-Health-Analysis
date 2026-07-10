"""
app.py

Climate Health AI Assistant — Streamlit Frontend.

This file is a PURE FRONTEND. It contains no routing, retrieval, LLM, or
data-query logic of its own. Every user question is sent through a single
call to chat_engine.chat(question); that module (and the modules it calls
internally — query_router.py, data_query.py, retriever.py, llm.py) is the
only "brain" in this system.

Run with:
    streamlit run app.py
"""

import logging
import sys
import traceback
from pathlib import Path

import streamlit as st

# =============================================================================
# LOGGING
# =============================================================================
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# =============================================================================
# CONSTANTS
# =============================================================================
FALLBACK_MESSAGE = "This information is not available in the uploaded dataset."

EXAMPLE_QUESTIONS = [
    "Average AQI in India",
    "Highest PM2.5 country",
    "Top 10 countries by respiratory disease",
    "Explain climate impacts on respiratory disease",
    "Relationship between heat waves and hospital admissions",
]

PROJECT_NAME = "Climate Health Analysis"
DEVELOPER_NAME = "Ayush Shukla"
TECH_STACK = ["Python", "Streamlit", "FAISS", "Gemini", "Sentence Transformers"]

# =============================================================================
# PAGE CONFIG (must be the first Streamlit call)
# =============================================================================
st.set_page_config(
    page_title="Climate Health AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# IMPORT chat_engine.chat() — the ONLY entry point into the chatbot brain
# =============================================================================
# chat_engine.py (and the modules it imports) live in source/chatbot/ and are
# used exactly as-is. We only extend sys.path so the import can resolve; no
# business logic from those modules is reimplemented here.
_CHATBOT_DIR = Path(__file__).resolve().parent / "source" / "chatbot"
if str(_CHATBOT_DIR) not in sys.path:
    sys.path.insert(0, str(_CHATBOT_DIR))

CHATBOT_AVAILABLE = False
CHATBOT_IMPORT_TRACEBACK = ""

try:
    from chat_engine import chat as chat_engine_chat  # noqa: E402
    CHATBOT_AVAILABLE = True
except Exception:
    CHATBOT_IMPORT_TRACEBACK = traceback.format_exc()
    logger.error("Failed to import chat_engine: %s", CHATBOT_IMPORT_TRACEBACK)

# =============================================================================
# GLOBAL CSS — professional, clean, centered look
# =============================================================================
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .app-title {
        text-align: center;
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .app-subtitle {
        text-align: center;
        font-size: 15px;
        color: #8A93A3;
        margin-top: 2px;
        margin-bottom: 18px;
    }
    .example-chip {
        display: inline-block;
        background-color: rgba(31, 143, 255, 0.10);
        border: 1px solid rgba(31, 143, 255, 0.35);
        color: #1F8FFF;
        border-radius: 16px;
        padding: 5px 12px;
        margin: 3px 6px 3px 0px;
        font-size: 12.5px;
        font-weight: 600;
    }
    .app-footer {
        text-align: center;
        color: #8A93A3;
        font-size: 12px;
        margin-top: 24px;
        padding-top: 10px;
        border-top: 1px solid rgba(255,255,255,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# SIDEBAR — project information
# =============================================================================
with st.sidebar:
    st.markdown("### ℹ️ Project Information")
    st.markdown(f"**Project Name:** {PROJECT_NAME}")
    st.markdown(f"**Developer:** {DEVELOPER_NAME}")

    st.markdown("---")
    st.markdown("**Technology Stack**")
    for tech in TECH_STACK:
        st.markdown(f"- {tech}")

    st.markdown("---")
    st.markdown("**Status**")
    if CHATBOT_AVAILABLE:
        st.success("AI Assistant: Online")
    else:
        st.error("AI Assistant: Unavailable")

    st.markdown("---")
    if st.session_state.get("chat_history"):
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# =============================================================================
# HEADER
# =============================================================================
st.markdown('<div class="app-title">🤖 Climate Health AI Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Ask questions related to Climate Change and Health '
    "using the uploaded dataset.</div>",
    unsafe_allow_html=True,
)

# Example questions
chips_html = "".join(f'<span class="example-chip">{q}</span>' for q in EXAMPLE_QUESTIONS)
st.markdown(f"<div style='text-align:center; margin-bottom:16px;'>{chips_html}</div>", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": "user"/"assistant", "content": str}

# =============================================================================
# RESPONSE FORMATTING (presentation only — no routing/retrieval/LLM logic)
# =============================================================================
def _rows_to_markdown_table(rows) -> str:
    """Render a list of dict rows as a plain markdown table (no extra deps)."""
    if not rows:
        return ""
    columns = list(rows[0].keys())
    header = "| " + " | ".join(str(c).replace("_", " ").title() for c in columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row.get(c, "")) for c in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, sep] + body)


def format_response(response) -> str:
    """
    Convert whatever chat_engine.chat() returns (a plain string for semantic
    answers, or a structured dict for analytical results) into a single
    markdown string for display. This is pure presentation — it does not
    alter, re-derive, or fabricate any value returned by chat_engine.
    """
    try:
        if isinstance(response, str):
            return response.strip() or FALLBACK_MESSAGE

        if isinstance(response, dict):
            if not response.get("status"):
                return response.get("message") or FALLBACK_MESSAGE

            data = response.get("data", [])
            if not data:
                return FALLBACK_MESSAGE

            operation = str(response.get("operation", "")).title()
            column = str(response.get("column", "")).replace("_", " ").title()
            heading = f"**{operation} — {column}**" if column else f"**{operation}**"

            if len(data) == 1:
                row = data[0]
                lines = [f"- **{str(k).replace('_', ' ').title()}:** {v}" for k, v in row.items()]
                return heading + "\n\n" + "\n".join(lines)

            return heading + "\n\n" + _rows_to_markdown_table(data)

        return FALLBACK_MESSAGE
    except Exception:
        logger.error("format_response() failed: %s", traceback.format_exc())
        return FALLBACK_MESSAGE


# =============================================================================
# ASK chat_engine.chat() — the single entry point into the chatbot brain
# =============================================================================
def ask_chatbot(question: str):
    """
    Send `question` to chat_engine.chat() and return (display_text, debug_traceback).
    debug_traceback is an empty string on success, or the full traceback text
    if chat_engine raised an exception.
    """
    if not CHATBOT_AVAILABLE:
        return FALLBACK_MESSAGE, CHATBOT_IMPORT_TRACEBACK

    try:
        raw_response = chat_engine_chat(question)
        return format_response(raw_response), ""
    except Exception:
        tb = traceback.format_exc()
        logger.error("chat_engine.chat() raised an exception: %s", tb)
        return FALLBACK_MESSAGE, tb


# =============================================================================
# CHATBOT UNAVAILABLE — show error + traceback, but never crash the app
# =============================================================================
if not CHATBOT_AVAILABLE:
    st.error("AI Assistant could not be loaded.")
    with st.expander("🐞 Debug — Import Traceback"):
        st.code(CHATBOT_IMPORT_TRACEBACK or "No traceback available.", language="text")

# =============================================================================
# CHAT INTERFACE
# =============================================================================
# Render existing conversation history
for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("debug"):
            with st.expander("🐞 Debug — Traceback"):
                st.code(turn["debug"], language="text")

# Chat input
user_question = st.chat_input(
    "Ask about temperature, AQI, PM2.5, GDP, respiratory disease, and more..."
    if CHATBOT_AVAILABLE
    else "AI Assistant is unavailable"
)

if user_question:
    # Display and store the user's message
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Get and display the assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer_text, debug_trace = ask_chatbot(user_question)
        st.markdown(answer_text)
        if debug_trace:
            with st.expander("🐞 Debug — Traceback"):
                st.code(debug_trace, language="text")

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer_text, "debug": debug_trace}
    )

# =============================================================================
# FOOTER
# =============================================================================
st.markdown(
    f'<div class="app-footer">{PROJECT_NAME} &nbsp;•&nbsp; Developed by {DEVELOPER_NAME} '
    f"&nbsp;•&nbsp; Powered by Gemini 2.5 Flash, FAISS &amp; Sentence Transformers</div>",
    unsafe_allow_html=True,
)