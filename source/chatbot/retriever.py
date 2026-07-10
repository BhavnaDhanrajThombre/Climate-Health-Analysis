"""
retriever.py

Retrieves relevant documents from the FAISS vector database.

Features
--------
✓ Cosine Similarity Search
✓ Case Insensitive Search
✓ Query Preprocessing
✓ Synonym Mapping
✓ Similarity Threshold
✓ Hallucination Prevention
✓ Context Builder
✓ Ready for Gemini Integration
"""

from pathlib import Path
import pickle
import re
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ==========================================================
# Configuration
# ==========================================================

TOP_K = 5

# Increase to become more strict
SIMILARITY_THRESHOLD = 0.60

# ==========================================================
# Synonyms
# ==========================================================

SYNONYMS = {

    # AQI
    "aqi": "air quality index",
    "air quality": "air quality index",

    # PM2.5
    "pm25": "pm2.5",
    "pm 2.5": "pm2.5",

    # Temperature
    "temp": "temperature",

    # GDP
    "gdp": "gdp per capita",

    # Heat Wave
    "heatwave": "heat wave",

    # Respiratory
    "resp": "respiratory",

    # Healthcare
    "hospital": "healthcare",

}

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

VECTOR_DB_DIR = BASE_DIR / "vector_db"

FAISS_PATH = VECTOR_DB_DIR / "climate_index.faiss"

METADATA_PATH = VECTOR_DB_DIR / "metadata.pkl"

# ==========================================================
# Load Metadata and Vector Resources
# ==========================================================

documents: List[str] = []
metadata: List[Dict[str, Any]] = []
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = None
index = None


def _load_vector_payload(metadata_path: Path) -> Tuple[List[str], List[Dict[str, Any]], str]:
    """Load and normalize vector metadata from disk, including older or partial files."""
    if not metadata_path.exists():
        raise FileNotFoundError("Vector database not found.\nRun vectorize.py first.")

    with open(metadata_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise ValueError("Vector metadata file has an unexpected format.")

    raw_documents = data.get("documents") or data.get("texts") or []
    if not isinstance(raw_documents, list):
        raw_documents = list(raw_documents)

    raw_metadata = data.get("metadata") or []
    if not isinstance(raw_metadata, list):
        raw_metadata = list(raw_metadata)

    if len(raw_metadata) < len(raw_documents):
        raw_metadata.extend([{}] * (len(raw_documents) - len(raw_metadata)))

    normalized_metadata: List[Dict[str, Any]] = []
    for item in raw_metadata:
        if isinstance(item, dict):
            normalized_metadata.append(item)
        else:
            normalized_metadata.append({})

    model_name = data.get("model_name") or MODEL_NAME
    return raw_documents, normalized_metadata, model_name


def _load_vector_resources() -> None:
    """Load the sentence-transformer model and FAISS index lazily when needed."""
    global documents, metadata, MODEL_NAME, model, index

    if model is not None and index is not None:
        return

    documents, metadata, MODEL_NAME = _load_vector_payload(METADATA_PATH)

    print("=" * 60)
    print("Loading Embedding Model...")
    print("=" * 60)

    model = SentenceTransformer(MODEL_NAME)

    print("\nLoading FAISS Index...")
    index = faiss.read_index(str(FAISS_PATH))

    print(f"Vectors Loaded : {index.ntotal}")
    print(f"Documents Loaded : {len(documents)}")

# ==========================================================
# Query Preprocessing
# ==========================================================

def preprocess_query(query: str) -> str:
    """
    Cleans and normalizes the user query.
    """

    # lowercase
    query = query.lower()

    # remove punctuation
    query = re.sub(r"[^\w\s]", " ", query)

    # remove extra spaces
    query = " ".join(query.split())

    # replace synonyms
    for word, replacement in SYNONYMS.items():
        query = query.replace(word, replacement)

    return query

# ==========================================================
# Retrieve Documents
# ==========================================================

def retrieve(query, top_k=TOP_K):

    try:
        _load_vector_resources()
    except Exception as exc:
        print(f"Vector retrieval unavailable: {exc}")
        return []

    query = preprocess_query(query)

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype(np.float32)

    # Normalize query vector
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        if idx == -1:
            continue

        if score < SIMILARITY_THRESHOLD:
            continue

        if idx < 0 or idx >= len(documents):
            continue

        metadata_entry = metadata[idx] if idx < len(metadata) else {}
        if not isinstance(metadata_entry, dict):
            metadata_entry = {}

        results.append({
            "rank": rank,
            "score": round(float(score), 4),
            "document": documents[idx],
            "metadata": metadata_entry
        })

    return results

# ==========================================================
# Answer Validation
# ==========================================================

def is_answerable(results):

    """
    Returns True only if
    relevant documents exist.
    """

    return len(results) > 0

# ==========================================================
# Build Context
# ==========================================================

def build_context(results):

    if not results:
        return None

    context = []

    for result in results:

        context.append(

            f"""
========================================================
Rank : {result['rank']}
Similarity : {result['score']}

{result['document']}
"""
        )

    return "\n".join(context)

# ==========================================================
# Retrieve + Context
# ==========================================================

def retrieve_context(query):

    results = retrieve(query)

    if not is_answerable(results):

        return {
            "status": False,
            "message": (
                "This information is not available "
                "in the uploaded dataset. "
                "I can't help you with it."
            ),
            "context": None,
            "results": []
        }

    return {

        "status": True,

        "message": "Relevant documents found.",

        "context": build_context(results),

        "results": results

    }

# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    print("\nClimate Health Retriever")
    print("-" * 60)

    while True:

        question = input("\nAsk Question (exit to quit): ")

        if question.lower() == "exit":
            break

        response = retrieve_context(question)

        if not response["status"]:

            print("\n" + response["message"])

            continue

        print("\nRetrieved Documents\n")

        for result in response["results"]:

            print("=" * 60)

            print(f"Rank       : {result['rank']}")
            print(f"Similarity : {result['score']}")
            print(f"Country    : {result['metadata']['country']}")
            print(f"Region     : {result['metadata']['region']}")
            print(f"Year       : {result['metadata']['year']}")

            print("=" * 60)

        print("\nContext Sent to LLM\n")

        print(response["context"])
