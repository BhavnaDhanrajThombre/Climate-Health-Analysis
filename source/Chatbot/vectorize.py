
"""
vectorize.py

Creates a FAISS vector database from the cleaned Climate Health dataset.

Run:
python source/chatbot/vectorize.py
"""

from pathlib import Path
import pickle

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------------------------------------------
# Paths
# ---------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

CSV_PATH = BASE_DIR / "data" / "processed" / "climate_health_cleaned.csv"

VECTOR_DB_DIR = Path(__file__).parent / "vector_db"
VECTOR_DB_DIR.mkdir(exist_ok=True)

FAISS_PATH = VECTOR_DB_DIR / "climate_index.faiss"
METADATA_PATH = VECTOR_DB_DIR / "metadata.pkl"

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

print("=" * 70)
print("Loading Cleaned Climate Health Dataset...")
print("=" * 70)

df = pd.read_csv(CSV_PATH)

print(f"Dataset Shape : {df.shape}")

# ---------------------------------------------------
# Load Embedding Model
# ---------------------------------------------------

print("\nLoading Sentence Transformer Model...")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

# ---------------------------------------------------
# Convert Rows into Documents
# ---------------------------------------------------

print("\nCreating Documents for Vector Database...")

documents = []
metadata = []

for _, row in tqdm(df.iterrows(), total=len(df)):

    document = f"""
Climate Health Report

Country: {row['country_name']}
Region: {row['region']}
Income Level: {row['income_level']}
Date: {row['date']}
Year: {row['year']}
Month: {row['month']}

Climate Indicators:
- Average temperature is {row['temperature_celsius']} °C.
- Temperature anomaly is {row['temp_anomaly_celsius']} °C.
- Annual precipitation is {row['precipitation_mm']} mm.
- Heat wave days recorded: {row['heat_wave_days']}.
- Drought indicator: {row['drought_indicator']}.
- Flood indicator: {row['flood_indicator']}.
- Extreme weather events recorded: {row['extreme_weather_events']}.

Air Quality:
- PM2.5 concentration is {row['pm25_ugm3']} µg/m³.
- Air Quality Index (AQI) is {row['air_quality_index']}.

Health Indicators:
- Respiratory disease rate is {row['respiratory_disease_rate']}.
- Cardiovascular mortality rate is {row['cardio_mortality_rate']}.
- Vector-borne disease risk score is {row['vector_disease_risk_score']}.
- Waterborne disease incidents: {row['waterborne_disease_incidents']}.
- Heat-related hospital admissions: {row['heat_related_admissions']}.

Socioeconomic Indicators:
- Healthcare access index is {row['healthcare_access_index']}.
- GDP per capita is {row['gdp_per_capita_usd']} USD.
- Mental health index is {row['mental_health_index']}.
- Food security index is {row['food_security_index']}.
"""

    documents.append(document)

    metadata.append({
        "record_id": int(row["record_id"]),
        "country": row["country_name"],
        "region": row["region"],
        "income_level": row["income_level"],
        "year": int(row["year"]),
        "month": int(row["month"]),
        "date": row["date"]
    })

print(f"\nDocuments Created : {len(documents)}")

# ---------------------------------------------------
# Generate Embeddings
# ---------------------------------------------------

print("\nGenerating Embeddings...")

embeddings = model.encode(
    documents,
    convert_to_numpy=True,
    show_progress_bar=True
)

embeddings = embeddings.astype(np.float32)

print(f"\nEmbedding Shape : {embeddings.shape}")
print(f"Embedding Dimension : {embeddings.shape[1]}")

# ---------------------------------------------------
# Normalize Embeddings
# ---------------------------------------------------

print("\nNormalizing Embeddings...")

faiss.normalize_L2(embeddings)

# ---------------------------------------------------
# Create FAISS Index
# ---------------------------------------------------

print("\nCreating FAISS Vector Database...")

dimension = embeddings.shape[1]

# Inner Product + Normalized Embeddings = Cosine Similarity
index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print(f"Vectors Stored : {index.ntotal}")

# ---------------------------------------------------
# Save Vector Database
# ---------------------------------------------------

print("\nSaving Vector Database...")

faiss.write_index(index, str(FAISS_PATH))

with open(METADATA_PATH, "wb") as f:

    pickle.dump(
        {
            "documents": documents,
            "metadata": metadata,
            "model_name": MODEL_NAME
        },
        f,
    )

# ---------------------------------------------------
# Summary
# ---------------------------------------------------

print("\n" + "=" * 70)
print("VECTOR DATABASE CREATED SUCCESSFULLY")
print("=" * 70)

print(f"Dataset Records      : {len(documents)}")
print(f"Embedding Model      : {MODEL_NAME}")
print(f"Embedding Dimension  : {dimension}")
print(f"FAISS Index Size     : {index.ntotal}")

print("\nSaved Files")
print(f"FAISS Index : {FAISS_PATH}")
print(f"Metadata    : {METADATA_PATH}")

print("\nReady for Retrieval!")