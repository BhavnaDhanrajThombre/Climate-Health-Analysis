import re
import string
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = BASE_DIR / "data" / "processed" / "climate_health_cleaned.csv"

_DATAFRAME: Optional[pd.DataFrame] = None

MONTH_MAP: Dict[str, int] = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

OPERATION_KEYWORDS: Dict[str, List[str]] = {
    "top": ["top"],
    "bottom": ["bottom"],
    "highest": ["highest", "maximum", "max", "peak", "greatest", "most"],
    "lowest": ["lowest", "minimum", "min", "least", "smallest"],
    "average": ["average", "avg", "mean"],
    "median": ["median"],
    "sum": ["sum", "total"],
    "count": ["count", "how many", "number of"],
    "compare": ["compare", "comparison", "versus", " vs "],
    "trend": ["trend", "over time", "year wise", "year-wise", "yearly"],
    "describe": ["describe", "description", "overview"],
}

ANALYTICAL_OPERATIONS = {
    "highest", "lowest", "average", "median", "sum", "count", "top", "bottom", "compare", "trend",
}

SEMANTIC_KEYWORDS: Dict[str, List[str]] = {
    "describe": ["describe", "description"],
    "relationship": ["relationship", "correlation", "link between"],
    "impact": ["impact", "influence"],
    "effect": ["effect", "affects", "affect"],
    "summary": ["summary", "summarize"],
    "profile": ["profile"],
}

COLUMN_ALIASES: Dict[str, str] = {
    "temperature": "temperature_celsius",
    "temp": "temperature_celsius",
    "temperature anomaly": "temp_anomaly_celsius",
    "temp anomaly": "temp_anomaly_celsius",
    "precipitation": "precipitation_mm",
    "rainfall": "precipitation_mm",
    "rain": "precipitation_mm",
    "heat wave": "heat_wave_days",
    "heatwave": "heat_wave_days",
    "heat wave days": "heat_wave_days",
    "drought": "drought_indicator",
    "flood": "flood_indicator",
    "extreme weather": "extreme_weather_events",
    "extreme weather events": "extreme_weather_events",
    "pm2.5": "pm25_ugm3",
    "pm25": "pm25_ugm3",
    "particulate matter": "pm25_ugm3",
    "air quality": "air_quality_index",
    "aqi": "air_quality_index",
    "respiratory disease": "respiratory_disease_rate",
    "respiratory": "respiratory_disease_rate",
    "cardio mortality": "cardio_mortality_rate",
    "cardiovascular mortality": "cardio_mortality_rate",
    "cardio": "cardio_mortality_rate",
    "vector disease": "vector_disease_risk_score",
    "vector disease risk": "vector_disease_risk_score",
    "waterborne disease": "waterborne_disease_incidents",
    "waterborne": "waterborne_disease_incidents",
    "heat related admissions": "heat_related_admissions",
    "heat admissions": "heat_related_admissions",
    "healthcare access": "healthcare_access_index",
    "health access": "healthcare_access_index",
    "gdp": "gdp_per_capita_usd",
    "gdp per capita": "gdp_per_capita_usd",
    "mental health": "mental_health_index",
    "food security": "food_security_index",
}

ALL_COLUMNS: List[str] = [
    "record_id", "country_name", "region", "income_level", "date", "year", "month",
    "temperature_celsius", "temp_anomaly_celsius", "precipitation_mm", "heat_wave_days",
    "drought_indicator", "flood_indicator", "extreme_weather_events", "pm25_ugm3",
    "air_quality_index", "respiratory_disease_rate", "cardio_mortality_rate",
    "vector_disease_risk_score", "waterborne_disease_incidents", "heat_related_admissions",
    "healthcare_access_index", "gdp_per_capita_usd", "mental_health_index", "food_security_index",
]


def load_data() -> pd.DataFrame:
    """Load and cache the cleaned climate-health dataset from disk."""
    global _DATAFRAME
    if _DATAFRAME is None:
        _DATAFRAME = pd.read_csv(CSV_PATH)
    return _DATAFRAME


def normalize_query(query: str) -> str:
    """Lowercase and strip punctuation from the raw user query."""
    lowered = query.lower().strip()
    cleaned = lowered.translate(str.maketrans("", "", string.punctuation.replace("-", "")))
    return re.sub(r"\s+", " ", cleaned).strip()


def _get_unique_values(df: pd.DataFrame, column: str) -> List[str]:
    """Return lowercased unique values for a categorical column."""
    try:
        return sorted({str(value).lower() for value in df[column].dropna().unique()})
    except Exception:
        return []


def _fuzzy_find(normalized_query: str, candidates: List[str]) -> Optional[str]:
    """Find the best matching candidate present in the query via substring or fuzzy matching."""
    for candidate in sorted(candidates, key=len, reverse=True):
        if candidate and candidate in normalized_query:
            return candidate

    tokens = normalized_query.split()
    for size in range(min(4, len(tokens)), 0, -1):
        for start in range(len(tokens) - size + 1):
            phrase = " ".join(tokens[start:start + size])
            matches = get_close_matches(phrase, candidates, n=1, cutoff=0.82)
            if matches:
                return matches[0]
    return None


def _detect_column(normalized_query: str) -> str:
    """Detect the dataset column referenced in the query using aliases and fuzzy matching."""
    for alias in sorted(COLUMN_ALIASES.keys(), key=len, reverse=True):
        if alias in normalized_query:
            return COLUMN_ALIASES[alias]

    for column in ALL_COLUMNS:
        readable = column.replace("_", " ")
        if readable in normalized_query:
            return column

    tokens = normalized_query.split()
    readable_columns = [column.replace("_", " ") for column in ALL_COLUMNS]
    for size in range(min(4, len(tokens)), 0, -1):
        for start in range(len(tokens) - size + 1):
            phrase = " ".join(tokens[start:start + size])
            matches = get_close_matches(phrase, readable_columns, n=1, cutoff=0.82)
            if matches:
                index = readable_columns.index(matches[0])
                return ALL_COLUMNS[index]
    return ""


def _detect_operation(normalized_query: str) -> str:
    """Detect the analytical or descriptive operation requested in the query."""
    for operation, keywords in OPERATION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in normalized_query:
                return operation
    return ""


def _detect_semantic_intent(normalized_query: str) -> str:
    """Detect a semantic intent keyword present in the query."""
    for intent_name, keywords in SEMANTIC_KEYWORDS.items():
        for keyword in keywords:
            if keyword in normalized_query:
                return intent_name
    return ""


def _detect_year(normalized_query: str) -> Optional[int]:
    """Detect a four-digit year mentioned in the query."""
    match = re.search(r"\b(19|20)\d{2}\b", normalized_query)
    return int(match.group()) if match else None


def _detect_month(normalized_query: str) -> Optional[int]:
    """Detect a month name or numeric month mentioned in the query."""
    for name, number in MONTH_MAP.items():
        if re.search(rf"\b{name}\b", normalized_query):
            return number
    match = re.search(r"\bmonth\s+(\d{1,2})\b", normalized_query)
    if match:
        value = int(match.group(1))
        if 1 <= value <= 12:
            return value
    return None


def _detect_top_n(normalized_query: str) -> int:
    """Detect the requested N value for top or bottom operations."""
    match = re.search(r"(?:top|bottom)\s+(\d+)", normalized_query)
    if match:
        return int(match.group(1))
    return 10


def route_query(query: str) -> Dict[str, Any]:
    """Parse a raw user question into a structured router output dictionary."""
    df = load_data()
    normalized = normalize_query(query)

    countries = _get_unique_values(df, "country_name")
    regions = _get_unique_values(df, "region")
    income_levels = _get_unique_values(df, "income_level")

    country_match = _fuzzy_find(normalized, countries)
    region_match = _fuzzy_find(normalized, regions)
    income_match = _fuzzy_find(normalized, income_levels)

    column = _detect_column(normalized)
    operation = _detect_operation(normalized)
    year = _detect_year(normalized)
    month = _detect_month(normalized)
    top_n = _detect_top_n(normalized)

    if operation in ANALYTICAL_OPERATIONS:
        intent = "analytical"
    else:
        semantic_match = _detect_semantic_intent(normalized)
        if semantic_match:
            intent = "semantic"
            operation = semantic_match
        else:
            intent = "unsupported"

    return {
        "original_query": query,
        "normalized_query": normalized,
        "intent": intent,
        "operation": operation,
        "column": column,
        "country": country_match.title() if country_match else None,
        "region": region_match.title() if region_match else None,
        "income_level": income_match.title() if income_match else None,
        "year": year,
        "month": month,
        "top_n": top_n,
    }


if __name__ == "__main__":
    import json

    while True:
        user_query = input("Ask a question (or 'exit'): ").strip()
        if user_query.lower() == "exit":
            break
        result = route_query(user_query)
        print(json.dumps(result, indent=2, default=str))
