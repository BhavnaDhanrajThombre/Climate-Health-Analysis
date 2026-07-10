"""
query_router.py

Parses a raw natural-language question into a structured dictionary that
tells chat_engine.py whether the question is analytical (routed to
data_query.py) or semantic (routed to retriever.py + llm.py).

Beyond the basic operation/column/single-filter detection, this router also
detects:
  - "count distinct" / "unique" requests over categorical dimensions
    (countries, regions, income levels, years, months, weeks)
  - explicit group-by breakdowns ("average AQI by region", "GDP region wise")
  - multi-entity comparisons ("compare India and China AQI",
    "compare AQI between years")
"""

import re
import string
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from data_query import NUMERIC_COLUMNS

BASE_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = BASE_DIR / "data" / "processed" / "climate_health_cleaned.csv"

_DATAFRAME: Optional[pd.DataFrame] = None

MONTH_MAP: Dict[str, int] = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}

# Analytical operations -> trigger keywords. Every key here is a supported
# data_query.py operation, so this set doubles as the analytical-intent list.
OPERATION_KEYWORDS: Dict[str, List[str]] = {
    "top": ["top"],
    "bottom": ["bottom"],
    "mode": ["most common", "most frequent", "mode of"],
    "variance": ["variance"],
    "std": ["standard deviation", "std dev", "stddev", "std deviation"],
    "highest": ["highest", "maximum", "max", "peak", "greatest", "most"],
    "lowest": ["lowest", "minimum", "min", "least", "smallest"],
    "average": ["average", "avg", "mean"],
    "median": ["median"],
    "sum": ["sum", "total"],
    "count": ["count", "how many", "number of"],
    "compare": ["compare", "comparison", "versus", "vs"],
    "trend": ["trend", "over time", "year wise", "year-wise", "yearly"],
}
ANALYTICAL_OPERATIONS = set(OPERATION_KEYWORDS)

# Semantic intents -> trigger keywords, handled by retriever.py + llm.py.
SEMANTIC_KEYWORDS: Dict[str, List[str]] = {
    "describe": ["describe", "description", "overview"],
    "relationship": ["relationship", "correlation", "link between"],
    "impact": ["impact", "influence"],
    "effect": ["effect", "affects", "affect"],
    "summary": ["summary", "summarize"],
    "profile": ["profile"],
}

COLUMN_ALIASES: Dict[str, str] = {
    "temperature anomaly": "temp_anomaly_celsius", "temp anomaly": "temp_anomaly_celsius",
    "temperature": "temperature_celsius", "temp": "temperature_celsius",
    "precipitation": "precipitation_mm", "rainfall": "precipitation_mm", "rain": "precipitation_mm",
    "heat wave days": "heat_wave_days", "heat wave": "heat_wave_days", "heatwave": "heat_wave_days",
    "drought": "drought_indicator", "flood": "flood_indicator",
    "extreme weather events": "extreme_weather_events", "extreme weather": "extreme_weather_events",
    "particulate matter": "pm25_ugm3", "pm2.5": "pm25_ugm3", "pm25": "pm25_ugm3",
    "air quality index": "air_quality_index", "air quality": "air_quality_index", "aqi": "air_quality_index",
    "respiratory disease": "respiratory_disease_rate", "respiratory": "respiratory_disease_rate",
    "cardiovascular mortality": "cardio_mortality_rate", "cardio mortality": "cardio_mortality_rate",
    "cardio": "cardio_mortality_rate",
    "vector disease risk": "vector_disease_risk_score", "vector disease": "vector_disease_risk_score",
    "waterborne disease": "waterborne_disease_incidents", "waterborne": "waterborne_disease_incidents",
    "heat related admissions": "heat_related_admissions", "heat admissions": "heat_related_admissions",
    "healthcare access": "healthcare_access_index", "health access": "healthcare_access_index",
    "healthcare": "healthcare_access_index",
    "gdp per capita": "gdp_per_capita_usd", "gdp": "gdp_per_capita_usd",
    "mental health": "mental_health_index",
    "food security": "food_security_index",
    "population millions": "population_millions", "population": "population_millions",
    "latitude": "latitude", "longitude": "longitude",
    "country code": "country_code",
}

ALL_COLUMNS: List[str] = [
    "record_id", "country_code", "country_name", "region", "income_level", "date", "year", "month", "week",
    "latitude", "longitude", "population_millions",
    "temperature_celsius", "temp_anomaly_celsius", "precipitation_mm", "heat_wave_days",
    "drought_indicator", "flood_indicator", "extreme_weather_events", "pm25_ugm3",
    "air_quality_index", "respiratory_disease_rate", "cardio_mortality_rate",
    "vector_disease_risk_score", "waterborne_disease_incidents", "heat_related_admissions",
    "healthcare_access_index", "gdp_per_capita_usd", "mental_health_index", "food_security_index",
]

# "how many countries" / "distinct regions" / "unique income levels" -> categorical column
ENTITY_ALIASES: Dict[str, str] = {
    "countries": "country_name", "country": "country_name", "nations": "country_name",
    "regions": "region", "region": "region",
    "income levels": "income_level", "income level": "income_level",
    "income groups": "income_level", "income group": "income_level",
    "years": "year", "year": "year",
    "months": "month", "month": "month",
    "weeks": "week", "week": "week",
}
DISTINCT_TRIGGERS = ["how many", "distinct", "unique", "number of", "count of"]

# Explicit group-by phrasing -> the column to group by
GROUP_BY_ALIASES: Dict[str, str] = {
    "by country": "country_name", "country wise": "country_name",
    "per country": "country_name", "each country": "country_name",
    "between countries": "country_name", "across countries": "country_name",
    "by region": "region", "region wise": "region",
    "per region": "region", "each region": "region",
    "between regions": "region", "across regions": "region",
    "by income level": "income_level", "income level wise": "income_level",
    "by income": "income_level", "income wise": "income_level",
    "between income levels": "income_level", "across income levels": "income_level",
    "by year": "year", "year wise": "year", "yearly": "year",
    "per year": "year", "each year": "year",
    "between years": "year", "across years": "year",
    "by month": "month", "month wise": "month", "monthly": "month",
    "per month": "month", "each month": "month",
}


# Common country abbreviations/nicknames -> the canonical name typically
# used in country_name columns. Only unambiguous, safe-to-replace terms are
# included (no bare "us" — too easily confused with the pronoun).
COUNTRY_ALIASES: Dict[str, str] = {
    "usa": "united states", "america": "united states", "the united states": "united states",
    "uk": "united kingdom", "britain": "united kingdom", "great britain": "united kingdom",
}


def load_data() -> pd.DataFrame:
    """Load and cache the cleaned climate-health dataset from disk."""
    global _DATAFRAME
    if _DATAFRAME is None:
        _DATAFRAME = pd.read_csv(CSV_PATH)
    return _DATAFRAME


def _expand_country_aliases(text: str) -> str:
    """Replace common country abbreviations/nicknames with their canonical form."""
    for alias, canonical in sorted(COUNTRY_ALIASES.items(), key=lambda kv: len(kv[0]), reverse=True):
        text = re.sub(rf"(?<!\w){re.escape(alias)}(?!\w)", canonical, text)
    return text


def normalize_query(query: str) -> str:
    """Lowercase, strip punctuation (keeping hyphens), collapse whitespace, and expand country aliases."""
    cleaned = query.lower().translate(str.maketrans("", "", string.punctuation.replace("-", "")))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return _expand_country_aliases(cleaned)


def _unique_values(df: pd.DataFrame, column: str) -> List[str]:
    """Return lowercased unique values for a categorical column."""
    try:
        return sorted({str(value).lower() for value in df[column].dropna().unique()})
    except Exception:
        return []


def _contains_phrase(text: str, phrase: str) -> bool:
    """True if `phrase` appears in `text` on whole-word boundaries (not as a substring of another word)."""
    return re.search(rf"(?<!\w){re.escape(phrase.strip())}(?!\w)", text) is not None


def _fuzzy_find(text: str, candidates: List[str]) -> Optional[str]:
    """Find a single candidate present in `text`: first by whole-word match, then by fuzzy n-gram match."""
    for candidate in sorted(candidates, key=len, reverse=True):
        if candidate and _contains_phrase(text, candidate):
            return candidate

    tokens = text.split()
    for size in range(min(4, len(tokens)), 0, -1):
        for start in range(len(tokens) - size + 1):
            phrase = " ".join(tokens[start:start + size])
            match = get_close_matches(phrase, candidates, n=1, cutoff=0.82)
            if match:
                return match[0]
    return None


def _find_all_entities(text: str, candidates: List[str]) -> List[str]:
    """Find every candidate whole-word-present in `text` (used for multi-entity 'compare' questions)."""
    return [c for c in sorted(candidates, key=len, reverse=True) if c and _contains_phrase(text, c)]


def _detect_years_all(text: str) -> List[int]:
    """Detect every distinct four-digit year mentioned in the query."""
    return sorted({int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", text)})


def _detect_column(text: str) -> str:
    """Detect the dataset column referenced in the query using aliases and fuzzy matching."""
    for alias in sorted(COLUMN_ALIASES, key=len, reverse=True):
        if alias in text:
            return COLUMN_ALIASES[alias]

    for column in ALL_COLUMNS:
        if column.replace("_", " ") in text:
            return column

    readable_columns = [column.replace("_", " ") for column in ALL_COLUMNS]
    match = _fuzzy_find(text, readable_columns)
    return ALL_COLUMNS[readable_columns.index(match)] if match else ""


def _detect_keyword(text: str, keyword_map: Dict[str, List[str]]) -> str:
    """Return the first key in `keyword_map` whose keyword list matches `text`."""
    for name, keywords in keyword_map.items():
        if any(keyword in text for keyword in keywords):
            return name
    return ""


def _detect_count_distinct(text: str) -> Optional[str]:
    """
    Detect a 'count distinct <entity>' request, e.g. 'how many countries',
    'distinct regions', 'unique income levels'. Returns the target
    categorical column, or None if this isn't that kind of question.
    """
    if not any(trigger in text for trigger in DISTINCT_TRIGGERS):
        return None
    for alias in sorted(ENTITY_ALIASES, key=len, reverse=True):
        if _contains_phrase(text, alias):
            return ENTITY_ALIASES[alias]
    return None


def _detect_group_by(text: str) -> Optional[str]:
    """Detect an explicit group-by dimension, e.g. 'by region', 'country wise'."""
    for phrase in sorted(GROUP_BY_ALIASES, key=len, reverse=True):
        if phrase in text:
            return GROUP_BY_ALIASES[phrase]
    return None


def _detect_grouped_extreme(text: str) -> Optional[tuple]:
    """
    Detect a "which group has the highest/lowest AVERAGE of X" question, e.g.
    'average minimum aqi country' (country with the lowest average AQI),
    'maximum aqi country wise' (country with the highest average AQI),
    'region with lowest average gdp'.

    Deliberately narrow: only fires when either (a) an aggregate keyword
    (average/median/sum) AND an entity word/phrase are both present, or
    (b) an explicit group-by phrase ("by country" / "country wise") is
    present alongside a highest/lowest keyword. This keeps plain "highest
    aqi country" (a single record) from being reinterpreted as a group-by
    average just because the word "country" appears in it.

    Returns (aggregate_name, group_by_column, ascending) or None.
    """
    has_lowest = any(keyword in text for keyword in OPERATION_KEYWORDS["lowest"])
    has_highest = any(keyword in text for keyword in OPERATION_KEYWORDS["highest"])
    if not (has_lowest or has_highest):
        return None
    ascending = has_lowest

    explicit_aggregate = None
    for name in ("average", "median", "sum"):
        if any(keyword in text for keyword in OPERATION_KEYWORDS[name]):
            explicit_aggregate = name
            break

    explicit_group_phrase = _detect_group_by(text)

    if explicit_aggregate:
        group_by = explicit_group_phrase
        if not group_by:
            for alias in sorted(ENTITY_ALIASES, key=len, reverse=True):
                if _contains_phrase(text, alias):
                    group_by = ENTITY_ALIASES[alias]
                    break
        if not group_by:
            return None
        return explicit_aggregate, group_by, ascending

    if explicit_group_phrase:
        # No explicit aggregate keyword, but an explicit "by X"/"X wise"
        # phrase plus highest/lowest strongly implies "highest/lowest
        # average X per group" -> default the aggregate to average.
        return "average", explicit_group_phrase, ascending

    return None


def _detect_year(text: str) -> Optional[int]:
    """Detect a four-digit year mentioned in the query."""
    match = re.search(r"\b(19|20)\d{2}\b", text)
    return int(match.group()) if match else None


def _detect_month(text: str) -> Optional[int]:
    """Detect a month name or numeric month mentioned in the query."""
    for name, number in MONTH_MAP.items():
        if re.search(rf"\b{name}\b", text):
            return number
    match = re.search(r"\bmonth\s+(\d{1,2})\b", text)
    if match and 1 <= int(match.group(1)) <= 12:
        return int(match.group(1))
    return None


def _detect_top_n(text: str) -> int:
    """Detect the requested N value for top/bottom operations (default 10)."""
    match = re.search(r"(?:top|bottom)\s+(\d+)", text)
    return int(match.group(1)) if match else 10


def route_query(query: str) -> Dict[str, Any]:
    """Parse a raw user question into a structured router output dictionary."""
    df = load_data()
    text = normalize_query(query)

    countries_found = _find_all_entities(text, _unique_values(df, "country_name"))
    regions_found = _find_all_entities(text, _unique_values(df, "region"))
    income_found = _find_all_entities(text, _unique_values(df, "income_level"))
    years_found = _detect_years_all(text)

    column = _detect_column(text)
    group_by = _detect_group_by(text)
    distinct_column = _detect_count_distinct(text)
    grouped_extreme = _detect_grouped_extreme(text)
    operation = _detect_keyword(text, OPERATION_KEYWORDS)

    compare_group_by: Optional[str] = None
    compare_values: Optional[List[Any]] = None
    aggregate: Optional[str] = None
    extreme: Optional[str] = None

    if distinct_column:
        # "how many countries" beats a plain "count" match on the same text.
        intent, operation, column = "analytical", "count_distinct", distinct_column
    elif operation == "compare":
        intent = "analytical"
        if len(countries_found) >= 2:
            compare_group_by = "country_name"
            compare_values = [c.title() for c in countries_found]
        elif len(regions_found) >= 2:
            compare_group_by = "region"
            compare_values = [r.title() for r in regions_found]
        elif len(income_found) >= 2:
            compare_group_by = "income_level"
            compare_values = [i.title() for i in income_found]
        elif len(years_found) >= 2:
            compare_group_by = "year"
            compare_values = years_found
        else:
            # No explicit pair named (e.g. "compare AQI between years") ->
            # fall back to a full breakdown across the implied dimension.
            compare_group_by = group_by or "country_name"
            compare_values = None
    elif grouped_extreme:
        # "average minimum aqi country" / "maximum aqi country wise" style
        # question: which group has the highest/lowest AVERAGE of a metric.
        intent, operation = "analytical", "grouped_extreme"
        aggregate, group_by, ascending = grouped_extreme
        extreme = "lowest" if ascending else "highest"
    elif operation in ANALYTICAL_OPERATIONS:
        intent = "analytical"
    else:
        semantic_op = _detect_keyword(text, SEMANTIC_KEYWORDS)
        if semantic_op:
            intent, operation = "semantic", semantic_op
        elif column and column in NUMERIC_COLUMNS:
            # No explicit operation keyword (e.g. "aqi of india", "pm2.5 in
            # china"). The most natural reading of a bare numeric-column
            # lookup is an implicit average, rather than "unsupported".
            intent, operation = "analytical", "average"
        else:
            intent = "unsupported"

    # Single-value filters only make sense when we're not already doing a
    # multi-entity compare (in that case compare_values carries the list).
    country = countries_found[0] if len(countries_found) == 1 else None
    region = regions_found[0] if len(regions_found) == 1 else None
    income_level = income_found[0] if len(income_found) == 1 else None
    year = years_found[0] if len(years_found) == 1 else _detect_year(text)

    return {
        "original_query": query,
        "normalized_query": text,
        "intent": intent,
        "operation": operation,
        "column": column,
        "group_by": group_by,
        "aggregate": aggregate,
        "extreme": extreme,
        "compare_group_by": compare_group_by,
        "compare_values": compare_values,
        "country": country.title() if country else None,
        "region": region.title() if region else None,
        "income_level": income_level.title() if income_level else None,
        "year": year,
        "month": _detect_month(text),
        "top_n": _detect_top_n(text),
    }


if __name__ == "__main__":
    import json

    while True:
        user_query = input("Ask a question (or 'exit'): ").strip()
        if user_query.lower() == "exit":
            break
        print(json.dumps(route_query(user_query), indent=2, default=str))