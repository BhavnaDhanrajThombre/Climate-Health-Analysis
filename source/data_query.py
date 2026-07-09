import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = BASE_DIR / "data" / "processed" / "climate_health_cleaned.csv"

NUMERIC_COLUMNS = {
    "temperature_celsius",
    "temp_anomaly_celsius",
    "precipitation_mm",
    "heat_wave_days",
    "pm25_ugm3",
    "air_quality_index",
    "respiratory_disease_rate",
    "cardio_mortality_rate",
    "vector_disease_risk_score",
    "waterborne_disease_incidents",
    "heat_related_admissions",
    "healthcare_access_index",
    "gdp_per_capita_usd",
    "mental_health_index",
    "food_security_index",
}

SUPPORTED_OPERATIONS = {
    "highest",
    "lowest",
    "average",
    "median",
    "sum",
    "count",
    "top",
    "bottom",
    "compare",
    "trend",
}

_DATAFRAME: Optional[pd.DataFrame] = None


def load_data() -> pd.DataFrame:
    """Load and cache the cleaned climate-health dataset from disk."""
    global _DATAFRAME
    if _DATAFRAME is None:
        try:
            _DATAFRAME = pd.read_csv(CSV_PATH)
        except Exception as exc:
            logger.error("Failed to load dataset: %s", exc)
            _DATAFRAME = pd.DataFrame()
    return _DATAFRAME


def _fail(message: str = "This information is not available in the uploaded dataset.") -> Dict[str, Any]:
    """Build a standardized failure response."""
    return {"status": False, "operation": "", "column": "", "rows": 0, "data": [], "message": message}


def _success(operation: str, column: str, data: List[Dict[str, Any]], message: str = "") -> Dict[str, Any]:
    """Build a standardized success response."""
    return {
        "status": True,
        "operation": operation,
        "column": column,
        "rows": len(data),
        "data": data,
        "message": message,
    }


def validate_router(router: Dict[str, Any], df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Validate router output before executing any operation."""
    if df is None or df.empty:
        return _fail()

    operation = router.get("operation")
    if operation not in SUPPORTED_OPERATIONS:
        return _fail()

    column = router.get("column")
    if operation == "count":
        return None

    if not column or column not in df.columns:
        return _fail()

    if operation in {"average", "median", "sum", "trend", "compare"} and column not in NUMERIC_COLUMNS:
        return _fail()

    return None


def apply_filters(df: pd.DataFrame, router: Dict[str, Any]) -> pd.DataFrame:
    """Apply optional country, region, income level, year, and month filters."""
    filtered = df.copy()

    country = router.get("country")
    if country:
        filtered = filtered[filtered["country_name"].str.lower() == str(country).lower()]

    region = router.get("region")
    if region:
        filtered = filtered[filtered["region"].str.lower() == str(region).lower()]

    income_level = router.get("income_level")
    if income_level:
        filtered = filtered[filtered["income_level"].str.lower() == str(income_level).lower()]

    year = router.get("year")
    if year is not None:
        filtered = filtered[filtered["year"] == year]

    month = router.get("month")
    if month is not None:
        filtered = filtered[filtered["month"] == month]

    return filtered


def highest(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Return the complete row having the maximum value of the given column."""
    try:
        if df.empty or column not in df.columns:
            return _fail()
        row = df.loc[df[column].idxmax()]
        return _success("highest", column, [row.to_dict()])
    except Exception as exc:
        logger.error("highest() failed: %s", exc)
        return _fail()


def lowest(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Return the complete row having the minimum value of the given column."""
    try:
        if df.empty or column not in df.columns:
            return _fail()
        row = df.loc[df[column].idxmin()]
        return _success("lowest", column, [row.to_dict()])
    except Exception as exc:
        logger.error("lowest() failed: %s", exc)
        return _fail()


def average(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Return the mean of the given numeric column rounded to two decimals."""
    try:
        if df.empty or column not in NUMERIC_COLUMNS:
            return _fail()
        value = round(float(df[column].mean()), 2)
        return _success("average", column, [{column: value}])
    except Exception as exc:
        logger.error("average() failed: %s", exc)
        return _fail()


def median(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Return the median of the given numeric column rounded to two decimals."""
    try:
        if df.empty or column not in NUMERIC_COLUMNS:
            return _fail()
        value = round(float(df[column].median()), 2)
        return _success("median", column, [{column: value}])
    except Exception as exc:
        logger.error("median() failed: %s", exc)
        return _fail()


def sum_values(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Return the total sum of the given numeric column rounded to two decimals."""
    try:
        if df.empty or column not in NUMERIC_COLUMNS:
            return _fail()
        value = round(float(df[column].sum()), 2)
        return _success("sum", column, [{column: value}])
    except Exception as exc:
        logger.error("sum_values() failed: %s", exc)
        return _fail()


def count_rows(df: pd.DataFrame, column: str = "") -> Dict[str, Any]:
    """Return the row count of the filtered dataframe."""
    try:
        if df.empty:
            return _fail()
        return _success("count", column, [{"count": int(len(df))}])
    except Exception as exc:
        logger.error("count_rows() failed: %s", exc)
        return _fail()


def top_n(df: pd.DataFrame, column: str, n: int) -> Dict[str, Any]:
    """Return the top N rows sorted descending by the given column."""
    try:
        if df.empty or column not in df.columns:
            return _fail()
        result = df.sort_values(by=column, ascending=False).head(n)
        return _success("top", column, result.to_dict(orient="records"))
    except Exception as exc:
        logger.error("top_n() failed: %s", exc)
        return _fail()


def bottom_n(df: pd.DataFrame, column: str, n: int) -> Dict[str, Any]:
    """Return the bottom N rows sorted ascending by the given column."""
    try:
        if df.empty or column not in df.columns:
            return _fail()
        result = df.sort_values(by=column, ascending=True).head(n)
        return _success("bottom", column, result.to_dict(orient="records"))
    except Exception as exc:
        logger.error("bottom_n() failed: %s", exc)
        return _fail()


def trend(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Group by year and return the yearly average of the given numeric column."""
    try:
        if df.empty or column not in NUMERIC_COLUMNS:
            return _fail()
        result = (
            df.groupby("year")[column]
            .mean()
            .round(2)
            .reset_index()
            .sort_values(by="year")
        )
        return _success("trend", column, result.to_dict(orient="records"))
    except Exception as exc:
        logger.error("trend() failed: %s", exc)
        return _fail()


def compare(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Compare country-wise averages of the selected numeric column."""
    try:
        if df.empty or column not in NUMERIC_COLUMNS:
            return _fail()
        result = (
            df.groupby("country_name")[column]
            .mean()
            .round(2)
            .reset_index()
            .sort_values(by=column, ascending=False)
        )
        return _success("compare", column, result.to_dict(orient="records"))
    except Exception as exc:
        logger.error("compare() failed: %s", exc)
        return _fail()


def execute_query(router: Dict[str, Any]) -> Dict[str, Any]:
    """Execute an analytical query based on the router output dictionary."""
    try:
        df = load_data()

        validation_error = validate_router(router, df)
        if validation_error is not None:
            return validation_error

        filtered_df = apply_filters(df, router)
        if filtered_df.empty:
            return _fail()

        operation = router.get("operation")
        column = router.get("column", "")
        top_n_value = router.get("top_n", 10) or 10

        if operation == "highest":
            return highest(filtered_df, column)
        if operation == "lowest":
            return lowest(filtered_df, column)
        if operation == "average":
            return average(filtered_df, column)
        if operation == "median":
            return median(filtered_df, column)
        if operation == "sum":
            return sum_values(filtered_df, column)
        if operation == "count":
            return count_rows(filtered_df, column)
        if operation == "top":
            return top_n(filtered_df, column, int(top_n_value))
        if operation == "bottom":
            return bottom_n(filtered_df, column, int(top_n_value))
        if operation == "trend":
            return trend(filtered_df, column)
        if operation == "compare":
            return compare(filtered_df, column)

        return _fail()
    except Exception as exc:
        logger.error("execute_query() failed: %s", exc)
        return _fail()


if __name__ == "__main__":
    import json

    from query_router import route_query

    while True:
        question = input("Ask a question (or 'exit'): ").strip()
        if question.lower() == "exit":
            break
        router_output = route_query(question)
        response = execute_query(router_output)
        print(json.dumps(response, indent=2, default=str))
