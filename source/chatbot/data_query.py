"""
data_query.py

Executes structured analytical queries against the cleaned climate-health
dataset, using the structured output of query_router.py. Supports:
  - row-level highest/lowest, top-N/bottom-N
  - scalar aggregates: average, median, sum, mode, variance, std
  - the same aggregates broken down by an explicit group_by dimension
    (country/region/income_level/year/month), e.g. "average AQI by region"
  - count / count_distinct (distinct countries, regions, income levels,
    years, months, weeks)
  - trend (yearly, or another group_by dimension)
  - compare: either a named list of entities (2+ countries/regions/income
    levels/years) or a full breakdown across an implied dimension
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = BASE_DIR / "data" / "processed" / "climate_health_cleaned.csv"

NUMERIC_COLUMNS = {
    "population_millions",
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

# Valid group-by / count-distinct dimensions.
CATEGORICAL_COLUMNS = {"country_name", "region", "income_level", "year", "month", "week"}

# operation -> pandas Series method name, for the simple aggregate cases
AGGREGATE_OPS = {"average": "mean", "median": "median", "sum": "sum", "variance": "var", "std": "std"}

SUPPORTED_OPERATIONS = set(AGGREGATE_OPS) | {
    "mode", "highest", "lowest", "count", "count_distinct", "top", "bottom", "compare", "trend",
    "grouped_extreme",
}

FALLBACK_MESSAGE = "This information is not available in the uploaded dataset."

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


def _fail(message: str = FALLBACK_MESSAGE) -> Dict[str, Any]:
    """Build a standardized failure response."""
    return {"status": False, "operation": "", "column": "", "rows": 0, "data": [], "message": message}


def _success(operation: str, column: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a standardized success response."""
    return {"status": True, "operation": operation, "column": column, "rows": len(data), "data": data, "message": ""}


def validate_router(router: Dict[str, Any], df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Return a failure response if the router output can't be executed, else None."""
    if df.empty:
        return _fail()

    operation = router.get("operation")
    if operation not in SUPPORTED_OPERATIONS:
        return _fail()

    if operation == "count":
        return None

    if operation == "count_distinct":
        column = router.get("column")
        if not column or column not in df.columns:
            return _fail()
        return None

    if operation == "compare":
        column = router.get("column")
        group_by = router.get("compare_group_by")
        if not column or column not in NUMERIC_COLUMNS:
            return _fail()
        if not group_by or group_by not in df.columns:
            return _fail()
        return None

    if operation == "grouped_extreme":
        column = router.get("column")
        group_by = router.get("group_by")
        aggregate = router.get("aggregate")
        extreme = router.get("extreme")
        if not column or column not in NUMERIC_COLUMNS:
            return _fail()
        if not group_by or group_by not in df.columns:
            return _fail()
        if aggregate not in AGGREGATE_OPS:
            return _fail()
        if extreme not in {"lowest", "highest"}:
            return _fail()
        return None

    column = router.get("column")
    if not column or column not in df.columns:
        return _fail()
    if operation in AGGREGATE_OPS or operation == "trend":
        if column not in NUMERIC_COLUMNS:
            return _fail()

    group_by = router.get("group_by")
    if group_by and group_by not in df.columns:
        return _fail()

    return None


def apply_filters(df: pd.DataFrame, router: Dict[str, Any]) -> pd.DataFrame:
    """Apply optional country, region, income-level, year, and month filters."""
    filtered = df

    text_filters = {
        "country_name": router.get("country"),
        "region": router.get("region"),
        "income_level": router.get("income_level"),
    }
    for column, value in text_filters.items():
        if value and column in filtered.columns:
            filtered = filtered[filtered[column].astype(str).str.lower() == str(value).lower()]

    for column in ("year", "month"):
        value = router.get(column)
        if value is not None and column in filtered.columns:
            filtered = filtered[filtered[column] == value]

    return filtered


def _row_result(df: pd.DataFrame, column: str, operation: str, ascending: bool) -> Dict[str, Any]:
    """Return the full row at the min/max of `column`."""
    idx = df[column].idxmin() if ascending else df[column].idxmax()
    return _success(operation, column, [df.loc[idx].to_dict()])


def _sorted_result(df: pd.DataFrame, column: str, operation: str, n: int, ascending: bool) -> Dict[str, Any]:
    """Return the top/bottom N rows sorted by `column`."""
    result = df.sort_values(by=column, ascending=ascending).head(n)
    return _success(operation, column, result.to_dict(orient="records"))


def _aggregate_result(df: pd.DataFrame, column: str, operation: str) -> Dict[str, Any]:
    """Return a single scalar aggregate (average / median / sum / variance / std) for `column`."""
    value = round(float(getattr(df[column], AGGREGATE_OPS[operation])()), 2)
    return _success(operation, column, [{column: value}])


def _grouped_aggregate_result(df: pd.DataFrame, column: str, operation: str, group_by: str) -> Dict[str, Any]:
    """Return the aggregate of `column` broken down by `group_by`, sorted descending."""
    method = AGGREGATE_OPS[operation]
    result = (
        getattr(df.groupby(group_by)[column], method)()
        .round(2)
        .reset_index()
        .sort_values(column, ascending=False)
    )
    return _success(operation, column, result.to_dict(orient="records"))


def _mode_result(df: pd.DataFrame, column: str, group_by: Optional[str] = None) -> Dict[str, Any]:
    """Return the most frequent value of `column` (numeric or categorical), optionally by `group_by`."""

    def _first_mode(series: pd.Series) -> Any:
        modes = series.mode()
        if modes.empty:
            return None
        value = modes.iloc[0]
        return round(float(value), 2) if pd.api.types.is_numeric_dtype(series) else str(value)

    if group_by:
        result = df.groupby(group_by)[column].apply(_first_mode).reset_index(name=column)
        return _success("mode", column, result.to_dict(orient="records"))

    mode_value = _first_mode(df[column])
    if mode_value is None:
        return _fail()
    return _success("mode", column, [{column: mode_value}])


def _trend_result(df: pd.DataFrame, column: str, group_by: str = "year") -> Dict[str, Any]:
    """Return the average of `column` over time (by year, or another group_by dimension)."""
    result = df.groupby(group_by)[column].mean().round(2).reset_index().sort_values(group_by)
    return _success("trend", column, result.to_dict(orient="records"))


def _compare_result(
    df: pd.DataFrame, column: str, group_by: str, compare_values: Optional[List[Any]]
) -> Dict[str, Any]:
    """
    Compare per-group averages of `column`. If `compare_values` is given
    (e.g. ["India", "China"] or [2019, 2023]), restrict to just those
    groups; otherwise return a full breakdown across every group.
    """
    working = df
    if compare_values:
        if group_by in {"country_name", "region", "income_level"}:
            wanted = {str(v).lower() for v in compare_values}
            working = df[df[group_by].astype(str).str.lower().isin(wanted)]
        else:
            working = df[df[group_by].isin(compare_values)]
        if working.empty:
            return _fail()

    result = (
        working.groupby(group_by)[column]
        .mean()
        .round(2)
        .reset_index()
        .sort_values(column, ascending=False)
    )
    return _success("compare", column, result.to_dict(orient="records"))


def _grouped_extreme_result(
    df: pd.DataFrame, column: str, group_by: str, aggregate: str, extreme: str
) -> Dict[str, Any]:
    """
    Return the single group with the highest/lowest aggregate of `column`,
    e.g. the country with the lowest *average* AQI (as opposed to `_row_result`,
    which finds the single row with the lowest recorded AQI value).
    """
    method = AGGREGATE_OPS[aggregate]
    grouped = getattr(df.groupby(group_by)[column], method)().round(2)
    if grouped.empty:
        return _fail()

    idx = grouped.idxmin() if extreme == "lowest" else grouped.idxmax()
    operation_label = f"{extreme}_{aggregate}"
    return _success(operation_label, column, [{group_by: idx, column: float(grouped.loc[idx])}])


def _count_result(df: pd.DataFrame) -> Dict[str, Any]:
    """Return the row count of the filtered dataframe."""
    return _success("count", "", [{"count": int(len(df))}])


def _count_distinct_result(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Return the number of distinct values in `column`, plus the sorted list of values."""
    values = sorted({str(v) for v in df[column].dropna().unique()})
    return _success("count_distinct", column, [{"distinct_count": len(values), "values": values}])


def execute_query(router: Dict[str, Any]) -> Dict[str, Any]:
    """Execute an analytical query based on the router output dictionary."""
    try:
        df = load_data()

        error = validate_router(router, df)
        if error is not None:
            return error

        filtered = apply_filters(df, router)
        if filtered.empty:
            return _fail()

        operation = router.get("operation")
        column = router.get("column", "")
        group_by = router.get("group_by")
        n = int(router.get("top_n", 10) or 10)

        if operation == "count":
            return _count_result(filtered)
        if operation == "count_distinct":
            return _count_distinct_result(filtered, column)
        if operation == "highest":
            return _row_result(filtered, column, operation, ascending=False)
        if operation == "lowest":
            return _row_result(filtered, column, operation, ascending=True)
        if operation == "grouped_extreme":
            return _grouped_extreme_result(
                filtered, column, group_by, router.get("aggregate"), router.get("extreme")
            )
        if operation == "mode":
            return _mode_result(filtered, column, group_by)
        if operation in AGGREGATE_OPS:
            if group_by:
                return _grouped_aggregate_result(filtered, column, operation, group_by)
            return _aggregate_result(filtered, column, operation)
        if operation == "top":
            return _sorted_result(filtered, column, operation, n, ascending=False)
        if operation == "bottom":
            return _sorted_result(filtered, column, operation, n, ascending=True)
        if operation == "trend":
            return _trend_result(filtered, column, group_by or "year")
        if operation == "compare":
            return _compare_result(filtered, column, router.get("compare_group_by"), router.get("compare_values"))

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
        print(json.dumps(execute_query(route_query(question)), indent=2, default=str))