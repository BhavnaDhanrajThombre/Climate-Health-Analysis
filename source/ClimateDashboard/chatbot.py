

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

MAX_HISTORY_MESSAGES = 12  # keep conversation compact

SYSTEM_PROMPT = """You are Climate Health AI, the analytical assistant embedded in a Climate Health Impact Analytics Dashboard covering global climate and public health data from 2015-2025.

Your responsibility is to help users understand the data currently displayed in the dashboard.

You must answer using only:
1. The supplied dataset context.
2. The current dashboard filter context.
3. The supplied Python-computed analytical results.

Rules:
- Never fabricate data. Never invent statistics, countries, regions, years or values.
- Never assume missing information.
- Use exact computed values when provided; do not re-estimate numbers yourself.
- Clearly distinguish averages, totals, rates and indices, and be careful with units.
- Explain calculations when useful.
- Correlation must not be presented as causation. Use language like "correlated with",
  "associated with", or "appears related to" instead of "causes", "proves", or "leads to".
- If the requested information cannot be determined from the supplied data, say:
  "I don't have enough information in the current dashboard data to answer that reliably."
- Keep responses concise but informative. Use bullet points for comparisons and
  markdown tables only when they truly improve clarity.
- Mention the active filters when they materially affect the answer.
- If asked about risk, explain the dashboard's existing risk methodology (normalized
  average of respiratory disease rate, cardio mortality rate, AQI, and heat-related
  admissions, categorized by 33rd/66th percentile thresholds into Low/Medium/High Risk)
  rather than inventing a new one.
- Do not give medical diagnosis or individualized medical advice — this is an analytical
  dashboard assistant, not a medical professional.
- If the question is unrelated to climate, health, or the supplied dataset, politely
  redirect the user to supported dashboard topics.
"""

SUGGESTED_QUESTIONS = [
    "What are the top 5 high-risk countries?",
    "Which region has the highest AQI?",
    "How is temperature related to respiratory disease?",
    "Summarize the current filtered data.",
]


# =========================================================================
# GROQ CLIENT
# =========================================================================
@st.cache_resource
def get_groq_client():
    if not GROQ_API_KEY:
        return None
    try:
        from groq import Groq
        return Groq(api_key=GROQ_API_KEY)
    except Exception:
        return None


def _dashboard_module():
    """Lazily fetch the already-imported dashboard module to reuse its
    existing compute_risk() / CORR_COLS without duplicating logic."""
    return sys.modules.get("__main__") or sys.modules.get("dashboard")


# =========================================================================
# FILTER CONTEXT
# =========================================================================
def get_active_filter_context():
    ss = st.session_state
    country = ss.get("f_country", "All")
    region = ss.get("f_region", "All")
    income = ss.get("f_income", "All")
    year = ss.get("f_year")
    month = ss.get("f_month", "All")
    temp = ss.get("f_temp")
    aqi = ss.get("f_aqi")

    lines = [
        f"Country: {country}",
        f"Region: {region}",
        f"Income Level: {income}",
        f"Year Range: {year[0]}\u2013{year[1]}" if year else "Year Range: full available range",
        f"Month: {month}",
        f"Temperature: {temp[0]:.1f}\u00b0C to {temp[1]:.1f}\u00b0C" if temp else "Temperature: full range",
        f"AQI: {aqi[0]:.1f} to {aqi[1]:.1f}" if aqi else "AQI: full range",
    ]
    return "\n".join(lines)


def _filters_summary_line():
    ss = st.session_state
    year = ss.get("f_year")
    return (
        f"Country: {ss.get('f_country', 'All')} | "
        f"Region: {ss.get('f_region', 'All')} | "
        f"Income: {ss.get('f_income', 'All')} | "
        f"Year: {year[0]}\u2013{year[1]}" if year else "All filters: default"
    )


# =========================================================================
# LOCAL ANALYTICAL LAYER (Python computes, LLM only explains)
# =========================================================================
KEY_METRICS = [
    "temperature_celsius", "temp_anomaly_celsius", "heat_wave_days", "precipitation_mm",
    "air_quality_index", "respiratory_disease_rate", "cardio_mortality_rate",
    "heat_related_admissions", "waterborne_disease_incidents", "vector_disease_risk_score",
    "healthcare_access_index", "gdp_per_capita_usd", "food_security_index",
]


def get_dataset_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"rows": 0}
    return {
        "rows": int(len(df)),
        "countries": int(df.country_name.nunique()),
        "regions": int(df.region.nunique()),
        "year_min": int(df.year.min()),
        "year_max": int(df.year.max()),
    }


def _metric_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for col in KEY_METRICS:
        if col in df.columns and df[col].notna().any():
            s = df[col]
            stats[col] = {
                "mean": round(float(s.mean()), 2),
                "min": round(float(s.min()), 2),
                "max": round(float(s.max()), 2),
                "median": round(float(s.median()), 2),
                "std": round(float(s.std()), 2) if len(s) > 1 else 0.0,
            }
    return stats


def _top_countries_by(df: pd.DataFrame, metric: str, n=5, ascending=False):
    if metric not in df.columns:
        return []
    g = df.groupby("country_name")[metric].mean().sort_values(ascending=ascending)
    return [{"country": c, "value": round(float(v), 2)} for c, v in g.head(n).items()]


def _region_breakdown(df: pd.DataFrame, metric: str):
    if metric not in df.columns:
        return []
    g = df.groupby("region")[metric].mean().sort_values(ascending=False)
    return [{"region": r, "value": round(float(v), 2)} for r, v in g.items()]


def _year_over_year(df: pd.DataFrame):
    if df.empty:
        return {}
    first_yr, last_yr = int(df.year.min()), int(df.year.max())
    out = {"first_year": first_yr, "last_year": last_yr, "changes": {}}
    for col in ["temperature_celsius", "air_quality_index", "respiratory_disease_rate",
                "cardio_mortality_rate", "heat_related_admissions"]:
        if col not in df.columns:
            continue
        v0 = df[df.year == first_yr][col].mean()
        v1 = df[df.year == last_yr][col].mean()
        if pd.notna(v0) and pd.notna(v1):
            out["changes"][col] = {
                "first": round(float(v0), 2),
                "last": round(float(v1), 2),
                "delta": round(float(v1 - v0), 2),
            }
    return out


def _risk_analysis(df: pd.DataFrame):
    """Reuse the dashboard's existing compute_risk() methodology, applied
    to the CURRENT filtered dataframe, rather than a new definition."""
    dash = _dashboard_module()
    try:
        if dash is not None and hasattr(dash, "compute_risk"):
            risk_df = dash.compute_risk(df)
        else:
            risk_df = _fallback_compute_risk(df)
    except Exception:
        risk_df = _fallback_compute_risk(df)

    if risk_df is None or risk_df.empty:
        return {}

    top = risk_df.sort_values("risk_score", ascending=False).head(10)
    return {
        "methodology": (
            "Country-level averages of respiratory disease rate, cardio mortality rate, "
            "AQI, and heat-related admissions are each min-max normalized, then averaged "
            "into a single risk_score. Countries are split into Low/Medium/High Risk using "
            "the 33rd and 66th percentile thresholds of risk_score."
        ),
        "top_risk_countries": [
            {
                "country": r.country_name,
                "region": r.region,
                "risk_score": round(float(r.risk_score), 2),
                "risk_category": r.risk_category,
            }
            for r in top.itertuples()
        ],
    }


def _fallback_compute_risk(d: pd.DataFrame) -> pd.DataFrame:
    """Fallback identical to dashboard.compute_risk, used only if the
    dashboard module cannot be located (should not normally happen)."""
    if d.empty:
        return pd.DataFrame()
    g = d.groupby("country_name").agg(
        region=("region", "first"),
        respiratory_disease_rate=("respiratory_disease_rate", "mean"),
        cardio_mortality_rate=("cardio_mortality_rate", "mean"),
        air_quality_index=("air_quality_index", "mean"),
        heat_related_admissions=("heat_related_admissions", "mean"),
    ).reset_index()

    def norm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0

    g["risk_score"] = (
        norm(g.respiratory_disease_rate) + norm(g.cardio_mortality_rate) +
        norm(g.air_quality_index) + norm(g.heat_related_admissions)
    ) / 4
    q1, q2 = g.risk_score.quantile([0.33, 0.66])
    g["risk_category"] = np.where(g.risk_score >= q2, "High Risk",
                          np.where(g.risk_score >= q1, "Medium Risk", "Low Risk"))
    return g


def _correlation_snapshot(df: pd.DataFrame):
    """Reuse the dashboard's CORR_COLS mapping if available so the
    chatbot references the same metric names used in the Correlation tab."""
    dash = _dashboard_module()
    corr_cols = getattr(dash, "CORR_COLS", None) if dash is not None else None
    if not corr_cols:
        corr_cols = {
            "temperature_celsius": "Temperature", "air_quality_index": "AQI",
            "respiratory_disease_rate": "Respiratory Rate",
            "cardio_mortality_rate": "Cardio Mortality",
            "heat_related_admissions": "Heat Admissions",
        }
    cols = [c for c in corr_cols if c in df.columns]
    if len(cols) < 2 or len(df) < 3:
        return {}
    corr = df[cols].corr().rename(index=corr_cols, columns=corr_cols)
    vals = corr.values.copy()
    np.fill_diagonal(vals, 0)
    c2 = pd.DataFrame(vals, index=corr.index, columns=corr.columns)
    try:
        max_pair = c2.abs().stack().idxmax()
        max_val = round(float(c2.loc[max_pair]), 2)
        strongest = {"pair": [max_pair[0], max_pair[1]], "r": max_val}
    except Exception:
        strongest = {}
    return {
        "strongest_correlation": strongest,
        "matrix_note": "Correlation coefficients (r) range from -1 to 1. Values shown are Pearson correlations on the current filtered data.",
    }


def build_data_context(df: pd.DataFrame) -> str:
    """Compact analytical context sent to the LLM. Does NOT dump raw rows."""
    if df.empty:
        return "The current filtered dataset is EMPTY. No records match the active filters."

    summary = get_dataset_summary(df)
    metric_stats = _metric_stats(df)
    yoy = _year_over_year(df)
    risk = _risk_analysis(df)
    corr = _correlation_snapshot(df)

    top_aqi = _top_countries_by(df, "air_quality_index", n=5)
    top_resp = _top_countries_by(df, "respiratory_disease_rate", n=5)
    top_heat_admit = _top_countries_by(df, "heat_related_admissions", n=5)
    region_resp = _region_breakdown(df, "respiratory_disease_rate")
    region_aqi = _region_breakdown(df, "air_quality_index")
    region_temp = _region_breakdown(df, "temperature_celsius")

    lines = []
    lines.append("=== ACTIVE DASHBOARD FILTERS ===")
    lines.append(get_active_filter_context())
    lines.append("")
    lines.append("=== DATASET SUMMARY (current filtered data) ===")
    lines.append(f"Filtered rows: {summary['rows']}")
    lines.append(f"Countries represented: {summary['countries']}")
    lines.append(f"Regions represented: {summary['regions']}")
    lines.append(f"Year range: {summary['year_min']}\u2013{summary['year_max']}")
    lines.append("")
    lines.append("=== KEY METRIC STATISTICS (mean/min/max/median/std) ===")
    for col, s in metric_stats.items():
        lines.append(f"{col}: mean={s['mean']}, min={s['min']}, max={s['max']}, median={s['median']}, std={s['std']}")
    lines.append("")
    lines.append("=== TOP COUNTRIES ===")
    lines.append(f"Top 5 by AQI: {top_aqi}")
    lines.append(f"Top 5 by respiratory disease rate: {top_resp}")
    lines.append(f"Top 5 by heat-related admissions: {top_heat_admit}")
    lines.append("")
    lines.append("=== REGION BREAKDOWNS (avg, high to low) ===")
    lines.append(f"Respiratory disease rate by region: {region_resp}")
    lines.append(f"AQI by region: {region_aqi}")
    lines.append(f"Temperature by region: {region_temp}")
    lines.append("")
    lines.append("=== YEAR-OVER-YEAR CHANGE (first vs last year in filtered range) ===")
    lines.append(str(yoy))
    lines.append("")
    lines.append("=== RISK ANALYSIS (dashboard's existing methodology) ===")
    lines.append(str(risk))
    lines.append("")
    lines.append("=== CORRELATION SNAPSHOT ===")
    lines.append(str(corr))

    return "\n".join(lines)


# =========================================================================
# GROQ CALL
# =========================================================================
def ask_groq(question: str, df: pd.DataFrame, history: list) -> str:
    client = get_groq_client()
    if client is None:
        return ("\u26a0\ufe0f Groq API key is not configured. Add GROQ_API_KEY to your "
                ".env file to enable the AI assistant.")

    context = build_data_context(df)
    trimmed_history = history[-MAX_HISTORY_MESSAGES:]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({
        "role": "system",
        "content": "DATA CONTEXT FOR THIS TURN (computed locally with pandas, treat as ground truth):\n" + context,
    })
    for m in trimmed_history:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": question})

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=700,
            temperature=0.3,
            messages=messages,
        )
        answer = resp.choices[0].message.content
        if not answer or not answer.strip():
            return "\u26a0\ufe0f The AI assistant returned an empty response. Please try rephrasing your question."
        return answer
    except Exception:
        return "\u26a0\ufe0f The AI assistant is temporarily unavailable. Please try again."


# =========================================================================
# CHAT UI (rendered at the bottom of the Overview tab)
# =========================================================================
def render_climate_health_chatbot(df: pd.DataFrame):
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    st.markdown(
        "<div style='font-size:12px;color:#B9C4D0;margin-bottom:8px;'>"
        "Ask questions about the climate-health data \u2014 answers use only the current filtered dataset."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div style='font-size:11.5px;color:#B9C4D0;margin-bottom:8px;'>"
        f"<b>Active filters:</b> {_filters_summary_line()}</div>",
        unsafe_allow_html=True,
    )

    if not GROQ_API_KEY:
        st.warning(
            "\u26a0\ufe0f Groq API key is not configured. Add GROQ_API_KEY to your .env "
            "file to enable the AI assistant."
        )

    # Suggested question buttons
    btn_cols = st.columns(len(SUGGESTED_QUESTIONS))
    clicked_question = None
    for c, q in zip(btn_cols, SUGGESTED_QUESTIONS):
        if c.button(q, key=f"suggest_{q}", use_container_width=True):
            clicked_question = q

    # Chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    typed_question = st.chat_input("Ask about climate, health, risk, trends...")
    question = clicked_question or typed_question

    if question:
        st.session_state.chat_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing filtered data..."):
                answer = ask_groq(question, df, st.session_state.chat_messages[:-1])
            st.markdown(answer)
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        if len(st.session_state.chat_messages) > MAX_HISTORY_MESSAGES:
            st.session_state.chat_messages = st.session_state.chat_messages[-MAX_HISTORY_MESSAGES:]

    if st.session_state.chat_messages:
        if st.button("Clear chat", key="clear_chat_btn"):
            st.session_state.chat_messages = []
            st.rerun()
