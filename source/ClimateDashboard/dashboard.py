import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================================================================
# PAGE CONFIG
# =========================================================================
st.set_page_config(
    page_title="Climate Health Impact Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================================
# COLOR THEME
# =========================================================================
BG = "#07121F"
CARD = "#0E1C2B"
PANEL = "#102033"
BORDER = "rgba(255,255,255,0.08)"
BLUE = "#1F8FFF"
GREEN = "#3DDC84"
ORANGE = "#F4A825"
RED = "#FF4D57"
PURPLE = "#B15EFF"
TEXT = "#FFFFFF"
SUBTEXT = "#B9C4D0"
GRID = "rgba(255,255,255,0.05)"

REGION_SEQ = [BLUE, ORANGE, GREEN, RED, PURPLE, "#38BDF8", "#F97316", "#84CC16"]
FONT_FAMILY = "'Inter','Segoe UI','Roboto',sans-serif"

# =========================================================================
# GLOBAL CSS
# =========================================================================
st.markdown(f"""
<style>
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}
div[data-testid="stToolbar"] {{visibility: hidden;}}
div[data-testid="stDecoration"] {{visibility: hidden;}}

html, body, [class*="css"] {{
    font-family: {FONT_FAMILY};
}}

.stApp {{
    background-color: {BG};
    color: {TEXT};
}}

section[data-testid="stSidebar"] {{
    background-color: #060D16;
    border-right: 1px solid {BORDER};
    width: 240px !important;
    min-width: 240px !important;
}}
section[data-testid="stSidebar"] > div {{
    padding-top: 6px;
}}
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p {{
    color: {SUBTEXT} !important;
    font-size: 12.5px !important;
}}

section[data-testid="stSidebar"] div.stButton > button {{
    width: 100% !important;
    min-height: 38px !important;
    background-color: rgba(31, 143, 255, 0.16) !important;
    color: {BLUE} !important;
    border: 1px solid rgba(31, 143, 255, 0.35) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}}
section[data-testid="stSidebar"] div.stButton > button:hover {{
    background-color: rgba(31, 143, 255, 0.24) !important;
}}

.block-container {{
    padding-top: 0.6rem;
    padding-bottom: 0.4rem;
    padding-left: 1rem;
    padding-right: 1rem;
    max-width: 100% !important;
}}

::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.15); border-radius: 10px; }}
::-webkit-scrollbar-track {{ background: transparent; }}

/* Panel container */
.panel {{
    background-color: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 10px 12px;
    margin-bottom: 8px;
}}
.panel-title {{
    font-size: 15px;
    font-weight: 700;
    color: {BLUE};
    letter-spacing: 0.4px;
    margin-bottom: 6px;
    text-transform: uppercase;
}}
.panel-title.green {{ color: {GREEN}; }}
.panel-title.orange {{ color: {ORANGE}; }}
.panel-title.red {{ color: {RED}; }}
.panel-title.purple {{ color: {PURPLE}; }}
.chart-title {{
    font-size: 14px;
    font-weight: 700;
    color: {TEXT};
    margin-bottom: 3px;
}}

/* KPI cards */
.kpi-card {{
    background-color: {CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 10px 14px;
    display: flex;
    align-items: center;
    gap: 12px;
    height: 90px;
}}
.kpi-icon {{
    font-size: 26px;
    width: 44px;
    text-align: center;
}}
.kpi-title {{
    font-size: 12px;
    color: {SUBTEXT};
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.3px;
}}
.kpi-value {{
    font-size: 26px;
    font-weight: 800;
    color: {TEXT};
    line-height: 1.1;
}}
.kpi-sub {{
    font-size: 11px;
    color: {SUBTEXT};
}}

/* Header */
.dash-title {{
    font-size: 26px;
    font-weight: 800;
    color: {TEXT};
    margin-bottom: 0px;
}}
.dash-subtitle {{
    font-size: 12.5px;
    color: {SUBTEXT};
    margin-top: -4px;
}}

/* About box */
.about-box {{
    background-color: {CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 10px;
    font-size: 12px;
    color: {SUBTEXT};
    line-height: 1.5;
}}
.about-title {{
    color: {BLUE};
    font-weight: 700;
    font-size: 13px;
    margin-bottom: 4px;
}}

/* insight rows */
.insight-row {{
    display:flex; gap:8px; align-items:flex-start;
    font-size: 12.5px; color: {SUBTEXT};
    padding: 4px 0px;
    border-bottom: 1px dashed {BORDER};
}}

/* Tabs */
button[data-baseweb="tab"] {{
    color: {SUBTEXT} !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    padding: 14px 20px !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {BLUE} !important;
    border-bottom-color: {BLUE} !important;
}}
div[data-baseweb="tab-highlight"] {{ background-color: {BLUE} !important; }}
div[data-baseweb="tab-border"] {{ background-color: {BORDER} !important; }}

hr {{ border-color: {BORDER}; margin: 6px 0px; }}

div[data-testid="stMetric"] {{
    background-color: {CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 8px;
}}
</style>
""", unsafe_allow_html=True)

# =========================================================================
# DATA LOADING
# =========================================================================
@st.cache_data
def load_data():
    d = pd.read_csv("climate_health_cleaned.csv")
    d["date"] = pd.to_datetime(d["date"])
    return d


@st.cache_data
def compute_outliers(d: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = d.select_dtypes(include=["int64", "float64"]).columns
    rows = []
    for col in numeric_cols:
        if d[col].nunique() <= 2:
            continue
        Q1, Q3 = d[col].quantile(0.25), d[col].quantile(0.75)
        IQR = Q3 - Q1
        lb, ub = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = d[(d[col] < lb) | (d[col] > ub)]
        rows.append({
            "Column": col,
            "Total Values": len(d),
            "Outliers": len(outliers),
            "Outlier %": round(len(outliers) / len(d) * 100, 2),
            "Lower Bound": round(lb, 2),
            "Upper Bound": round(ub, 2),
            "Future Insight": "Potential climate extreme / health hotspot" if len(outliers) > 0 else "No significant outliers",
        })
    return pd.DataFrame(rows)


@st.cache_data
def compute_risk(d: pd.DataFrame) -> pd.DataFrame:
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


df_raw = load_data()
outlier_df = compute_outliers(df_raw)
risk_df_global = compute_risk(df_raw)

# =========================================================================
# SIDEBAR FILTERS
# =========================================================================
DEFAULTS = {
    "f_country": "All", "f_region": "All", "f_income": "All",
    "f_year": (int(df_raw.year.min()), int(df_raw.year.max())),
    "f_month": "All",
    "f_temp": (float(np.floor(df_raw.temperature_celsius.min())), float(np.ceil(df_raw.temperature_celsius.max()))),
    "f_aqi": (0.0, float(np.ceil(df_raw.air_quality_index.max()))),
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_filters():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


def cascade_options(field, exclude):
    """Options for `field`, narrowed by every OTHER active category filter.
    Keeps Country / Region / Income Level mutually consistent so no
    combination can ever silently produce zero rows."""
    d = df_raw
    if exclude != "f_country" and st.session_state.f_country != "All":
        d = d[d.country_name == st.session_state.f_country]
    if exclude != "f_region" and st.session_state.f_region != "All":
        d = d[d.region == st.session_state.f_region]
    if exclude != "f_income" and st.session_state.f_income != "All":
        d = d[d.income_level == st.session_state.f_income]
    return ["All"] + sorted(d[field].unique().tolist())


def sync_categorical_filters():
    """If a previously-picked value is no longer reachable given the other
    active filters (e.g. Country was set, then a conflicting Region was
    picked), fall that value back to 'All' instead of returning an empty
    dataframe."""
    country_opts = cascade_options("country_name", "f_country")
    if st.session_state.f_country not in country_opts:
        st.session_state.f_country = "All"
    region_opts = cascade_options("region", "f_region")
    if st.session_state.f_region not in region_opts:
        st.session_state.f_region = "All"
    income_opts = cascade_options("income_level", "f_income")
    if st.session_state.f_income not in income_opts:
        st.session_state.f_income = "All"


sync_categorical_filters()

with st.sidebar:
    st.button("↺ Reset Filters", on_click=reset_filters, use_container_width=True)
    st.markdown("<hr style='margin: 6px 0px;'/>", unsafe_allow_html=True)

    st.selectbox("Country", cascade_options("country_name", "f_country"), key="f_country")
    st.selectbox("Region", cascade_options("region", "f_region"), key="f_region")
    st.selectbox("Income Level", cascade_options("income_level", "f_income"), key="f_income")
    st.markdown("**Year Range**")
    st.slider("Year Range", int(df_raw.year.min()), int(df_raw.year.max()), key="f_year", label_visibility="collapsed")
    st.selectbox("Month", ["All"] + list(range(1, 13)), key="f_month")
    st.markdown("**Temperature (°C)**")
    st.slider("Temperature (°C)", float(np.floor(df_raw.temperature_celsius.min())),
               float(np.ceil(df_raw.temperature_celsius.max())), key="f_temp", label_visibility="collapsed")
    st.markdown("**Air Quality Index (AQI)**")
    st.slider("AQI", 0.0, float(np.ceil(df_raw.air_quality_index.max())), key="f_aqi", label_visibility="collapsed")

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="about-box">
        <div class="about-title">ℹ️ About This Dashboard</div>
        This dashboard explores the impact of climate change on public health
        outcomes using global data from 2015 to 2025. It helps policymakers and
        health organizations identify vulnerable regions, understand risk
        drivers, and make data-driven decisions for a healthier, more resilient
        future.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-top:10px; height:110px; border-radius:10px; border:1px solid rgba(255,255,255,0.08);
    background: linear-gradient(180deg, #16324a 0%, #0c2233 60%, #0a1a26 100%);
    display:flex; align-items:center; justify-content:center; font-size:34px;">
    🌬️🌍
    </div>
    """, unsafe_allow_html=True)

# =========================================================================
# APPLY FILTERS
# =========================================================================
def apply_filters(d):
    out = d
    if st.session_state.f_country != "All":
        out = out[out.country_name == st.session_state.f_country]
    if st.session_state.f_region != "All":
        out = out[out.region == st.session_state.f_region]
    if st.session_state.f_income != "All":
        out = out[out.income_level == st.session_state.f_income]
    y0, y1 = st.session_state.f_year
    out = out[(out.year >= y0) & (out.year <= y1)]
    if st.session_state.f_month != "All":
        out = out[out.month == st.session_state.f_month]
    t0, t1 = st.session_state.f_temp
    out = out[(out.temperature_celsius >= t0) & (out.temperature_celsius <= t1)]
    a0, a1 = st.session_state.f_aqi
    out = out[(out.air_quality_index >= a0) & (out.air_quality_index <= a1)]
    return out


df = apply_filters(df_raw)

with st.sidebar:
    st.caption(f"Showing **{len(df):,}** of **{len(df_raw):,}** records "
               f"across **{df.country_name.nunique() if not df.empty else 0}** countries")

if df.empty:
    st.warning(
        "No data matches the current Temperature / AQI / Year / Month range. "
        "Country, Region and Income Level stay in sync automatically — "
        "try widening the Temperature or AQI slider, or use Reset in the sidebar."
    )
    st.stop()

# =========================================================================
# CHART STYLING HELPER
# =========================================================================
def style_fig(fig, height=170, show_legend=False, margin=None):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, size=11, color=SUBTEXT),
        margin=margin or dict(l=30, r=10, t=18, b=24),
        showlegend=show_legend,
        legend=dict(font=dict(size=9, color=SUBTEXT), orientation="h", y=1.18, x=0),
        hoverlabel=dict(bgcolor=CARD, font_size=11, font_family=FONT_FAMILY),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=BORDER, tickfont=dict(size=9))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, tickfont=dict(size=9))
    return fig


def chart_card(title, fig, container=st):
    container.markdown(f"<div class='chart-title'>{title}</div>", unsafe_allow_html=True)
    container.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def panel_open(title, color="blue"):
    st.markdown(f"<div class='panel'><div class='panel-title {color}'>{title}</div>", unsafe_allow_html=True)


def panel_close():
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================================
# KPI ROW
# =========================================================================
def kpi_row(d):
    n_countries = d.country_name.nunique()
    pop_covered = d.groupby("country_name")["population_millions"].first().sum() / 1000
    avg_temp = d.temperature_celsius.mean()
    first_yr, last_yr = d.year.min(), d.year.max()
    temp_first = d[d.year == first_yr].temperature_celsius.mean()
    temp_last = d[d.year == last_yr].temperature_celsius.mean()
    temp_delta = temp_last - temp_first if pd.notna(temp_first) and pd.notna(temp_last) else 0
    avg_aqi = d.air_quality_index.mean()
    aqi_label = "Good" if avg_aqi <= 50 else "Moderate" if avg_aqi <= 100 else "Unhealthy" if avg_aqi <= 150 else "Hazardous"
    avg_resp = d.respiratory_disease_rate.mean()
    avg_hai = d.healthcare_access_index.mean() / 100
    hai_label = "Good" if avg_hai >= 0.6 else "Fair" if avg_hai >= 0.4 else "Poor"

    cards = [
        ("🌐", "Total Countries", f"{n_countries}", "Countries", BLUE),
        ("👥", "Population Covered", f"{pop_covered:.2f}B", "Millions", GREEN),
        ("🌡️", "Avg Temperature", f"{avg_temp:.2f}°C", f"{'▲' if temp_delta>=0 else '▼'} {abs(temp_delta):.2f} vs first yr", ORANGE),
        ("💨", "Avg AQI", f"{avg_aqi:.2f}", aqi_label, PURPLE),
        ("🫁", "Avg Respiratory Disease Rate", f"{avg_resp:.2f}", "per 100K", RED),
        ("➕", "Avg Healthcare Access Index", f"{avg_hai:.2f}", hai_label, GREEN),
    ]
    cols = st.columns(6)
    for c, (icon, title, val, sub, color) in zip(cols, cards):
        c.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div>
                <div class="kpi-title">{title}</div>
                <div class="kpi-value" style="color:{color}">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# =========================================================================
# CHART BUILDERS
# =========================================================================
def fig_line(d, x, y, color, height=150):
    g = d.groupby(x, as_index=False)[y].mean()
    fig = px.line(g, x=x, y=y, markers=True)
    fig.update_traces(line=dict(color=color, width=2.5), marker=dict(size=5, color=color))
    return style_fig(fig, height=height)


def fig_bar_region(d, y, color, height=150, top_n=None, ascending=False):
    g = d.groupby("region", as_index=False)[y].sum().sort_values(y, ascending=ascending)
    if top_n:
        g = g.head(top_n)
    fig = px.bar(g, x="region", y=y, text=y)
    fig.update_traces(
        marker_color=color,
        texttemplate="%{text:.0f}",
        textposition="outside",
        textfont=dict(size=11, color=TEXT),
        cliponaxis=False,
    )
    fig.update_layout(
        margin=dict(l=40, r=20, t=12, b=90),
        xaxis_tickangle=-20,
        yaxis=dict(range=[0, max(g[y]) * 1.18]),
    )
    return style_fig(fig, height=height)


def fig_climate_corr_heatmap(d, height=150):
    cols = {
        "temperature_celsius": "Temp", "heat_wave_days": "Heatwave",
        "flood_indicator": "Flood Ind.", "drought_indicator": "Drought Ind.",
        "precipitation_mm": "Precip.", "air_quality_index": "AQI",
    }
    corr = d[list(cols.keys())].corr().rename(index=cols, columns=cols)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=[[0, BLUE], [0.5, PANEL], [1, RED]], zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        textfont=dict(size=8, color=TEXT),
        colorbar=dict(thickness=8, tickfont=dict(size=8, color=SUBTEXT)),
    ))
    return style_fig(fig, height=height, margin=dict(l=45, r=10, t=10, b=30))


def fig_top10_bar(d, col, color, height=150, n=10):
    g = d.groupby("country_name", as_index=False)[col].mean().sort_values(col, ascending=False).head(n)
    g = g.sort_values(col)
    fig = px.bar(g, x=col, y="country_name", orientation="h")
    fig.update_traces(
        marker_color=color,
        text=g[col].round(2).astype(str),
        texttemplate="%{text}",
        textposition="outside",
        textfont=dict(size=11, color=SUBTEXT),
    )
    fig.update_yaxes(tickfont=dict(size=11, color=SUBTEXT))
    fig.update_xaxes(tickfont=dict(size=11, color=SUBTEXT))
    return style_fig(fig, height=height, margin=dict(l=80, r=25, t=10, b=24))


def fig_risk_health_matrix(d, height=250):
    g = d.groupby("country_name", as_index=False).agg(
        avg_aqi=("air_quality_index", "mean"),
        avg_temp_anomaly=("temp_anomaly_celsius", "mean"),
        respiratory_disease_rate=("respiratory_disease_rate", "mean"),
        cardio_mortality_rate=("cardio_mortality_rate", "mean"),
    )
    def norm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0

    g["Climate Risk Score"] = norm(g.avg_aqi + g.avg_temp_anomaly)
    g["Health Burden Score"] = norm((g.respiratory_disease_rate + g.cardio_mortality_rate) / 2)

    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, x1=0.5, y0=0, y1=0.5, fillcolor="rgba(59,130,246,0.14)", line_width=0)
    fig.add_shape(type="rect", x0=0.5, x1=1, y0=0, y1=0.5, fillcolor="rgba(249,115,22,0.14)", line_width=0)
    fig.add_shape(type="rect", x0=0, x1=0.5, y0=0.5, y1=1, fillcolor="rgba(16,185,129,0.14)", line_width=0)
    fig.add_shape(type="rect", x0=0.5, x1=1, y0=0.5, y1=1, fillcolor="rgba(255,77,87,0.14)", line_width=0)

    fig.add_trace(go.Scatter(
        x=g["Climate Risk Score"], y=g["Health Burden Score"], mode="markers+text",
        text=g["country_name"], textposition="top center",
        marker=dict(size=9, color=RED, opacity=0.8), hovertemplate="%{text}<br>Climate Risk: %{x:.2f}<br>Health Burden: %{y:.2f}<extra></extra>"
    ))

    fig.update_layout(
        xaxis=dict(title="Climate Risk Score", range=[0, 1], showgrid=False, zeroline=False, linecolor=BORDER),
        yaxis=dict(title="Health Burden Score", range=[0, 1], showgrid=True, gridcolor=GRID, zeroline=False, linecolor=BORDER),
        margin=dict(l=50, r=10, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, size=11, color=SUBTEXT), showlegend=False,
    )
    return fig


def fig_top10_composite_risk(d, height=250, n=10):
    g = d.groupby("country_name", as_index=False).agg(
        avg_aqi=("air_quality_index", "mean"),
        avg_temp_anomaly=("temp_anomaly_celsius", "mean"),
        respiratory_disease_rate=("respiratory_disease_rate", "mean"),
        cardio_mortality_rate=("cardio_mortality_rate", "mean"),
        heat_related_admissions=("heat_related_admissions", "mean"),
    )
    def norm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0

    g["Climate Risk Score"] = (norm(g.avg_aqi) + norm(g.avg_temp_anomaly) + norm(g.heat_related_admissions)) / 3
    g["Health Burden Score"] = (norm(g.respiratory_disease_rate) + norm(g.cardio_mortality_rate)) / 2
    g["Composite Score"] = (g["Climate Risk Score"] + g["Health Burden Score"]) / 2
    g = g.sort_values("Composite Score", ascending=False).head(n).sort_values("Composite Score")

    fig = px.bar(g, x="Composite Score", y="country_name", orientation="h", text="Composite Score",
                 color="Climate Risk Score", color_continuous_scale=[[0, GREEN], [0.5, ORANGE], [1, RED]])
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", marker_line_color=BG, marker_line_width=1)
    fig.update_layout(coloraxis_showscale=False)
    return style_fig(fig, height=height, margin=dict(l=90, r=25, t=10, b=24), show_legend=False)


def fig_disease_burden_region(d, height=320):
    g = d.groupby("region", as_index=False).agg(
        Respiratory_Diseases=("respiratory_disease_rate", "mean"),
        Cardiovascular_Mortality=("cardio_mortality_rate", "mean"),
        Heat_related_Illness=("heat_related_admissions", "mean"),
        Waterborne_Diseases=("waterborne_disease_incidents", "mean"),
        Vector_Disease_Risk=("vector_disease_risk_score", "mean"),
    )
    
    # Calculate percentages
    disease_cols = ["Respiratory_Diseases", "Cardiovascular_Mortality", "Heat_related_Illness", "Waterborne_Diseases", "Vector_Disease_Risk"]
    g["Total"] = g[disease_cols].sum(axis=1)
    for col in disease_cols:
        g[col] = (g[col] / g["Total"] * 100).round(1)
    g = g.drop("Total", axis=1)
    
    # Rename for display
    g = g.rename(columns={
        "Respiratory_Diseases": "Respiratory Diseases",
        "Cardiovascular_Mortality": "Cardiovascular Mortality",
        "Heat_related_Illness": "Heat-related Illness",
        "Waterborne_Diseases": "Waterborne Diseases",
        "Vector_Disease_Risk": "Vector Diseases",
    })
    
    fig = px.bar(
        g,
        x=["Respiratory Diseases", "Cardiovascular Mortality", "Heat-related Illness", "Waterborne Diseases", "Vector Diseases"],
        y="region",
        orientation="h",
        color_discrete_sequence=[PURPLE, RED, ORANGE, BLUE, GREEN],
        labels={"value": "% Contribution", "region": "Region", "variable": "Disease Category"},
    )
    
    fig.update_layout(
        barmode="stack",
        legend_title_text="Disease Categories",
        xaxis_title="% Contribution of Total Disease Burden",
        yaxis_title="Region",
        margin=dict(l=150, r=10, t=25, b=40),
        height=height,
    )
    fig.update_traces(
        texttemplate="%{x:.1f}%",
        textposition="inside",
        textfont=dict(size=10, color=TEXT),
    )
    fig.update_xaxes(range=[0, 100])
    
    return style_fig(fig, height=height, show_legend=True, margin=dict(l=150, r=10, t=25, b=40))


def fig_public_health_trend(d, height=250):
    grouped = d.groupby("year", as_index=False).agg(
        Respiratory_Rate=("respiratory_disease_rate", "mean"),
        Cardio_Mortality=("cardio_mortality_rate", "mean"),
        Heat_Admissions=("heat_related_admissions", "mean"),
        Waterborne_Incidents=("waterborne_disease_incidents", "mean"),
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grouped["year"], y=grouped["Respiratory_Rate"], mode="lines+markers", name="Respiratory Rate", line=dict(color=PURPLE, width=2.5), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=grouped["year"], y=grouped["Cardio_Mortality"], mode="lines+markers", name="Cardio Mortality", line=dict(color=RED, width=2.5), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=grouped["year"], y=grouped["Heat_Admissions"], mode="lines+markers", name="Heat Admissions", line=dict(color=ORANGE, width=2.5), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=grouped["year"], y=grouped["Waterborne_Incidents"], mode="lines+markers", name="Waterborne Incidents", line=dict(color=GREEN, width=2.5), marker=dict(size=6)))
    fig.update_layout(xaxis_title="Year", yaxis_title="Average Rate / Admissions",
                      legend=dict(font=dict(size=9, color=SUBTEXT), orientation="h", y=1.12, x=0),
                      margin=dict(l=40, r=10, t=20, b=40))
    return style_fig(fig, height=height, show_legend=True)


def fig_resp_rate_vs_aqi(d, height=280):
    g = d.groupby("year", as_index=False).agg(
        avg_aqi=("air_quality_index", "mean"),
        avg_resp=("respiratory_disease_rate", "mean"),
    ).sort_values("year")
    corr = d["air_quality_index"].corr(d["respiratory_disease_rate"])
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=g["year"], y=g["avg_aqi"], name="Average AQI",
        marker_color=ORANGE, text=g["avg_aqi"].round(0), textposition="outside",
        hovertemplate="Year %{x}<br>AQI %{y:.1f}<extra></extra>"
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=g["year"], y=g["avg_resp"], name="Respiratory Disease Rate",
        mode="lines+markers", line=dict(color=PURPLE, width=2.5), marker=dict(size=7, color=PURPLE),
        hovertemplate="Year %{x}<br>Respiratory Rate %{y:.1f}<extra></extra>"
    ), secondary_y=True)
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Average AQI",
        yaxis2_title="Respiratory Disease Rate",
        legend=dict(orientation="h", y=1.12, x=0.02, font=dict(size=10, color=SUBTEXT)),
        margin=dict(l=50, r=50, t=30, b=40),
        bargap=0.25,
    )
    fig.update_xaxes(showgrid=False, tickmode="linear")
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, secondary_y=False)
    fig.update_yaxes(showgrid=False, zeroline=False, secondary_y=True)
    return style_fig(fig, height=height, show_legend=True)


def fig_heatwave_admissions_combo(d, height=320):
    g = d.groupby("region", as_index=False).agg(
        heatwave_days=("heat_wave_days", "mean"),
        heat_admissions=("heat_related_admissions", "mean"),
    ).sort_values("heatwave_days", ascending=False)
    colors = [RED if v >= g.heatwave_days.mean() else ORANGE for v in g.heatwave_days]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=g["region"], y=g["heatwave_days"], name="Heatwave Days",
        marker_color=colors, text=g["heatwave_days"].round(1), textposition="outside",
        hovertemplate="%{x}<br>Heatwave Days %{y:.1f}<extra></extra>"
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=g["region"], y=g["heat_admissions"], name="Heat-related Admissions",
        mode="lines+markers", line=dict(color=RED, width=2.5), marker=dict(size=7, color=RED),
        hovertemplate="%{x}<br>Admissions %{y:.1f}<extra></extra>"
    ), secondary_y=True)
    fig.update_layout(
        xaxis_title="Region",
        yaxis_title="Heatwave Days",
        yaxis2_title="Heat-related Admissions",
        legend=dict(orientation="h", y=1.12, x=0.02, font=dict(size=10, color=SUBTEXT)),
        margin=dict(l=50, r=50, t=30, b=50),
        bargap=0.2,
    )
    fig.update_xaxes(tickangle=-15)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, secondary_y=False)
    fig.update_yaxes(showgrid=False, zeroline=False, secondary_y=True)
    return style_fig(fig, height=height, show_legend=True)


def fig_precipitation_waterborne(d, height=320):
    g = d.groupby("year", as_index=False).agg(
        avg_precipitation=("precipitation_mm", "mean"),
        avg_waterborne=("waterborne_disease_incidents", "mean"),
    ).sort_values("year")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=g["avg_precipitation"], y=g["avg_waterborne"],
        mode="markers+text",
        text=g["year"].astype(str),
        textposition="top center",
        name="Years",
        marker=dict(size=10, color=ORANGE, line=dict(color=TEXT, width=1)),
        hovertemplate="Year %{text}<br>Precipitation %{x:.1f} mm<br>Waterborne Incidents %{y:.1f}<extra></extra>"
    ))

    trend = np.poly1d(np.polyfit(g["avg_precipitation"], g["avg_waterborne"], 1))
    x_range = np.linspace(g["avg_precipitation"].min(), g["avg_precipitation"].max(), 50)
    fig.add_trace(go.Scatter(
        x=x_range, y=trend(x_range),
        mode="lines",
        name="Trend Line",
        line=dict(color=RED, width=2, dash="dash"),
        hoverinfo="skip"
    ))

    fig.update_layout(
        xaxis_title="Average Precipitation (mm)",
        yaxis_title="Waterborne Disease Incidents",
        legend=dict(orientation="h", y=1.12, x=0.02, font=dict(size=10, color=SUBTEXT)),
        margin=dict(l=50, r=50, t=30, b=40),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    return style_fig(fig, height=height, show_legend=True)


def fig_temperature_resp_trend(d, height=320):
    g = d.groupby("year", as_index=False).agg(
        avg_temp=("temperature_celsius", "mean"),
        avg_resp=("respiratory_disease_rate", "mean"),
    ).sort_values("year")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=g["year"], y=g["avg_temp"], name="Average Temperature",
        mode="lines+markers", line=dict(color=ORANGE, width=2.5), marker=dict(size=7, color=ORANGE),
        hovertemplate="Year %{x}<br>Temperature %{y:.1f}°C<extra></extra>"
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=g["year"], y=g["avg_resp"], name="Respiratory Disease Rate",
        mode="lines+markers", line=dict(color=PURPLE, width=2.5), marker=dict(size=7, color=PURPLE),
        hovertemplate="Year %{x}<br>Respiratory Rate %{y:.1f}<extra></extra>"
    ), secondary_y=True)
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Average Temperature (°C)",
        yaxis2_title="Respiratory Disease Rate",
        legend=dict(orientation="h", y=1.12, x=0.02, font=dict(size=10, color=SUBTEXT)),
        margin=dict(l=50, r=50, t=30, b=40),
    )
    fig.update_xaxes(showgrid=False, tickmode="linear")
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, secondary_y=False)
    fig.update_yaxes(showgrid=False, zeroline=False, secondary_y=True)
    return style_fig(fig, height=height, show_legend=True)


def fig_climate_indicator_heatmap(d, height=260):
    g = d.groupby("region", as_index=False)[["temperature_celsius", "temp_anomaly_celsius", "heat_wave_days", "precipitation_mm", "air_quality_index"]].mean()
    labels = {
        "temperature_celsius": "Temperature",
        "temp_anomaly_celsius": "Temp Anomaly",
        "heat_wave_days": "Heatwave Days",
        "precipitation_mm": "Precipitation",
        "air_quality_index": "AQI",
    }
    data = g.rename(columns=labels).set_index("region")
    fig = go.Figure(go.Heatmap(
        z=data.values,
        x=data.columns,
        y=data.index,
        colorscale=[[0, BLUE], [0.5, PANEL], [1, RED]],
        text=np.round(data.values, 2),
        texttemplate="%{text:.2f}",
        textfont=dict(size=9, color=TEXT),
        colorbar=dict(thickness=8, tickfont=dict(size=8, color=SUBTEXT)),
    ))
    fig.update_layout(xaxis_title="Indicator", yaxis_title="Region", margin=dict(l=90, r=10, t=20, b=50))
    return style_fig(fig, height=height, show_legend=False)


def fig_feature_importance_chart(d, height=250):
    climate_vars = ["temperature_celsius", "temp_anomaly_celsius", "heat_wave_days", "precipitation_mm", "air_quality_index"]
    health_outcomes = ["respiratory_disease_rate", "cardio_mortality_rate", "waterborne_disease_incidents", "heat_related_admissions"]
    corr = d[climate_vars + health_outcomes].corr()
    scores = []
    for var in climate_vars:
        abs_corr = corr.loc[var, health_outcomes].abs().mean()
        scores.append((var, abs_corr))
    score_df = pd.DataFrame(scores, columns=["variable", "importance"]).sort_values("importance", ascending=True)
    score_df["label"] = score_df.variable.map({
        "temperature_celsius": "Temperature",
        "temp_anomaly_celsius": "Temp Anomaly",
        "heat_wave_days": "Heatwave Days",
        "precipitation_mm": "Precipitation",
        "air_quality_index": "AQI",
    })
    fig = px.bar(score_df, x="importance", y="label", orientation="h", color="importance",
                 color_continuous_scale=[[0, BLUE], [0.5, ORANGE], [1, RED]])
    fig.update_layout(showlegend=False, margin=dict(l=90, r=10, t=20, b=40))
    fig.update_traces(texttemplate="%{x:.2f}", textposition="outside", textfont=dict(color=SUBTEXT, size=9))
    return style_fig(fig, height=height, show_legend=False)


def fig_risk_factor_ranking(d, height=250):
    risk_df = compute_risk(d)
    enriched = d.groupby("country_name", as_index=False).agg(
        temperature_celsius=("temperature_celsius", "mean"),
        temp_anomaly_celsius=("temp_anomaly_celsius", "mean"),
        heat_wave_days=("heat_wave_days", "mean"),
        air_quality_index=("air_quality_index", "mean"),
        gdp_per_capita_usd=("gdp_per_capita_usd", "mean"),
        food_security_index=("food_security_index", "mean"),
    )
    merged = enriched.merge(risk_df[["country_name", "risk_score"]], on="country_name")
    corr_values = merged.corr(numeric_only=True)["risk_score"].drop("risk_score").abs().sort_values(ascending=True)
    score_df = corr_values.reset_index().rename(columns={"index": "variable", "risk_score": "importance"})
    labels = {
        "temperature_celsius": "Temperature",
        "temp_anomaly_celsius": "Temp Anomaly",
        "heat_wave_days": "Heatwave Days",
        "air_quality_index": "AQI",
        "gdp_per_capita_usd": "GDP per Capita",
        "food_security_index": "Food Security",
    }
    score_df["label"] = score_df.variable.map(labels)
    fig = px.bar(score_df, x="importance", y="label", orientation="h", color="importance",
                 color_continuous_scale=[[0, BLUE], [0.5, ORANGE], [1, RED]])
    fig.update_layout(showlegend=False, margin=dict(l=90, r=10, t=20, b=40))
    fig.update_traces(texttemplate="%{x:.2f}", textposition="outside", textfont=dict(color=SUBTEXT, size=9))
    return style_fig(fig, height=height, show_legend=False)


def fig_country_risk_map(d, height=320):
    risk_df = compute_risk(d)
    g = d.groupby("country_name", as_index=False).agg(
        latitude=("latitude", "mean"), longitude=("longitude", "mean"),
        population_millions=("population_millions", "first"),
    ).merge(risk_df[["country_name", "risk_score"]], on="country_name")
    fig = px.scatter_geo(g, lat="latitude", lon="longitude", size="population_millions",
                         color="risk_score", hover_name="country_name",
                         color_continuous_scale=[[0, GREEN], [0.5, ORANGE], [1, RED]], size_max=30,
                         projection="natural earth")
    fig.update_geos(bgcolor="rgba(0,0,0,0)", showland=True, landcolor="#13253A",
                    showocean=True, oceancolor=BG, showcountries=True, countrycolor=BORDER,
                    showcoastlines=False, showframe=False)
    fig.update_layout(coloraxis_colorbar=dict(title="Risk", thickness=8, tickfont=dict(size=8, color=SUBTEXT)))
    return style_fig(fig, height=height, margin=dict(l=0, r=0, t=0, b=0))


def fig_region_risk_ranking(d, height=250):
    risk_df = compute_risk(d)
    g = risk_df.groupby("region", as_index=False).risk_score.mean().sort_values("risk_score", ascending=True)
    fig = px.bar(g, x="risk_score", y="region", orientation="h", text="risk_score",
                 color="risk_score", color_continuous_scale=[[0, GREEN], [0.5, ORANGE], [1, RED]])
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    return style_fig(fig, height=height, margin=dict(l=90, r=10, t=20, b=40), show_legend=False)


def fig_population_at_risk_by_region(d, height=250):
    risk_df = compute_risk(d)
    g = d.merge(risk_df[["country_name", "risk_category"]], on="country_name")
    g = g[g.risk_category == "High Risk"].groupby("region", as_index=False).population_millions.sum().sort_values("population_millions", ascending=True)
    fig = px.bar(g, x="population_millions", y="region", orientation="h", text="population_millions",
                 color_discrete_sequence=[ORANGE])
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(xaxis_title="Population (Millions)")
    return style_fig(fig, height=height, margin=dict(l=90, r=10, t=20, b=40), show_legend=False)


def fig_healthcare_access_by_income(d, height=250):
    g = d.groupby("income_level", as_index=False).healthcare_access_index.mean().sort_values("healthcare_access_index", ascending=True)
    fig = px.bar(g, x="healthcare_access_index", y="income_level", orientation="h", text="healthcare_access_index",
                 color_discrete_sequence=[GREEN])
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(xaxis_title="Healthcare Access Index")
    return style_fig(fig, height=height, margin=dict(l=90, r=10, t=20, b=40), show_legend=False)


def fig_geo_map(d, height=280):
    g = d.groupby("country_name", as_index=False).agg(
        latitude=("latitude", "mean"), longitude=("longitude", "mean"),
        population_millions=("population_millions", "first"),
        respiratory_disease_rate=("respiratory_disease_rate", "mean"),
    )
    fig = px.scatter_geo(
        g, lat="latitude", lon="longitude", size="population_millions",
        color="respiratory_disease_rate", hover_name="country_name",
        color_continuous_scale=[[0, GREEN], [0.5, ORANGE], [1, RED]], size_max=32,
        projection="natural earth",
    )
    fig.update_geos(
        bgcolor="rgba(0,0,0,0)", showland=True, landcolor="#13253A",
        showocean=True, oceancolor=BG, showcountries=True, countrycolor=BORDER,
        showcoastlines=False, showframe=False,
    )
    fig.update_layout(coloraxis_colorbar=dict(title="", thickness=8, tickfont=dict(size=8, color=SUBTEXT)))
    return style_fig(fig, height=height, margin=dict(l=0, r=0, t=0, b=0))


def fig_risk_donut(risk_df, height=200):
    counts = risk_df.risk_category.value_counts().reindex(["High Risk", "Medium Risk", "Low Risk"]).fillna(0)
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values, hole=0.62,
        marker=dict(colors=[RED, ORANGE, GREEN], line=dict(color=BG, width=2)),
        textfont=dict(size=10, color=TEXT),
    ))
    fig.add_annotation(text=f"<b>{int(risk_df.shape[0])}</b><br>Countries", showarrow=False,
                        font=dict(size=13, color=TEXT))
    return style_fig(fig, height=height, show_legend=True, margin=dict(l=10, r=10, t=10, b=10))


def fig_top15_risk(risk_df, height=200):
    g = risk_df.sort_values("risk_score", ascending=False).head(15).sort_values("risk_score")
    fig = px.bar(g, x="risk_score", y="country_name", orientation="h", text="risk_score")
    fig.update_traces(marker_color=RED, texttemplate="%{text:.2f}", textposition="outside",
                       textfont=dict(size=8, color=SUBTEXT))
    return style_fig(fig, height=height, margin=dict(l=90, r=25, t=10, b=24))


def fig_risk_treemap(risk_df, height=200):
    g = risk_df.groupby("region", as_index=False).risk_score.mean()
    fig = px.treemap(g, path=["region"], values="risk_score", color="risk_score",
                      color_continuous_scale=[[0, GREEN], [0.5, ORANGE], [1, RED]])
    fig.update_traces(textfont=dict(size=11, color=TEXT), textinfo="label+value")
    fig.update_layout(coloraxis_showscale=False)
    return style_fig(fig, height=height, margin=dict(l=2, r=2, t=2, b=2))


CORR_COLS = {
    "temperature_celsius": "Temperature", "temp_anomaly_celsius": "Temp Anomaly",
    "heat_wave_days": "Heatwave Days", "air_quality_index": "AQI",
    "respiratory_disease_rate": "Respiratory Rate", "cardio_mortality_rate": "Cardio Mortality",
    "waterborne_disease_incidents": "Waterborne Rate", "heat_related_admissions": "Heat Admissions",
    "gdp_per_capita_usd": "GDP per Capita", "food_security_index": "Food Security Index",
}


def fig_correlation_matrix(d, height=300):
    corr = d[list(CORR_COLS.keys())].corr().rename(index=CORR_COLS, columns=CORR_COLS)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=[[0, BLUE], [0.5, PANEL], [1, RED]], zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        textfont=dict(size=11, color=TEXT),
        colorbar=dict(thickness=8, tickfont=dict(size=8, color=SUBTEXT)),
    ))
    fig.update_xaxes(tickangle=-35)
    return style_fig(fig, height=height, margin=dict(l=90, r=10, t=10, b=70)), corr


def build_insights(d, corr):
    lines = []
    top_temp_region = d.groupby("region").temperature_celsius.mean().idxmax()
    top_temp_val = d.groupby("region").temperature_celsius.mean().max()
    lines.append(("🌡️", f"<b>{top_temp_region}</b> has the highest average temperature ({top_temp_val:.1f}°C)."))

    top_aqi_country = d.groupby("country_name").air_quality_index.mean().idxmax()
    top_aqi_val = d.groupby("country_name").air_quality_index.mean().max()
    lines.append(("💨", f"<b>{top_aqi_country}</b> records the highest average AQI ({top_aqi_val:.0f})."))

    c = corr.copy()
    vals = c.values.copy()
    np.fill_diagonal(vals, 0)
    c = pd.DataFrame(vals, index=c.index, columns=c.columns)
    max_pair = c.abs().stack().idxmax()
    max_val = c.loc[max_pair]
    lines.append(("🔗", f"<b>{max_pair[0]}</b> shows the strongest correlation with <b>{max_pair[1]}</b> ({max_val:.2f})."))

    top_risk_country = risk_df_global.sort_values("risk_score", ascending=False).iloc[0].country_name
    lines.append(("⚠️", f"<b>{top_risk_country}</b> is currently the highest overall climate-health risk country."))

    top_disease_region = d.groupby("region").respiratory_disease_rate.mean().idxmax()
    lines.append(("🫁", f"<b>{top_disease_region}</b> has the highest average respiratory disease rate."))

    lines.append(("🏥", "Improving healthcare access in low-income regions can materially reduce climate-driven health burdens."))
    lines.append(("🌍", "Extreme weather mitigation (heatwaves, floods) should prioritize high-AQI, high-temperature regions."))
    lines.append(("🌾", "Food security support is recommended for regions combining low healthcare access and high climate risk."))
    return lines

# =========================================================================
# HEADER
# =========================================================================
h_l, h_r = st.columns([3, 0.01])
with h_l:
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:2px;">
        <span style="font-size:26px;">🌍</span>
        <div>
            <div class="dash-title">Climate Health Impact Analytics Dashboard (2015–2025)</div>
            <div class="dash-subtitle">Understanding the Impact of Climate Change on Global Public Health</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

tabs = st.tabs(["📊 Overview", "🌡️ Climate", "🏥 Health", "📈 Climate vs Health",
                "🌎 Geographic", "⚠️ Risk Analysis", "🔗 Correlation"])

# =========================================================================
# TAB: OVERVIEW
# =========================================================================
with tabs[0]:
    kpi_row(df)

    r1c1, r1c2, r1c3 = st.columns([1, 1, 1])
    with r1c1:
        panel_open("Priority Countries", "orange")
        chart_card("Top 10 Countries by Composite Risk", fig_top10_composite_risk(df, height=380))
        panel_close()
    with r1c2:
        panel_open("Air Quality & Respiratory Burden", "green")
        chart_card("Respiratory Disease Rate vs AQI", fig_resp_rate_vs_aqi(df, height=380))
        panel_close()
    with r1c3:
        panel_open("Temperature Trend", "blue")
        chart_card("Temperature Trend (Yearly)", fig_line(df, "year", "temperature_celsius", ORANGE, height=380))
        panel_close()

    r2c1, r2c2 = st.columns([2.2, 0.8])
    with r2c1:
        panel_open("Disease Burden by Region", "red")
        chart_card("Disease Burden Composition by Region", fig_disease_burden_region(df, height=400))
        panel_close()
    with r2c2:
        panel_open("Key Insights", "green")
        corr = df[list(CORR_COLS.keys())].corr().rename(index=CORR_COLS, columns=CORR_COLS)
        for icon, text in build_insights(df, corr):
            st.markdown(f"<div class='insight-row'><span>{icon}</span><span>{text}</span></div>", unsafe_allow_html=True)
        panel_close()

# =========================================================================
# TAB: CLIMATE (deep dive)
# =========================================================================
with tabs[1]:
    kpi_row(df)
    panel_open("Climate Trend Analysis — Detailed View", "blue")
    a, b = st.columns(2)
    chart_card("Average Temperature by Year (°C)", fig_line(df, "year", "temperature_celsius", BLUE, height=260), a)
    chart_card("Average Temperature Anomaly by Year (°C)", fig_line(df, "year", "temp_anomaly_celsius", RED, height=260), b)
    a2, b2 = st.columns(2)
    chart_card("Extreme Weather Events by Region", fig_bar_region(df, "extreme_weather_events", ORANGE, height=320), a2)
    chart_card("Climate Variable Correlation Heatmap", fig_climate_corr_heatmap(df, height=340), b2)
    panel_close()

    a3, b3, c3 = st.columns(3)
    chart_card("Heatwave Days by Year", fig_line(df, "year", "heat_wave_days", ORANGE, height=220), a3)
    chart_card("Precipitation by Year (mm)", fig_line(df, "year", "precipitation_mm", BLUE, height=220), b3)
    chart_card("AQI by Year", fig_line(df, "year", "air_quality_index", PURPLE, height=220), c3)

# =========================================================================
# TAB: HEALTH (deep dive)
# =========================================================================
with tabs[2]:
    kpi_row(df)
    panel_open("Public Health Trend Analysis — Detailed View", "green")
    a, b = st.columns(2)
    chart_card("Respiratory Disease Rate Over Years", fig_line(df, "year", "respiratory_disease_rate", PURPLE, height=250), a)
    chart_card("Cardio Mortality Rate Over Years", fig_line(df, "year", "cardio_mortality_rate", RED, height=250), b)
    a2, b2 = st.columns([2.2, 0.8])
    chart_card("Top 10 Heat Related Admission Countries", fig_top10_bar(df, "heat_related_admissions", RED, height=320), a2)
    b2.empty()
    # chart_card("Respiratory Disease Rate by Income Level", fig_box_income(df, "respiratory_disease_rate", height=250), b2)
    # chart_card("Mental Health Index Distribution", fig_hist(df, "mental_health_index", GREEN, height=250), c2)
    panel_close()

# =========================================================================
# TAB: CLIMATE VS HEALTH (deep dive)
# =========================================================================
with tabs[3]:
    kpi_row(df)
    panel_open("Climate vs Health Relationship — Detailed View", "purple")
    c1, c2 = st.columns(2)
    chart_card("Temperature & Respiratory Disease Trend", fig_temperature_resp_trend(df, height=320), c1)
    chart_card("Heatwave Days vs Heat Admissions", fig_heatwave_admissions_combo(df, height=390), c2)
    c3, c4 = st.columns(2)
    chart_card("Precipitation & Waterborne Disease Trend", fig_precipitation_waterborne(df, height=380), c3)
    chart_card("Climate Indicator Heatmap", fig_climate_indicator_heatmap(df, height=380), c4)
    panel_close()

# =========================================================================
# TAB: GEOGRAPHIC (deep dive)
# =========================================================================
with tabs[4]:
    kpi_row(df)
    panel_open("Geographic Risk Map — Detailed View", "blue")
    chart_card("Respiratory Disease Rate & Population by Country", fig_geo_map(df, height=520))
    panel_close()

# =========================================================================
# TAB: RISK ANALYSIS (deep dive)
# =========================================================================
with tabs[5]:
    kpi_row(df)
    panel_open("Risk Analysis — Detailed View", "red")
    risk_df = compute_risk(df)
    rc1, rc2, rc3 = st.columns([0.9, 1.3, 1])
    chart_card("Risk Distribution (Global)", fig_risk_donut(risk_df, height=380), rc1)
    chart_card("Top 15 High Risk Countries", fig_top15_risk(risk_df, height=380), rc2)
    chart_card("Risk by Region (Treemap)", fig_risk_treemap(risk_df, height=380), rc3)
    panel_close()
    st.dataframe(risk_df.sort_values("risk_score", ascending=False)
                 .rename(columns={"risk_score": "Risk Score", "risk_category": "Risk Category",
                                   "country_name": "Country", "region": "Region"})
                 [["Country", "Region", "Risk Score", "Risk Category"]],
                 use_container_width=True, height=220)

# =========================================================================
# TAB: CORRELATION (deep dive)
# =========================================================================
with tabs[6]:
    kpi_row(df)
    panel_open("Correlation Matrix — Detailed View", "orange")
    fig_corr, corr_matrix = fig_correlation_matrix(df, height=560)
    st.plotly_chart(fig_corr, width="stretch", config={"displayModeBar": False})
    panel_close()
    panel_open("Executive Insights", "green")
    ins_cols = st.columns(2)
    all_insights = build_insights(df, corr_matrix)
    half = len(all_insights) // 2 + len(all_insights) % 2
    for i, (icon, text) in enumerate(all_insights):
        col = ins_cols[0] if i < half else ins_cols[1]
        col.markdown(f"<div class='insight-row'><span>{icon}</span><span>{text}</span></div>", unsafe_allow_html=True)
    panel_close()
