import streamlit as st

from database import load_data
from components.kpi_cards import show_kpis

st.set_page_config(layout="wide")

# Load data
df = load_data()

# Sidebar filters
country = st.sidebar.multiselect(
    "Country",
    sorted(df["country_name"].unique()),
    default=sorted(df["country_name"].unique())
)

year = st.sidebar.multiselect(
    "Year",
    sorted(df["year"].unique()),
    default=sorted(df["year"].unique())
)

region = st.sidebar.multiselect(
    "Region",
    sorted(df["region"].unique()),
    default=sorted(df["region"].unique())
)

income = st.sidebar.multiselect(
    "Income Level",
    sorted(df["income_level"].unique()),
    default=sorted(df["income_level"].unique())
)

# Apply filters
filtered_df = df[
    (df["country_name"].isin(country)) &
    (df["year"].isin(year)) &
    (df["region"].isin(region)) &
    (df["income_level"].isin(income))
]

# Show KPI cards
show_kpis(filtered_df)