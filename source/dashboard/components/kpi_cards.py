import streamlit as st

def show_kpis(filtered_df):

    # Handle empty dataframe
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        return

    # ==========================
    # DEBUG CODE (Add Here)
    # ==========================
    st.subheader("Debug Information")

    st.write("Data Types:")
    st.write(filtered_df.dtypes)

    st.write("Temperature Column:")
    st.write(filtered_df["temperature_celsius"].head(10))

    st.write("First Value Type:")
    st.write(type(filtered_df["temperature_celsius"].iloc[0]))

    # ==========================
    # KPI Calculations
    # ==========================

    total_countries = filtered_df["country_name"].nunique()

    avg_temp = filtered_df["temperature_celsius"].mean()

    avg_pm25 = filtered_df["pm25_ugm3"].mean()

    avg_air_quality = filtered_df["air_quality_index"].mean()

    avg_respiratory = filtered_df["respiratory_disease_rate"].mean()

    total_heat_admissions = filtered_df["heat_related_admissions"].sum()

    total_weather_events = filtered_df["extreme_weather_events"].sum()

    # -------------------------
    # Display KPI Cards
    # -------------------------
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

    with col1:
        st.metric(
            label="🌍 Countries",
            value=total_countries
        )

    with col2:
        st.metric(
            label="🌡 Avg Temp",
            value=f"{avg_temp:.2f} °C"
        )

    with col3:
        st.metric(
            label="🌫 Avg PM2.5",
            value=f"{avg_pm25:.2f}"
        )

    with col4:
        st.metric(
            label="🌬 Avg AQI",
            value=f"{avg_air_quality:.2f}"
        )

    with col5:
        st.metric(
            label="🫁 Respiratory",
            value=f"{avg_respiratory:.2f}"
        )

    with col6:
        st.metric(
            label="🏥 Heat Admissions",
            value=f"{total_heat_admissions:,.0f}"
        )

    with col7:
        st.metric(
            label="⚡ Weather Events",
            value=f"{total_weather_events:,.0f}"
        )