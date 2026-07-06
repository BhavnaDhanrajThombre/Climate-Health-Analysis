import pandas as pd

# Select numerical columns
df = pd.read_csv("data/raw/global_climate_health_impact_tracker_2015_2025.csv")
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

outlier_summary = []

for col in numerical_columns:
    
    # Skip binary columns
    if df[col].nunique() <= 2:
        continue

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    outlier_summary.append({
        "Column": col,
        "Total Values": len(df),
        "Outliers": len(outliers),
        "Outlier %": round((len(outliers) / len(df)) * 100, 2),
        "Lower Bound": round(lower_bound, 2),
        "Upper Bound": round(upper_bound, 2),
        "Future Insight": (
            "Potential climate extreme / health hotspot"
            if len(outliers) > 0
            else "No significant outliers"
        )
    })

# Create summary DataFrame
outlier_df = pd.DataFrame(outlier_summary)

# Display results
print(outlier_df)

# Save report (optional)
outlier_df.to_csv("outlier_analysis_report.csv", index=False)
