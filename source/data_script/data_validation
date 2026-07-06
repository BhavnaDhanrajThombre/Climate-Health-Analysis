import pandas as pd

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/processed/climate_health_cleaned.csv")

print("=" * 60)
print("DATA VALIDATION REPORT")
print("=" * 60)

# -----------------------------
# Dataset Shape
# -----------------------------
print("\nDataset Shape")
print(df.shape)

# -----------------------------
# Missing Values
# -----------------------------
print("\nMissing Values")

missing = df.isnull().sum()

if missing.sum() == 0:
    print("No Missing Values Found")
else:
    print(missing[missing > 0])

# -----------------------------
# Duplicate Records
# -----------------------------
print("\nDuplicate Records")

duplicates = df.duplicated().sum()

print(f"Duplicate Rows : {duplicates}")

# -----------------------------
# Data Types
# -----------------------------
print("\nData Types")

print(df.dtypes)

# -----------------------------
# Latitude Validation
# -----------------------------
invalid_lat = df[
    (df["latitude"] < -90) |
    (df["latitude"] > 90)
]

print(f"\nInvalid Latitude Values : {len(invalid_lat)}")

# -----------------------------
# Longitude Validation
# -----------------------------
invalid_lon = df[
    (df["longitude"] < -180) |
    (df["longitude"] > 180)
]

print(f"Invalid Longitude Values : {len(invalid_lon)}")

# -----------------------------
# Temperature Validation
# -----------------------------
invalid_temp = df[
    (df["temperature_celsius"] < -90) |
    (df["temperature_celsius"] > 60)
]

print(f"Invalid Temperature Values : {len(invalid_temp)}")

# -----------------------------
# PM2.5 Validation
# -----------------------------
invalid_pm = df[
    df["pm25_ugm3"] < 0
]

print(f"Invalid PM2.5 Values : {len(invalid_pm)}")

# -----------------------------
# AQI Validation
# -----------------------------
invalid_aqi = df[
    df["air_quality_index"] < 0
]

print(f"Invalid AQI Values : {len(invalid_aqi)}")

# -----------------------------
# Binary Validation
# -----------------------------
invalid_drought = df[
    ~df["drought_indicator"].isin([0,1])
]

invalid_flood = df[
    ~df["flood_indicator"].isin([0,1])
]

print(f"Invalid Drought Indicator : {len(invalid_drought)}")
print(f"Invalid Flood Indicator : {len(invalid_flood)}")

# -----------------------------
# Month Validation
# -----------------------------
invalid_month = df[
    ~df["month"].between(1,12)
]

print(f"Invalid Month Values : {len(invalid_month)}")

# -----------------------------
# Week Validation
# -----------------------------
invalid_week = df[
    ~df["week"].between(1,53)
]

print(f"Invalid Week Values : {len(invalid_week)}")

# -----------------------------
# GDP Validation
# -----------------------------
invalid_gdp = df[
    df["gdp_per_capita_usd"] < 0
]

print(f"Invalid GDP Values : {len(invalid_gdp)}")

# -----------------------------
# Healthcare Index
# -----------------------------
invalid_health = df[
    ~df["healthcare_access_index"].between(0,100)
]

print(f"Invalid Healthcare Index : {len(invalid_health)}")

# -----------------------------
# Food Security Index
# -----------------------------
invalid_food = df[
    ~df["food_security_index"].between(0,100)
]

print(f"Invalid Food Security Index : {len(invalid_food)}")

# -----------------------------
# Mental Health Index
# -----------------------------
invalid_mental = df[
    ~df["mental_health_index"].between(0,100)
]

print(f"Invalid Mental Health Index : {len(invalid_mental)}")

# -----------------------------
# Overall Validation
# -----------------------------
total_errors = (
    len(invalid_lat)
    + len(invalid_lon)
    + len(invalid_temp)
    + len(invalid_pm)
    + len(invalid_aqi)
    + len(invalid_drought)
    + len(invalid_flood)
    + len(invalid_month)
    + len(invalid_week)
    + len(invalid_gdp)
    + len(invalid_health)
    + len(invalid_food)
    + len(invalid_mental)
    + duplicates
    + missing.sum()
)

print("\n" + "=" * 60)

if total_errors == 0:
    print("DATA VALIDATION PASSED")
else:
    print(f"DATA VALIDATION FAILED")
    print(f"Total Validation Errors : {total_errors}")

print("=" * 60)

# extra remove afterward

print(df["air_quality_index"].min())
print(df["air_quality_index"].max())

print(df["healthcare_access_index"].min())
print(df["healthcare_access_index"].max())

print(df[df["air_quality_index"] < 0].head())

print(df[~df["healthcare_access_index"].between(0,100)].head())