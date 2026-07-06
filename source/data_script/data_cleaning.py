import pandas as pd

# Load the raw dataset
df = pd.read_csv("data/raw/global_climate_health_impact_tracker_2015_2025.csv")

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Verify data types
print("\nUpdated Data Types:")
print(df.dtypes)

#aqi shoul not be negative sign error conversion
df["air_quality_index"] = df["air_quality_index"].abs()

# Healthcare Access Index should be between 0 and 100
df["healthcare_access_index"] = df["healthcare_access_index"].clip(lower=0, upper=100)

# Save cleaned dataset
df.to_csv("data/processed/climate_health_cleaned.csv", index=False)

print("\nData type correction completed successfully!")

