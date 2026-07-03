import pandas as pd

# Load the raw dataset
df = pd.read_csv("data/raw/global_climate_health_impact_tracker_2015_2025.csv")

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Verify data types
print("\nUpdated Data Types:")
print(df.dtypes)

# Save cleaned dataset
df.to_csv("data/processed/climate_health_cleaned.csv", index=False)

print("\nData type correction completed successfully!")
