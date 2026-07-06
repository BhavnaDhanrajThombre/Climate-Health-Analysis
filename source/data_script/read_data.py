import pandas as pd

# Load the dataset
df = pd.read_csv("data/raw/global_climate_health_impact_tracker_2015_2025.csv")

# Display the first 5 rows
print("First 5 Rows:")
print(df.head())

# Display the shape of the dataset
print("\nDataset Shape:")
print(df.shape)
# Display the column names
print("\nColumn Names:")
print(df.columns.tolist())

# Display information about the dataset
print("\nDataset Information:")
df.info()

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check for duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Display summary statistics
print("\nSummary Statistics:")
print(df.describe(include="all"))