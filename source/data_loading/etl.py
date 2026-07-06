import os
import pandas as pd
import mysql.connector
from mysql.connector import Error

# -----------------------------
# Database Configuration
# -----------------------------
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "1234",          # Change if required
    "database": "climate_health_analysis"
}

TABLE_NAME = "climate_health"

# -----------------------------
# Locate CSV File
# -----------------------------
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

CSV_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "climate_health_cleaned.csv"
)

try:
    # -----------------------------
    # Step 1: Read CSV
    # -----------------------------
    print("Reading CSV file...")

    df = pd.read_csv(CSV_FILE)

    print("Dataset loaded successfully.")
    print(f"Number of records : {len(df)}")
    print(f"Number of columns : {len(df.columns)}")

    # -----------------------------
    # Step 2: Connect MySQL
    # -----------------------------
    print("\nConnecting to MySQL...")

    conn = mysql.connector.connect(**DB_CONFIG)

    if conn.is_connected():
        print("Connected successfully!")

    cursor = conn.cursor()
    
    

    # -----------------------------
    # Step 3: Create Table
    # -----------------------------
    print("\nCreating table...")

    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (

        record_id INT PRIMARY KEY,

        country_code VARCHAR(10),
        country_name VARCHAR(100),
        region VARCHAR(50),
        income_level VARCHAR(30),

        date DATE,
        year INT,
        month TINYINT,
        week TINYINT,

        latitude DECIMAL(9,6),
        longitude DECIMAL(9,6),

        population_millions INT,

        temperature_celsius DECIMAL(6,2),
        temp_anomaly_celsius DECIMAL(6,2),
        precipitation_mm DECIMAL(8,2),

        heat_wave_days INT,

        drought_indicator BOOLEAN,
        flood_indicator BOOLEAN,

        extreme_weather_events INT,

        pm25_ugm3 DECIMAL(8,2),
        air_quality_index DECIMAL(8,2),

        respiratory_disease_rate DECIMAL(8,2),
        cardio_mortality_rate DECIMAL(8,2),

        vector_disease_risk_score DECIMAL(8,2),

        waterborne_disease_incidents DECIMAL(10,2),

        heat_related_admissions DECIMAL(10,2),

        healthcare_access_index DECIMAL(6,2),

        gdp_per_capita_usd DECIMAL(12,2),

        mental_health_index DECIMAL(6,2),

        food_security_index DECIMAL(6,2)

    );
    """

    cursor.execute(create_table_query)

    print(f"Table '{TABLE_NAME}' is ready.")
    
        # -----------------------------
    # Step 3.5: Delete Existing Data
    # -----------------------------
    print("\nDeleting existing records...")

    cursor.execute(f"DELETE FROM {TABLE_NAME}")

    conn.commit()

    print("Existing records deleted successfully.")
    
    

    # -----------------------------
    # Step 4: Insert Records
    # -----------------------------
    print("\nLoading data into MySQL...")

    column_names = ", ".join(df.columns)
    placeholders = ", ".join(["%s"] * len(df.columns))

    insert_query = f"""
    INSERT INTO {TABLE_NAME}
    ({column_names})
    VALUES ({placeholders})
    """

    data = df.where(pd.notnull(df), None).values.tolist()

    cursor.executemany(insert_query, data)

    conn.commit()

    print(f"{cursor.rowcount} records inserted successfully.")

    # -----------------------------
    # Step 5: Verify
    # -----------------------------
    print("\nVerifying data...")

    cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")

    total = cursor.fetchone()[0]

    print(f"Total Records : {total}")

    print("\nFirst 5 Records:\n")

    cursor.execute(f"SELECT * FROM {TABLE_NAME} LIMIT 5")

    rows = cursor.fetchall()

    for row in rows:
        print(row)

    print("\nETL Process Completed Successfully!")

except FileNotFoundError:
    print(f"\nCSV File Not Found:\n{CSV_FILE}")

except Error as e:
    print("\nMySQL Error:")
    print(e)

except Exception as e:
    print("\nUnexpected Error:")
    print(e)

finally:
    try:
        if conn.is_connected():
            cursor.close()
            conn.close()
            print("\nMySQL connection closed.")
    except:
        pass