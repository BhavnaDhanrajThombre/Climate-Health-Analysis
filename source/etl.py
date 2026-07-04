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
    "password": "root",          # Change if your MySQL password is different
    "database": "climate_health_analysis"
}

TABLE_NAME = "climate_health"

# -----------------------------
# Locate CSV File
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
    print(f"Number of records: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")

    # -----------------------------
    # Step 2: Connect to MySQL
    # -----------------------------
    print("\nConnecting to MySQL...")

    conn = mysql.connector.connect(**DB_CONFIG)

    if conn.is_connected():
        print("Connected to MySQL successfully!")

    cursor = conn.cursor()

    # -----------------------------
    # Step 3: Create Table
    # -----------------------------
    print("\nCreating table if it doesn't exist...")

    columns = []

    for col in df.columns:
        columns.append(f"`{col}` TEXT")

    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        id INT AUTO_INCREMENT PRIMARY KEY,
        {", ".join(columns)}
    );
    """

    cursor.execute(create_table_query)

    print(f"Table '{TABLE_NAME}' is ready.")

    # -----------------------------
    # Step 4: Insert Records
    # -----------------------------
    print("\nLoading data into MySQL...")

    column_names = ", ".join([f"`{col}`" for col in df.columns])
    placeholders = ", ".join(["%s"] * len(df.columns))

    insert_query = f"""
    INSERT INTO {TABLE_NAME}
    ({column_names})
    VALUES ({placeholders})
    """

    data = df.fillna("").values.tolist()

    cursor.executemany(insert_query, data)

    conn.commit()

    print(f"{cursor.rowcount} records inserted successfully.")

    # -----------------------------
    # Step 5: Verify Data
    # -----------------------------
    print("\nVerifying data...")

    cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
    total = cursor.fetchone()[0]

    print(f"Total records in database: {total}")

    print("\nFirst 5 Records:")

    cursor.execute(f"SELECT * FROM {TABLE_NAME} LIMIT 5")

    rows = cursor.fetchall()

    for row in rows:
        print(row)

    print("\nETL Process Completed Successfully!")

except FileNotFoundError:
    print(f"\nCSV file not found:\n{CSV_FILE}")

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