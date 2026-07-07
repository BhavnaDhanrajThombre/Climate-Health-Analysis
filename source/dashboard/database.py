# import pandas as pd
# from sqlalchemy import create_engine

# DB_CONFIG = {
#     "host": "localhost",
#     "user": "root",
#     "password": "root",
#     "database": "climate_health_analysis"
# }

# engine = create_engine(
#     f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
#     f"@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
# )

# def load_data():
#     query = "SELECT * FROM climate_health"
#     return pd.read_sql(query, engine)


import pandas as pd
from sqlalchemy import create_engine

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "climate_health_analysis"
}

engine = create_engine(
    f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
)

def load_data():

    query = "SELECT * FROM climate_health"

    df = pd.read_sql(query, engine)

    print("\n===== DATA TYPES =====")
    print(df.dtypes)

    print("\n===== FIRST 5 ROWS =====")
    print(df.head())

    print("\n===== TEMPERATURE =====")
    print(df["temperature_celsius"].head())

    return df