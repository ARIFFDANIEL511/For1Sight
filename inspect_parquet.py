import pandas as pd

# Update this path if your parquet file/folder name is different
path = "for1sight_warehouse/hf_laptimes.parquet"

try:
    df = pd.read_parquet(path)

    print("--- SCHEMA ---")
    print(df.info())

    print("\n--- FIRST 10 ROWS ---")
    print(df.head(10))

    print("\n--- SUMMARY ---")
    print(df.describe())

except Exception as e:
    print(f"Error reading Parquet file: {e}")