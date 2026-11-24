import time
import requests
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# ==========================================
# 1. INITIALIZE (Set up PySpark within your 16GB limits)
# ==========================================
spark = SparkSession.builder \
    .appName("For1Sight_ETL_Pipeline") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

# ==========================================
# 2. EXTRACT: Kaggle Vopani Dataset
# ==========================================
# Ergast uses '\N' for missing values. We parse them directly into True Nulls.
drivers_df = spark.read.csv(
    "drivers.csv", 
    header=True, 
    inferSchema=True, 
    nullValue=r"\N"
)

# ==========================================
# 3. EXTRACT: OpenF1 API Data (2024-2026)
# ==========================================
years_to_fetch = [2024, 2025, 2026]
all_laps = []

for year in years_to_fetch:
    print(f"Fetching Race sessions for {year}...")
    
    # Filter the sessions API to only return Sunday Grand Prix Races
    session_url = f"https://api.openf1.org/v1/sessions?year={year}&session_name=Race"
    sessions_response = requests.get(session_url)
    sessions_data = sessions_response.json()
    
    for session in sessions_data:
        session_key = session['session_key']
        print(f"  -> Pulling laps for Session {session_key} ({session['country_name']})...")
        
        # Query the laps API using the unique session key
        laps_url = f"https://api.openf1.org/v1/laps?session_key={session_key}"
        laps_response = requests.get(laps_url)
        
        if laps_response.status_code == 200:
            all_laps.extend(laps_response.json())
        
        # CRITICAL: The OpenF1 free tier allows a maximum of 30 requests per minute.
        # We must pause for 2.5 seconds per loop to prevent your IP from being blocked.
        time.sleep(2.5)

# Convert the deeply nested JSON array into a Pandas DataFrame first (handles missing JSON keys safely), 
# then convert it to a distributed PySpark DataFrame.
laps_pandas_df = pd.DataFrame(all_laps).fillna(0)
# Force column types to string to avoid Spark schema inference conflicts on mixed arrays
laps_pandas_df = laps_pandas_df.astype(str) 
api_laps_df = spark.createDataFrame(laps_pandas_df)

# ==========================================
# 4. TRANSFORM: Clean and Merge the Data
# ==========================================
# Cast the OpenF1 API 'driver_number' and the Kaggle 'number' to integers so they match
api_laps_df = api_laps_df.withColumn("driver_number", col("driver_number").cast("integer"))
cleaned_drivers = drivers_df.withColumn("number", col("number").cast("integer")).dropna(subset=["number"])

# Join the API Lap telemetry with the historical Driver information
final_warehouse_df = api_laps_df.join(
    cleaned_drivers,
    api_laps_df.driver_number == cleaned_drivers.number,
    "inner"
)

# Select only the features necessary for your ML Prediction Model
model_ready_df = final_warehouse_df.select(
    "driverId",
    "driver_number",
    "code",
    "session_key",
    "lap_number",
    "lap_duration",
    "is_pit_out_lap",
    "forename",
    "surname"
)

# ==========================================
# 5. LOAD: Save to HDFS Data Warehouse
# ==========================================
print("Writing to HDFS Parquet Data Warehouse...")

# We use the hdfs:// URI to route the save operation to your active Hadoop node
model_ready_df.write.mode("overwrite").parquet("hdfs://localhost:9000/for1sight_warehouse/laps_2024_2026.parquet")

print("ETL Pipeline Complete! File saved successfully to HDFS.")