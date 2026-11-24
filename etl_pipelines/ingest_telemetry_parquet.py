import requests
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp
import os
import time

# ==========================================
# 1. INITIALIZE WAREHOUSE
# ==========================================
spark = SparkSession.builder \
    .appName("For1Sight_Bulk_Ingestion") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

os.makedirs("for1sight_warehouse", exist_ok=True)

# ==========================================
# 2. FETCH ALL RACE SESSIONS (2024-2026)
# ==========================================
print("Fetching official Formula 1 calendar...")
sessions_url = "https://api.openf1.org/v1/sessions?session_name=Race"
try:
    all_sessions = requests.get(sessions_url, timeout=10).json()
    
    # Filter only the years requested in your FYP proposal
    target_years = [2024, 2025, 2026]
    valid_races = [s for s in all_sessions if s.get('year') in target_years]
    
    # Sort chronologically
    valid_races = sorted(valid_races, key=lambda x: x['date_start'])
    
    print(f"Found {len(valid_races)} Grand Prix events to ingest.")
except Exception as e:
    print(f"Failed to fetch calendar: {e}")
    valid_races = []

# ==========================================
# 3. BULK INGESTION LOOP
# ==========================================
for index, race in enumerate(valid_races, 1):
    session_key = race['session_key']
    race_title = f"{race['year']} {race['country_name']} Grand Prix"
    
    print(f"\n[{index}/{len(valid_races)}] Extracting: {race_title} (Key: {session_key})")
    
    try:
        # 3 API calls per race
        car_data = requests.get(f"https://api.openf1.org/v1/car_data?session_key={session_key}", timeout=20).json()
        weather_data = requests.get(f"https://api.openf1.org/v1/weather?session_key={session_key}", timeout=20).json()
        stints_data = requests.get(f"https://api.openf1.org/v1/stints?session_key={session_key}", timeout=20).json()
        
        # --- PROCESS CAR TELEMETRY ---
        if isinstance(car_data, list) and len(car_data) > 0:
            df_car = spark.createDataFrame(pd.DataFrame(car_data))
            df_car = df_car \
                .withColumn("session_key", col("session_key").cast("integer")) \
                .withColumn("driver_number", col("driver_number").cast("integer")) \
                .withColumn("speed", col("speed").cast("double")) \
                .withColumn("rpm", col("rpm").cast("integer")) \
                .withColumn("throttle", col("throttle").cast("double")) \
                .withColumn("brake", col("brake").cast("double")) \
                .withColumn("n_gear", col("n_gear").cast("integer")) \
                .withColumn("drs", col("drs").cast("integer")) \
                .withColumn("date", to_timestamp(col("date")))
            
            df_car.write.mode("append").partitionBy("session_key").parquet("for1sight_warehouse/car_telemetry.parquet")
            print("  ✓ Car Telemetry Saved")

        # --- PROCESS WEATHER ---
        if isinstance(weather_data, list) and len(weather_data) > 0:
            df_weather = spark.createDataFrame(pd.DataFrame(weather_data))
            df_weather = df_weather \
                .withColumn("session_key", col("session_key").cast("integer")) \
                .withColumn("air_temperature", col("air_temperature").cast("double")) \
                .withColumn("track_temperature", col("track_temperature").cast("double")) \
                .withColumn("date", to_timestamp(col("date")))
            
            df_weather.write.mode("append").partitionBy("session_key").parquet("for1sight_warehouse/weather_telemetry.parquet")
            print("  ✓ Weather Data Saved")

        # --- PROCESS STINTS ---
        if isinstance(stints_data, list) and len(stints_data) > 0:
            df_stints = spark.createDataFrame(pd.DataFrame(stints_data))
            df_stints = df_stints \
                .withColumn("session_key", col("session_key").cast("integer")) \
                .withColumn("driver_number", col("driver_number").cast("integer")) \
                .withColumn("stint_number", col("stint_number").cast("integer"))
            
            df_stints.write.mode("append").partitionBy("session_key").parquet("for1sight_warehouse/tyre_stints.parquet")
            print("  ✓ Tyre Stints Saved")
            
    except Exception as e:
        print(f"  ✕ Error ingesting {race_title}: {e}")

    # =======================================================
    # CRITICAL: 7-Second Delay to avoid API Rate Limiting
    # (3 requests made per loop. 30 requests allowed per minute)
    # =======================================================
    print("  -> Cooling down API connection...")
    time.sleep(7) 

print("\n🚀 FULL 2024-2026 WAREHOUSE INGESTION COMPLETE!")
spark.stop()