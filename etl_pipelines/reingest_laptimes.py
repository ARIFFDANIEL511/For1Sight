import os
import glob
from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, split, col, element_at, regexp_replace

spark = SparkSession.builder \
    .appName("For1Sight_Fix_Laptimes") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("Searching for raw files...")

file_list = glob.glob("raw_hf_data/**/laptimes.json", recursive=True)

if not file_list:
    print("❌ ERROR: Could not find any files. Did you delete the 'raw_hf_data' folder?")
else:
    print(f"Found {len(file_list)} laptime files. Injecting context...")
    
    df = spark.read.json(file_list)
    df_with_path = df.withColumn("decoded_path", regexp_replace(input_file_name(), "%20", " "))
    path_array = split(col("decoded_path"), "/")

    # SHIFTED INDICES: Account for the hidden Driver folder
    df_fixed = df_with_path \
        .withColumn("Year", element_at(path_array, -5)) \
        .withColumn("Event", element_at(path_array, -4)) \
        .withColumn("Session", element_at(path_array, -3)) \
        .drop("decoded_path")

    df_fixed.write.mode("overwrite").parquet("for1sight_warehouse/hf_laptimes.parquet")
    print("✓ Successfully injected CORRECT Year, Event, and Session into Parquet warehouse!")

spark.stop()