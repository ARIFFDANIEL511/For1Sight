import os
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("For1Sight_Inspector") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("Ripping open Parquet warehouse...\n")
df = spark.read.parquet("for1sight_warehouse/hf_laptimes.parquet")

# Print the Driver, the exact raw string of 'time', and the 'Session'
df.select("drv", "time", "Session").show(20, truncate=False)

spark.stop()