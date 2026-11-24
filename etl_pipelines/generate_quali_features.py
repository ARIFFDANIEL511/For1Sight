import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, min as spark_min, explode, arrays_zip

# Create a local temp directory
temp_dir = os.path.join(os.getcwd(), "spark_temp")
os.makedirs(temp_dir, exist_ok=True)

spark = SparkSession.builder \
    .appName("For1Sight_Feature_Engineering") \
    .config("spark.driver.memory", "4g") \
    .config("spark.local.dir", temp_dir) \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
warehouse_dir = "for1sight_warehouse"

print("Loading compressed Parquet lap times...")
laptimes_df = spark.read.parquet(f"{warehouse_dir}/hf_laptimes.parquet")

# ==========================================
# 2. DATA CLEANING: THE "EXPLODE" FIX
# ==========================================
print("Exploding arrays into individual laps...")

zipped_df = laptimes_df.withColumn(
    "lap_struct", 
    arrays_zip(col("drv"), col("time"))
)
exploded_df = zipped_df.withColumn("lap_struct", explode(col("lap_struct")))

parsed_laps = exploded_df \
    .withColumn("Driver", col("lap_struct.drv").cast("string")) \
    .withColumn("LapTime", col("lap_struct.time").cast("float")) \
    .filter(col("LapTime").isNotNull()) \
    .filter(col("LapTime") > 40)

print(f"-> Total valid laps found after exploding: {parsed_laps.count()}")

# ==========================================
# 3. SEPARATE PRACTICE PACE FROM QUALIFYING
# ==========================================
# FIX: Added official FastF1 "Practice" session names
practice_df = parsed_laps.filter(col("Session").isin("Practice 1", "Practice 2", "Practice 3", "FP1", "FP2", "FP3")) \
    .groupBy("Year", "Event", "Driver") \
    .agg(spark_min("LapTime").alias("Best_FP_Time"))

print(f"-> Drivers with Practice times extracted: {practice_df.count()}")

quali_df = parsed_laps.filter(col("Session").isin("Qualifying", "Sprint Qualifying", "SQ", "Sprint")) \
    .groupBy("Year", "Event", "Driver") \
    .agg(spark_min("LapTime").alias("Best_Q_Time"))

print(f"-> Drivers with Qualifying times extracted: {quali_df.count()}")

# ==========================================
# 4. JOIN FEATURES TO TARGET VARIABLE
# ==========================================
print("Fusing Practice features with Qualifying targets...")

ml_training_table = practice_df.join(
    quali_df, 
    on=["Year", "Event", "Driver"], 
    how="inner"
)

ml_training_table = ml_training_table.withColumn(
    "Pace_Improvement", 
    col("Best_FP_Time") - col("Best_Q_Time")
)

# ==========================================
# 5. SAVE THE MACHINE LEARNING TABLE
# ==========================================
output_path = f"{warehouse_dir}/ml_training_table.parquet"
ml_training_table.write.mode("overwrite").parquet(output_path)

print(f"✓ Success! Machine Learning Training Table saved to {output_path}")
print(f"FINAL JOINED ROWS EXTRACTED: {ml_training_table.count()}")

spark.stop()