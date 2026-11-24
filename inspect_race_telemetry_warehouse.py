from pathlib import Path
import os
import tempfile

# Keep temporary files inside the existing project runtime
PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = PROJECT_ROOT / "for1sight_runtime"
TEMP_ROOT = RUNTIME_ROOT / "python_temp"
SPARK_ROOT = RUNTIME_ROOT / "spark_local" / "warehouse_inspection"

TEMP_ROOT.mkdir(parents=True, exist_ok=True)
SPARK_ROOT.mkdir(parents=True, exist_ok=True)

os.environ["TEMP"] = str(TEMP_ROOT)
os.environ["TMP"] = str(TEMP_ROOT)
os.environ["TMPDIR"] = str(TEMP_ROOT)
tempfile.tempdir = str(TEMP_ROOT)

from pyspark.sql import SparkSession

HDFS_URI = "hdfs://localhost:9000"

DATASETS = [
    (
        "SILVER TELEMETRY",
        f"{HDFS_URI}/for1sight_warehouse/silver/race_telemetry",
        [
            "year", "event_name", "event_slug", "session_type",
            "driver_code", "lap_number", "sample_index",
            "time_s", "rpm", "speed_kmh", "gear",
            "throttle_pct", "brake", "drs", "distance_m",
            "rel_distance", "x", "y", "driving_style_usable"
        ]
    ),
    (
        "GOLD TELEMETRY AUDIT",
        f"{HDFS_URI}/for1sight_warehouse/gold/race_telemetry/telemetry_audit",
        [
            "year", "event_name", "event_slug", "driver_code",
            "lap_number", "sample_count", "quality_status",
            "track_map_usable", "driving_style_usable", "issues"
        ]
    ),
    (
        "GOLD COMPARISON CANDIDATES",
        f"{HDFS_URI}/for1sight_warehouse/gold/race_telemetry/comparison_candidates",
        [
            "year", "event_name", "event_slug", "session_type",
            "driver_a", "driver_b", "common_lap_count",
            "first_common_lap", "last_common_lap", "common_laps"
        ]
    )
]

spark = (
    SparkSession.builder
    .appName("For1Sight_Warehouse_ReadOnly_Inspection")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .config("spark.driver.maxResultSize", "1g")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.sql.parquet.compression.codec", "snappy")
    .config("spark.local.dir", str(SPARK_ROOT))
    .config("spark.hadoop.fs.defaultFS", HDFS_URI)
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    print("=" * 75)
    print("FOR1SIGHT HDFS WAREHOUSE VALIDATION")
    print("READ-ONLY INSPECTION")
    print("=" * 75)

    print("Spark version:", spark.version)

    for name, path, requested_columns in DATASETS:
        print("\n" + "=" * 75)
        print(name)
        print("=" * 75)
        print("HDFS path:", path)

        df = spark.read.parquet(path)

        print("\nTotal columns:", len(df.columns))
        print("Column names:", df.columns)

        print("\nACTUAL PARQUET SCHEMA")
        df.printSchema()

        available = [
            column for column in requested_columns
            if column in df.columns
        ]

        if not available:
            available = df.columns[:10]

        print("\nSAMPLE RECORDS (maximum 3 rows)")
        df.select(*available).limit(3).show(
            3, truncate=False, vertical=True
        )

        print("\nDataset inspection completed:", name)

    print("\n" + "=" * 75)
    print("FOR1SIGHT WAREHOUSE INSPECTION COMPLETED")
    print("=" * 75)

finally:
    spark.stop()
