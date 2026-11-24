
from pathlib import Path
import os
import tempfile

# --------------------------------------------------
# 1. ENVIRONMENT CONFIGURATION
# --------------------------------------------------

RUNTIME = Path(r"C:\for1sight_runtime")
TEMP_ROOT = RUNTIME / "python_temp"
SPARK_ROOT = RUNTIME / "spark_local" / "pair_validation"

TEMP_ROOT.mkdir(parents=True, exist_ok=True)
SPARK_ROOT.mkdir(parents=True, exist_ok=True)

os.environ["TEMP"] = str(TEMP_ROOT)
os.environ["TMP"] = str(TEMP_ROOT)
os.environ["TMPDIR"] = str(TEMP_ROOT)
tempfile.tempdir = str(TEMP_ROOT)

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# --------------------------------------------------
# 2. HDFS CONFIGURATION
# --------------------------------------------------

HDFS = "hdfs://localhost:9000"

SILVER_PATH = (
    f"{HDFS}/for1sight_warehouse/"
    "silver/race_telemetry"
)

AUDIT_PATH = (
    f"{HDFS}/for1sight_warehouse/"
    "gold/race_telemetry/telemetry_audit"
)

CANDIDATE_PATH = (
    f"{HDFS}/for1sight_warehouse/"
    "gold/race_telemetry/comparison_candidates"
)

YEAR = 2026
EVENT = "australian_grand_prix"
DRIVERS = ["ANT", "HAM"]

KEYS = [
    "year",
    "event_slug",
    "session_type",
    "driver_code",
    "lap_number"
]

# --------------------------------------------------
# 3. START SPARK
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("For1Sight_Telemetry_Pair_Validation")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .config("spark.driver.maxResultSize", "1g")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.local.dir", str(SPARK_ROOT))
    .config("spark.hadoop.fs.defaultFS", HDFS)
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:

    print("=" * 70)
    print("FOR1SIGHT TELEMETRY PAIR VALIDATION")
    print("=" * 70)

    # --------------------------------------------------
    # 4. READ COMPARISON CANDIDATES
    # --------------------------------------------------

    candidates = spark.read.parquet(CANDIDATE_PATH)

    candidate = (
        candidates
        .filter(
            (F.col("year") == YEAR) &
            (F.col("event_slug") == EVENT) &
            (F.col("session_type") == "Race") &
            (F.col("driver_a") == DRIVERS[0]) &
            (F.col("driver_b") == DRIVERS[1])
        )
        .limit(1)
        .collect()
    )

    if not candidate:
        raise RuntimeError(
            "ANT versus HAM comparison candidate not found."
        )

    candidate = candidate[0]

    common_laps = [
        int(x)
        for x in candidate["common_laps"].split(",")
        if x.strip()
    ]

    print("\nCOMPARISON DETAILS")
    print("Year:", YEAR)
    print("Event:", candidate["event_name"])
    print("Session:", candidate["session_type"])
    print("Driver A:", DRIVERS[0])
    print("Driver B:", DRIVERS[1])
    print("Common laps:", len(common_laps))

    # --------------------------------------------------
    # 5. READ GOLD AUDIT
    # --------------------------------------------------

    audit = spark.read.parquet(AUDIT_PATH)

    audit_filtered = (
        audit
        .filter(
            (F.col("year") == YEAR) &
            (F.col("event_slug") == EVENT) &
            (F.col("session_type") == "Race") &
            (F.col("driver_code").isin(DRIVERS))
        )
    )

    print("\nAUDIT QUALITY SUMMARY")

    (
        audit_filtered
        .groupBy(
            "driver_code",
            "quality_status",
            "track_map_usable",
            "driving_style_usable"
        )
        .count()
        .orderBy("driver_code")
        .show(30, truncate=False)
    )

    # Only retain laps passing both quality checks.
    valid_laps = (
        audit_filtered
        .filter(
            (F.col("track_map_usable") == True) &
            (F.col("driving_style_usable") == True)
        )
        .select(*KEYS)
        .dropDuplicates(KEYS)
    )

    # --------------------------------------------------
    # 6. READ SILVER TELEMETRY
    # --------------------------------------------------

    silver = spark.read.parquet(SILVER_PATH)

    silver_filtered = (
        silver
        .filter(
            (F.col("year") == YEAR) &
            (F.col("event_slug") == EVENT) &
            (F.col("session_type") == "Race") &
            (F.col("driver_code").isin(DRIVERS)) &
            (F.col("lap_number").isin(common_laps))
        )
    )

    # Join to Gold quality-approved laps.
    telemetry = (
        silver_filtered
        .join(
            F.broadcast(valid_laps),
            on=KEYS,
            how="inner"
        )
        .filter(
            F.col("driving_style_usable") == True
        )
        .select(
            "year",
            "event_name",
            "driver_code",
            "lap_number",
            "sample_index",
            "time_s",
            "distance_m",
            "rel_distance",
            "rel_distance_source",
            "x",
            "y",
            "z",
            "speed_kmh",
            "rpm",
            "gear",
            "throttle_pct",
            "brake",
            "drs",
            "acc_x",
            "acc_y",
            "acc_z"
        )
    )

    # --------------------------------------------------
    # 7. LAP-BY-LAP METRICS
    # --------------------------------------------------

    lap_metrics = (
        telemetry
        .groupBy("driver_code", "lap_number")
        .agg(
            F.count("*").alias("total_samples"),

            F.count("x").alias("valid_x"),
            F.count("y").alias("valid_y"),

            F.countDistinct(
                "sample_index"
            ).alias("unique_samples"),

            F.countDistinct(
                "x", "y"
            ).alias("unique_positions"),

            F.min("distance_m").alias("min_distance"),
            F.max("distance_m").alias("max_distance"),

            F.min("rel_distance").alias("min_rel_distance"),
            F.max("rel_distance").alias("max_rel_distance"),

            F.min("speed_kmh").alias("min_speed"),
            F.max("speed_kmh").alias("max_speed"),
            F.avg("speed_kmh").alias("average_speed"),

            F.avg("throttle_pct").alias("average_throttle"),

            F.sum(
                F.when(F.col("brake") > 0, 1).otherwise(0)
            ).alias("braking_samples"),

            F.min("time_s").alias("first_sample_time"),
            F.max("time_s").alias("last_sample_time"),

            F.collect_set(
                "rel_distance_source"
            ).alias("distance_sources")
        )
        .orderBy("driver_code", "lap_number")
    )

    print("\nLAP-BY-LAP TELEMETRY SUMMARY")

    lap_metrics.show(20, truncate=False)

    # --------------------------------------------------
    # 8. DRIVER SUMMARY
    # --------------------------------------------------

    driver_summary = (
        lap_metrics
        .groupBy("driver_code")
        .agg(
            F.count("*").alias("valid_laps"),

            F.sum("total_samples").alias("total_samples"),

            F.sum("unique_positions").alias(
                "total_unique_positions"
            ),

            F.avg("average_speed").alias(
                "overall_average_speed"
            ),

            F.avg("average_throttle").alias(
                "overall_average_throttle"
            ),

            F.sum("braking_samples").alias(
                "total_braking_samples"
            ),

            F.min("min_distance").alias(
                "minimum_distance"
            ),

            F.max("max_distance").alias(
                "maximum_distance"
            )
        )
        .orderBy("driver_code")
    )

    print("\nDRIVER COMPARISON SUMMARY")

    driver_summary.show(truncate=False)

    # --------------------------------------------------
    # 9. MISSING LAP CHECK
    # --------------------------------------------------

    observed_rows = (
        lap_metrics
        .select("driver_code", "lap_number")
        .collect()
    )

    observed = {driver: set() for driver in DRIVERS}

    for row in observed_rows:
        observed[row["driver_code"]].add(row["lap_number"])

    print("\nMISSING COMMON LAP CHECK")

    for driver in DRIVERS:

        missing = sorted(
            set(common_laps) - observed[driver]
        )

        print(
            driver,
            "| Valid laps:",
            len(observed[driver]),
            "| Missing laps:",
            missing
        )

    # --------------------------------------------------
    # 10. FINAL RESULT
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("FOR1SIGHT PAIR VALIDATION COMPLETED")
    print("=" * 70)

    print("No HDFS datasets were modified.")
    print("No new Parquet datasets were written.")

finally:
    spark.stop()
