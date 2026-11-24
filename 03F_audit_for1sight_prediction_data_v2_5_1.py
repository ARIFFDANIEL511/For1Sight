r"""
For1Sight V2.5.1 — Local Parquet + Race-Label Coverage Audit
READ-ONLY. Corrects the V2.5.0 Windows local-Parquet URI issue.

Run from the For1Sight project root:
    python .\audit_for1sight_prediction_data_v2_5_1.py

Optional log:
    python .\audit_for1sight_prediction_data_v2_5_1.py *> .\prediction_data_audit_v2_5_1.log

This script:
- reads local Parquet through explicit file:/// URIs;
- reads the existing HDFS Silver lap-time dataset;
- reads existing race_results_2024/2025/2026.json caches;
- checks event/driver matching and pre-race practice/qualifying feature coverage;
- does not train models, call APIs, rerun ETL, or write to HDFS.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
WAREHOUSE = PROJECT_ROOT / "for1sight_warehouse"
RACE_RESULT_FILES = [
    WAREHOUSE / "race_results_2024.json",
    WAREHOUSE / "race_results_2025.json",
    WAREHOUSE / "race_results_2026.json",
]
LOCAL_PARQUET = {
    "EXISTING QUALIFYING ML TRAINING TABLE": WAREHOUSE / "ml_training_table.parquet",
    "LOCAL CLEAN LAP TIMES": WAREHOUSE / "hf_laptimes_clean.parquet",
}
HDFS_URI = "hdfs://localhost:9000"
SILVER_LAPS = f"{HDFS_URI}/for1sight_warehouse/silver/hf_laptimes"
SPARK_LOCAL = PROJECT_ROOT / "for1sight_runtime" / "spark_local" / "prediction_audit_v2_5_1"
SPARK_LOCAL.mkdir(parents=True, exist_ok=True)

REPORT = []

def emit(value=""):
    line = str(value)
    REPORT.append(line)
    print(line)

def section(title):
    emit("\n" + "=" * 88)
    emit(title)
    emit("=" * 88)

def normalize_event(value):
    if value is None:
        return ""
    text = str(value).lower().strip()
    text = re.sub(r"\bgrand\s*prix\b", "", text)
    return re.sub(r"[^a-z0-9]", "", text)

def extract_races(obj):
    if isinstance(obj, list):
        return obj
    if not isinstance(obj, dict):
        return []
    mrdata = obj.get("MRData") or obj.get("mrData") or {}
    race_table = mrdata.get("RaceTable") or mrdata.get("raceTable") or {}
    races = race_table.get("Races") or race_table.get("races")
    if isinstance(races, list):
        return races
    for key in ("races", "Races", "results", "Results"):
        if isinstance(obj.get(key), list):
            return obj[key]
    return []

def safe_count_distinct(df, col):
    from pyspark.sql import functions as F
    return df.select(F.countDistinct(F.col(col)).alias("n")).first()["n"]

def inspect_local_parquet(spark, label, path):
    section(label)
    emit(f"Local path: {path}")
    if not path.exists():
        emit("NOT FOUND")
        return None

    # Important: explicit file:/// URI prevents Spark from interpreting a Windows
    # C:\\ path as a path inside the configured HDFS default filesystem.
    local_uri = path.resolve().as_uri()
    emit(f"Spark file URI: {local_uri}")
    try:
        df = spark.read.parquet(local_uri)
        row_count = df.count()
        emit(f"Rows: {row_count:,}")
        emit(f"Columns ({len(df.columns)}): {', '.join(df.columns)}")
        for field in df.schema.fields:
            emit(f"  {field.name}: {field.dataType.simpleString()}")

        from pyspark.sql import functions as F
        lower = {c.lower(): c for c in df.columns}

        for candidate in ("year", "event", "event_name", "driver", "driver_code"):
            actual = lower.get(candidate)
            if actual:
                emit(f"Distinct {actual}: {safe_count_distinct(df, actual)}")

        if "ML TRAINING TABLE" in label:
            expected = ["Year", "Event", "Driver", "Best_FP_Time", "Best_Q_Time", "Pace_Improvement"]
            emit("Qualifying prototype column check:")
            for name in expected:
                actual = lower.get(name.lower())
                emit(f"  {name}: {'FOUND as ' + actual if actual else 'MISSING'}")

            numeric_candidates = [
                lower[x.lower()] for x in
                ["Best_FP_Time", "Best_Q_Time", "Pace_Improvement"]
                if x.lower() in lower
            ]
            if numeric_candidates and row_count:
                expressions = []
                for col in numeric_candidates:
                    expressions.append(F.sum(F.when(F.col(col).isNull(), 1).otherwise(0)).alias(col + "_null"))
                nulls = df.agg(*expressions).first().asDict()
                emit("Null counts for qualifying fields:")
                for col, value in nulls.items():
                    emit(f"  {col}: {int(value or 0):,}")

        emit("Sample rows:")
        for row in df.limit(5).toJSON().collect():
            emit("  " + row)
        return df
    except Exception as exc:
        emit(f"READ ERROR: {type(exc).__name__}: {exc}")
        return None

def load_race_results(spark):
    from pyspark.sql import types as T

    schema = T.StructType([
        T.StructField("year", T.IntegerType(), False),
        T.StructField("round", T.IntegerType(), True),
        T.StructField("event_name", T.StringType(), True),
        T.StructField("event_norm", T.StringType(), True),
        T.StructField("event_date", T.StringType(), True),
        T.StructField("driver_code", T.StringType(), True),
        T.StructField("constructor", T.StringType(), True),
        T.StructField("position", T.StringType(), True),
        T.StructField("grid", T.StringType(), True),
        T.StructField("status", T.StringType(), True),
        T.StructField("points", T.StringType(), True),
    ])
    rows = []
    section("EXISTING OFFICIAL RACE-RESULT JSON CACHES")
    for path in RACE_RESULT_FILES:
        if not path.exists():
            emit(f"NOT FOUND: {path.name}")
            continue
        year_match = re.search(r"(20\d{2})", path.name)
        year = int(year_match.group(1)) if year_match else None
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        races = extract_races(obj)
        file_rows = 0
        for race in races:
            event_name = race.get("raceName") or race.get("event_name") or race.get("name")
            event_norm = normalize_event(event_name)
            try:
                round_no = int(race.get("round")) if race.get("round") is not None else None
            except (TypeError, ValueError):
                round_no = None
            results = race.get("Results") or race.get("results") or []
            for result in results:
                driver = result.get("Driver") or result.get("driver") or {}
                constructor = result.get("Constructor") or result.get("constructor") or {}
                code = driver.get("code") or driver.get("driverId") or driver.get("familyName")
                position = result.get("positionOrder")
                if position in (None, ""):
                    position = result.get("position")
                rows.append((
                    year, round_no, event_name, event_norm, race.get("date"),
                    str(code).upper() if code else None,
                    constructor.get("name") or constructor.get("constructorId"),
                    str(position) if position is not None else None,
                    str(result.get("grid")) if result.get("grid") is not None else None,
                    result.get("status"),
                    str(result.get("points")) if result.get("points") is not None else None,
                ))
                file_rows += 1
        emit(f"{path.name}: {len(races)} events, {file_rows} driver-result rows")
    if not rows:
        emit("No race-result rows loaded.")
        return None
    return spark.createDataFrame(rows, schema=schema)

def main():
    section("FOR1SIGHT V2.5.1 — PREDICTION DATA AUDIT FIX")
    emit(f"Audit time: {datetime.now().isoformat(timespec='seconds')}")
    emit(f"Project root: {PROJECT_ROOT}")
    emit("Mode: READ-ONLY. No HDFS writes, no ETL rerun, no model training, no API requests.")

    try:
        from pyspark.sql import SparkSession, functions as F
    except Exception as exc:
        emit(f"PySpark import failed: {type(exc).__name__}: {exc}")
        return

    spark = None
    try:
        spark = (
            SparkSession.builder
            .appName("For1Sight_V2_5_1_Prediction_Audit")
            .master("local[2]")
            .config("spark.driver.memory", "3g")
            .config("spark.driver.maxResultSize", "1g")
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.local.dir", str(SPARK_LOCAL))
            .config("spark.hadoop.fs.defaultFS", HDFS_URI)
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")
        emit(f"Spark version: {spark.version}")
        emit(f"HDFS default: {HDFS_URI}")

        local_dfs = {}
        for label, path in LOCAL_PARQUET.items():
            local_dfs[label] = inspect_local_parquet(spark, label, path)

        section("HDFS SILVER LAP-TIME EVENT COVERAGE")
        laps = spark.read.parquet(SILVER_LAPS)
        emit(f"Silver lap rows: {laps.count():,}")
        emit("Distinct events by season:")
        event_counts = (
            laps.groupBy("year")
            .agg(F.countDistinct("event_name").alias("events"),
                 F.countDistinct("driver_code").alias("drivers"),
                 F.count("*").alias("lap_rows"))
            .orderBy("year")
        )
        for row in event_counts.collect():
            emit(f"  {row['year']}: {row['events']} events, {row['drivers']} drivers, {row['lap_rows']:,} rows")

        emit("Event counts by season and session:")
        session_counts = (
            laps.groupBy("year", "session_type")
            .agg(F.countDistinct("event_name").alias("events"),
                 F.count("*").alias("rows"))
            .orderBy("year", "session_type")
        )
        for row in session_counts.collect():
            emit(f"  {row['year']} | {row['session_type']}: {row['events']} events, {row['rows']:,} rows")

        section("PRE-RACE FEATURE AGGREGATION — IN MEMORY ONLY")
        # Only practice and qualifying are used as candidate pre-race features.
        # Race-session laps and telemetry are deliberately excluded to prevent leakage.
        base = (
            laps.withColumn("event_norm", F.regexp_replace(
                F.regexp_replace(F.lower(F.trim(F.col("event_name"))), r"\bgrand\s*prix\b", ""),
                r"[^a-z0-9]", ""
            ))
            .withColumn("driver_code_norm", F.upper(F.trim(F.col("driver_code"))))
            .withColumn("session_norm", F.lower(F.trim(F.col("session_type"))))
        )
        practice_types = ["practice 1", "practice 2", "practice 3"]
        feature_rows = (
            base.groupBy("year", "event_name", "event_norm", "driver_code_norm")
            .agg(
                F.min(F.when(F.col("session_norm").isin(practice_types), F.col("time"))).alias("best_practice_lap_s"),
                F.min(F.when(F.col("session_norm") == "qualifying", F.col("time"))).alias("best_qualifying_lap_s"),
                F.min(F.when(F.col("session_norm") == "qualifying", F.col("s1"))).alias("best_quali_s1_s"),
                F.min(F.when(F.col("session_norm") == "qualifying", F.col("s2"))).alias("best_quali_s2_s"),
                F.min(F.when(F.col("session_norm") == "qualifying", F.col("s3"))).alias("best_quali_s3_s"),
                F.sum(F.when(F.col("session_norm").isin(practice_types) & F.col("time").isNotNull(), 1).otherwise(0)).alias("practice_timed_laps"),
                F.sum(F.when((F.col("session_norm") == "qualifying") & F.col("time").isNotNull(), 1).otherwise(0)).alias("qualifying_timed_laps"),
            )
        )
        emit(f"Driver-event feature rows: {feature_rows.count():,}")
        emit("Feature columns: " + ", ".join(feature_rows.columns))

        results = load_race_results(spark)
        if results is not None:
            section("RACE-RESULT ↔ PRACTICE/QUALIFYING JOIN COVERAGE")
            results = (
                results.withColumn("driver_code_norm", F.upper(F.trim(F.col("driver_code"))))
                .withColumn("position_num", F.col("position").cast("int"))
                .withColumn("grid_num", F.col("grid").cast("int"))
            )
            key = ["year", "event_norm", "driver_code_norm"]
            duplicate_result_keys = (
                results.groupBy(*key).count().filter(F.col("count") > 1).count()
            )
            emit(f"Duplicate race-result driver/event keys: {duplicate_result_keys:,}")

            joined = results.join(
                feature_rows.select(
                    "year", "event_norm", "driver_code_norm",
                    "best_practice_lap_s", "best_qualifying_lap_s",
                    "best_quali_s1_s", "best_quali_s2_s", "best_quali_s3_s",
                    "practice_timed_laps", "qualifying_timed_laps"
                ),
                on=key,
                how="left"
            )
            total = joined.count()
            matched = joined.filter(F.col("best_practice_lap_s").isNotNull() | F.col("best_qualifying_lap_s").isNotNull()).count()
            practice = joined.filter(F.col("best_practice_lap_s").isNotNull()).count()
            qualifying = joined.filter(F.col("best_qualifying_lap_s").isNotNull()).count()
            numeric_position = joined.filter(F.col("position_num").isNotNull()).count()
            winners = joined.filter(F.col("position_num") == 1).count()
            emit(f"Race-result driver rows: {total:,}")
            emit(f"Rows with any practice/qualifying feature: {matched:,} ({matched/total*100:.1f}%)" if total else "No result rows")
            emit(f"Rows with practice best lap: {practice:,} ({practice/total*100:.1f}%)" if total else "")
            emit(f"Rows with qualifying best lap: {qualifying:,} ({qualifying/total*100:.1f}%)" if total else "")
            emit(f"Rows with numeric finishing position: {numeric_position:,}")
            emit(f"Winner-labelled rows (position = 1): {winners:,}")

            emit("Join coverage by season:")
            by_year = (
                joined.groupBy("year")
                .agg(
                    F.count("*").alias("result_rows"),
                    F.sum(F.when(F.col("best_practice_lap_s").isNotNull(), 1).otherwise(0)).alias("practice_rows"),
                    F.sum(F.when(F.col("best_qualifying_lap_s").isNotNull(), 1).otherwise(0)).alias("quali_rows"),
                ).orderBy("year")
            )
            for row in by_year.collect():
                emit(f"  {row['year']}: results={row['result_rows']:,}, practice={row['practice_rows']:,}, qualifying={row['quali_rows']:,}")

            emit("Unmatched race-result event names by season:")
            lap_events = base.select("year", "event_norm", "event_name").distinct()
            result_events = results.select("year", "event_norm", "event_name").distinct()
            unmatched = result_events.join(
                lap_events.select("year", "event_norm").distinct(),
                on=["year", "event_norm"], how="left_anti"
            ).orderBy("year", "event_name")
            unmatched_rows = unmatched.collect()
            if unmatched_rows:
                for row in unmatched_rows:
                    emit(f"  {row['year']}: {row['event_name']}")
            else:
                emit("  NONE — all result event names have a normalized event match in Silver lap data.")

            emit("Sample joined rows:")
            sample_cols = [
                "year", "event_name", "driver_code", "constructor",
                "position", "grid", "best_practice_lap_s",
                "best_qualifying_lap_s", "practice_timed_laps", "qualifying_timed_laps"
            ]
            for row in joined.select(*sample_cols).limit(10).toJSON().collect():
                emit("  " + row)

        section("INTERPRETATION")
        emit("1. The V2.5.0 local Parquet read failure was caused by passing a Windows C:\\\\ path")
        emit("   while Spark's default filesystem was HDFS. V2.5.1 reads local files using file:/// URIs.")
        emit("2. Practice and qualifying aggregates are candidate pre-race features.")
        emit("3. Race-session lap times and race telemetry are excluded from this pre-race join to avoid target leakage.")
        emit("4. The join coverage report determines whether driver + season + normalized event is a reliable key.")
        emit("5. No model-ready Parquet dataset is written yet. We review coverage and missingness first.")

    except Exception as exc:
        section("AUDIT ERROR")
        emit(f"{type(exc).__name__}: {exc}")
        import traceback
        emit(traceback.format_exc())
    finally:
        if spark is not None:
            try:
                spark.stop()
                emit("Spark session stopped.")
            except Exception:
                pass

    report_path = WAREHOUSE / "prediction_data_audit_v2_5_1.txt"
    try:
        WAREHOUSE.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(REPORT) + "\n", encoding="utf-8")
        print(f"\nReport saved to: {report_path}")
    except Exception as exc:
        print(f"\nCould not save report: {exc}")

if __name__ == "__main__":
    main()
