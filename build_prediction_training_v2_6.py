# -*- coding: utf-8 -*-
"""For1Sight V2.6 - build the post-qualifying race-winner training dataset.

This script creates a leakage-aware driver/event dataset from:
  1) HDFS Silver practice and qualifying lap times; and
  2) the existing official race-result JSON caches (labels + starting grid).

It does NOT train a model, call an API, rerun ETL, read Race-session telemetry,
modify existing dashboard pages, or overwrite existing warehouse outputs.

Run from the For1Sight project root:
    python build_prediction_training_v2_6.py

The script writes a versioned Gold Parquet dataset to HDFS and a local Parquet
copy for the later Streamlit Prediction Analysis integration.
"""
from __future__ import annotations

import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

# -----------------------------------------------------------------------------
# Configuration - aligned with the V2.5.2 audit
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
WAREHOUSE = ROOT / "for1sight_warehouse"
RUNTIME = ROOT / "for1sight_runtime" / "spark_local" / "prediction_training_v2_6"
HDFS_URI = "hdfs://localhost:9000"
HDFS_SILVER_LAPS = HDFS_URI + "/for1sight_warehouse/silver/hf_laptimes"
HDFS_GOLD_OUTPUT = HDFS_URI + "/for1sight_warehouse/gold/prediction/race_winner_training_v2_6"
HDFS_STAGING_ROOT = HDFS_URI + "/for1sight_warehouse/_staging/prediction_v2_6"
RACE_RESULT_FILES = [
    WAREHOUSE / "race_results_2024.json",
    WAREHOUSE / "race_results_2025.json",
    WAREHOUSE / "race_results_2026.json",
]
LOCAL_OUTPUT = WAREHOUSE / "prediction_training_v2_6.parquet"
MIN_EVENT_DRIVERS = 10
MIN_QUALIFYING_COVERAGE = 0.70
MIN_GRID_COVERAGE = 0.70
SPARK_DRIVER_MEMORY = "3g"
SPARK_SHUFFLE_PARTITIONS = "4"

# Deliberately no overwrite switch. Use a new output version after reviewing
# and backing up any existing output; this script never deletes prior data.

REPORT: list[str] = []


def emit(message: Any = "") -> None:
    line = str(message)
    REPORT.append(line)
    try:
        print(line)
    except UnicodeEncodeError:
        print(line.encode("ascii", errors="replace").decode("ascii"))


def section(title: str) -> None:
    emit("\n" + "=" * 88)
    emit(title)
    emit("=" * 88)


def normalize_event(value: Any) -> str:
    value = str(value or "").lower().strip()
    value = re.sub(r"\bgrand\s*prix\b", "", value)
    return re.sub(r"[^a-z0-9]", "", value)


def extract_races(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if not isinstance(obj, dict):
        return []
    mr = obj.get("MRData") or obj.get("mrData") or {}
    rt = mr.get("RaceTable") or mr.get("raceTable") or {}
    races = rt.get("Races") or rt.get("races")
    if isinstance(races, list):
        return [x for x in races if isinstance(x, dict)]
    for key in ("races", "Races", "results", "Results"):
        if isinstance(obj.get(key), list):
            return [x for x in obj[key] if isinstance(x, dict)]
    return []


def as_int(value: Any) -> int | None:
    try:
        if value in (None, "", "None", "null"):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def load_official_results() -> tuple[list[tuple[Any, ...]], int]:
    """Load official result labels and grid positions from local JSON caches."""
    rows: list[tuple[Any, ...]] = []
    missing_driver_codes = 0
    for path in RACE_RESULT_FILES:
        if not path.exists():
            raise FileNotFoundError(
                f"Required official result cache not found: {path}. "
                "Expected race_results_2024.json, race_results_2025.json and race_results_2026.json."
            )
        year_match = re.search(r"(20\d{2})", path.name)
        if not year_match:
            raise ValueError(f"Cannot determine season from result cache name: {path.name}")
        year = int(year_match.group(1))
        with path.open("r", encoding="utf-8-sig") as handle:
            obj = json.load(handle)
        races = extract_races(obj)
        if not races:
            raise ValueError(f"No race entries found in {path.name}; expected the existing race-result JSON structure.")

        event_rows = 0
        for race in races:
            event_name = race.get("raceName") or race.get("event_name") or race.get("name")
            if not event_name:
                continue
            round_no = as_int(race.get("round"))
            event_date = race.get("date")
            race_results = race.get("Results") or race.get("results") or []
            for result in race_results:
                driver = result.get("Driver") or result.get("driver") or {}
                constructor = result.get("Constructor") or result.get("constructor") or {}
                driver_code = (
                    driver.get("code")
                    or result.get("driver_code")
                    or driver.get("driverId")
                    or driver.get("familyName")
                )
                if not driver_code:
                    missing_driver_codes += 1
                    continue
                position_value = result.get("positionOrder")
                if position_value in (None, ""):
                    position_value = result.get("position")
                rows.append((
                    year,
                    round_no,
                    str(event_name),
                    normalize_event(event_name),
                    str(event_date) if event_date else None,
                    str(driver_code).upper().strip(),
                    str(constructor.get("name") or constructor.get("constructorId") or "Unknown"),
                    as_int(position_value),
                    as_int(result.get("grid")),
                    as_int(result.get("points")),
                    str(result.get("status") or ""),
                ))
                event_rows += 1
        emit(f"{path.name}: {len(races)} events, {event_rows:,} driver-result rows")
    return rows, missing_driver_codes


def hdfs_path_exists(spark: Any, uri: str) -> bool:
    jpath = spark._jvm.org.apache.hadoop.fs.Path(uri)
    fs = jpath.getFileSystem(spark._jsc.hadoopConfiguration())
    return bool(fs.exists(jpath))


def build_dataset(spark: Any, result_rows: list[tuple[Any, ...]]):
    from pyspark.sql import functions as F
    from pyspark.sql import types as T
    from pyspark.sql.window import Window

    result_schema = T.StructType([
        T.StructField("year", T.IntegerType(), False),
        T.StructField("round", T.IntegerType(), True),
        T.StructField("event_name", T.StringType(), False),
        T.StructField("event_norm", T.StringType(), False),
        T.StructField("event_date", T.StringType(), True),
        T.StructField("driver_code", T.StringType(), False),
        T.StructField("constructor", T.StringType(), True),
        T.StructField("finishing_position", T.IntegerType(), True),
        T.StructField("starting_grid", T.IntegerType(), True),
        T.StructField("race_points", T.IntegerType(), True),
        T.StructField("race_status", T.StringType(), True),
    ])
    results = spark.createDataFrame(result_rows, schema=result_schema)
    results = (
        results
        .withColumn("driver_code_norm", F.upper(F.trim(F.col("driver_code"))))
        .withColumn("constructor", F.coalesce(F.col("constructor"), F.lit("Unknown")))
        .withColumn("starting_grid", F.when(F.col("starting_grid") > 0, F.col("starting_grid")))
        .withColumn("is_winner", F.when(F.col("finishing_position") == 1, F.lit(1)).when(F.col("finishing_position").isNotNull(), F.lit(0)))
    )

    # Check official result uniqueness before building features.
    result_key_cols = ["year", "event_norm", "driver_code_norm"]
    duplicate_results = results.groupBy(*result_key_cols).count().filter(F.col("count") > 1).count()
    if duplicate_results:
        raise RuntimeError(f"Official result caches contain {duplicate_results:,} duplicate season/event/driver keys. Resolve duplicates before training.")

    section("PRE-RACE PRACTICE AND QUALIFYING FEATURE AGGREGATION")
    laps = spark.read.parquet(HDFS_SILVER_LAPS)
    required = {"year", "event_name", "session_type", "driver_code", "time", "s1", "s2", "s3"}
    missing = sorted(required - set(laps.columns))
    if missing:
        raise RuntimeError("HDFS Silver lap-time schema is missing required columns: " + ", ".join(missing))

    # Only practice and standard qualifying are read. Race-session laps are
    # intentionally excluded from every feature calculation.
    session = F.lower(F.trim(F.col("session_type")))
    lap_time = F.col("time").cast("double")
    deleted = F.lower(F.trim(F.coalesce(F.col("del").cast("string"), F.lit("false")))) if "del" in laps.columns else F.lit("false")
    valid = (lap_time > 20.0) & (lap_time < 300.0) & (~deleted.isin("true", "1", "yes"))
    practice_types = ["practice 1", "practice 2", "practice 3"]
    selected = (
        laps
        .withColumn("session_norm", session)
        .withColumn("lap_time_s", lap_time)
        .withColumn("s1_num", F.col("s1").cast("double"))
        .withColumn("s2_num", F.col("s2").cast("double"))
        .withColumn("s3_num", F.col("s3").cast("double"))
        .withColumn("event_norm", F.regexp_replace(
            F.regexp_replace(F.lower(F.trim(F.col("event_name"))), r"\bgrand\s*prix\b", ""),
            r"[^a-z0-9]", ""
        ))
        .withColumn("driver_code_norm", F.upper(F.trim(F.col("driver_code"))))
        .filter(F.col("session_norm").isin(*(practice_types + ["qualifying"])))
        .filter(valid & F.col("driver_code_norm").isNotNull() & (F.length("driver_code_norm") > 0))
    )

    features = (
        selected.groupBy("year", "event_norm", "driver_code_norm")
        .agg(
            F.first("event_name", ignorenulls=True).alias("silver_event_name"),
            F.min(F.when(F.col("session_norm").isin(practice_types), F.col("lap_time_s"))).alias("best_practice_lap_s"),
            F.min(F.when(F.col("session_norm") == "qualifying", F.col("lap_time_s"))).alias("best_qualifying_lap_s"),
            F.min(F.when(F.col("session_norm") == "qualifying", F.col("s1_num"))).alias("best_quali_s1_s"),
            F.min(F.when(F.col("session_norm") == "qualifying", F.col("s2_num"))).alias("best_quali_s2_s"),
            F.min(F.when(F.col("session_norm") == "qualifying", F.col("s3_num"))).alias("best_quali_s3_s"),
            F.sum(F.when(F.col("session_norm").isin(practice_types), 1).otherwise(0)).alias("practice_timed_laps"),
            F.sum(F.when(F.col("session_norm") == "qualifying", 1).otherwise(0)).alias("qualifying_timed_laps"),
        )
    )
    emit(f"Driver-event feature rows: {features.count():,}")

    section("LEAKAGE-SAFE HISTORICAL FORM FEATURES")
    # Every window ends at -1, so the current event's race result is never
    # included in its own historical features.
    driver_order = Window.partitionBy("driver_code_norm").orderBy("year", "round")
    driver_last3 = driver_order.rowsBetween(-3, -1)
    driver_circuit_prior = (
        Window.partitionBy("driver_code_norm", "event_norm")
        .orderBy("year", "round")
        .rowsBetween(Window.unboundedPreceding, -1)
    )
    results = (
        results
        .withColumn("driver_prior_avg_finish_3", F.avg("finishing_position").over(driver_last3))
        .withColumn("driver_prior_avg_points_3", F.avg("race_points").over(driver_last3))
        .withColumn("driver_circuit_prior_avg_finish", F.avg("finishing_position").over(driver_circuit_prior))
        .withColumn("driver_circuit_prior_starts", F.count("finishing_position").over(driver_circuit_prior))
    )

    constructor_events = (
        results.groupBy("year", "round", "event_norm", "constructor")
        .agg(
            F.avg("finishing_position").alias("constructor_event_avg_finish"),
            F.avg("race_points").alias("constructor_event_avg_points"),
        )
    )
    constructor_order = Window.partitionBy("constructor").orderBy("year", "round")
    constructor_last3 = constructor_order.rowsBetween(-3, -1)
    constructor_circuit_prior = (
        Window.partitionBy("constructor", "event_norm")
        .orderBy("year", "round")
        .rowsBetween(Window.unboundedPreceding, -1)
    )
    constructor_events = (
        constructor_events
        .withColumn("constructor_prior_avg_finish_3", F.avg("constructor_event_avg_finish").over(constructor_last3))
        .withColumn("constructor_prior_avg_points_3", F.avg("constructor_event_avg_points").over(constructor_last3))
        .withColumn("constructor_circuit_prior_avg_finish", F.avg("constructor_event_avg_finish").over(constructor_circuit_prior))
        .withColumn("constructor_circuit_prior_starts", F.count("constructor_event_avg_finish").over(constructor_circuit_prior))
    )
    results = results.join(
        constructor_events.select(
            "year", "event_norm", "constructor",
            "constructor_prior_avg_finish_3", "constructor_prior_avg_points_3",
            "constructor_circuit_prior_avg_finish", "constructor_circuit_prior_starts",
        ),
        on=["year", "event_norm", "constructor"], how="left"
    )

    section("JOIN OFFICIAL RESULTS TO PRE-RACE FEATURES")
    joined = results.join(features, on=["year", "event_norm", "driver_code_norm"], how="left")
    event_window = Window.partitionBy("year", "event_norm")
    quali_rank_window = Window.partitionBy("year", "event_norm").orderBy(F.col("best_qualifying_lap_s").asc_nulls_last())
    joined = (
        joined
        .withColumn("event_best_practice_lap_s", F.min("best_practice_lap_s").over(event_window))
        .withColumn("event_best_qualifying_lap_s", F.min("best_qualifying_lap_s").over(event_window))
        .withColumn("practice_delta_to_event_best_s", F.col("best_practice_lap_s") - F.col("event_best_practice_lap_s"))
        .withColumn("qualifying_delta_to_event_best_s", F.col("best_qualifying_lap_s") - F.col("event_best_qualifying_lap_s"))
        .withColumn("theoretical_best_sector_sum_s", F.col("best_quali_s1_s") + F.col("best_quali_s2_s") + F.col("best_quali_s3_s"))
        .withColumn("practice_to_qualifying_improvement_s", F.col("best_practice_lap_s") - F.col("best_qualifying_lap_s"))
        .withColumn("qualifying_rank", F.when(F.col("best_qualifying_lap_s").isNotNull(), F.dense_rank().over(quali_rank_window)))
        .withColumn("has_practice_data", F.col("best_practice_lap_s").isNotNull())
        .withColumn("has_qualifying_data", F.col("best_qualifying_lap_s").isNotNull())
        .withColumn("has_grid_data", F.col("starting_grid").isNotNull())
        .withColumn(
            "feature_match_status",
            F.when(F.col("best_practice_lap_s").isNotNull() & F.col("best_qualifying_lap_s").isNotNull(), "Practice + Qualifying")
             .when(F.col("best_practice_lap_s").isNotNull(), "Practice only")
             .when(F.col("best_qualifying_lap_s").isNotNull(), "Qualifying only")
             .otherwise("No matching lap-time features")
        )
    )

    event_quality = (
        joined.groupBy("year", "event_norm")
        .agg(
            F.first("event_name", ignorenulls=True).alias("event_name"),
            F.count("*").alias("event_driver_count"),
            F.sum(F.when(F.col("best_qualifying_lap_s").isNotNull(), 1).otherwise(0)).alias("qualifying_feature_count"),
            F.sum(F.when(F.col("starting_grid").isNotNull(), 1).otherwise(0)).alias("grid_feature_count"),
            F.sum(F.when(F.col("is_winner") == 1, 1).otherwise(0)).alias("winner_label_count"),
        )
        .withColumn("qualifying_coverage", F.col("qualifying_feature_count") / F.col("event_driver_count"))
        .withColumn("grid_coverage", F.col("grid_feature_count") / F.col("event_driver_count"))
        .withColumn(
            "event_prediction_ready",
            (F.col("event_driver_count") >= F.lit(MIN_EVENT_DRIVERS))
            & (F.col("qualifying_coverage") >= F.lit(MIN_QUALIFYING_COVERAGE))
            & (F.col("grid_coverage") >= F.lit(MIN_GRID_COVERAGE))
            & (F.col("winner_label_count") == 1)
        )
    )

    final = (
        joined.join(
            event_quality.select(
                "year", "event_norm", "event_driver_count", "qualifying_feature_count",
                "grid_feature_count", "winner_label_count", "qualifying_coverage",
                "grid_coverage", "event_prediction_ready",
            ),
            on=["year", "event_norm"], how="left"
        )
        .withColumn("model_training_eligible", F.col("event_prediction_ready") & F.col("finishing_position").isNotNull())
        .withColumn("dataset_version", F.lit("V2.6"))
        .select(
            "dataset_version", "year", "round", "event_date", "event_name", "silver_event_name", "event_norm",
            "driver_code", "driver_code_norm", "constructor", "starting_grid",
            "best_practice_lap_s", "best_qualifying_lap_s", "best_quali_s1_s", "best_quali_s2_s", "best_quali_s3_s",
            "theoretical_best_sector_sum_s", "practice_to_qualifying_improvement_s",
            "practice_delta_to_event_best_s", "qualifying_delta_to_event_best_s", "event_best_practice_lap_s",
            "event_best_qualifying_lap_s", "qualifying_rank", "practice_timed_laps", "qualifying_timed_laps",
            "driver_prior_avg_finish_3", "driver_prior_avg_points_3", "driver_circuit_prior_avg_finish",
            "driver_circuit_prior_starts", "constructor_prior_avg_finish_3", "constructor_prior_avg_points_3",
            "constructor_circuit_prior_avg_finish", "constructor_circuit_prior_starts",
            "has_practice_data", "has_qualifying_data", "has_grid_data", "feature_match_status",
            "event_driver_count", "qualifying_feature_count", "grid_feature_count", "winner_label_count",
            "qualifying_coverage", "grid_coverage", "event_prediction_ready", "model_training_eligible",
            "finishing_position", "is_winner", "race_points", "race_status",
        )
        .orderBy("year", "round", "event_name", "starting_grid", "driver_code")
    )

    return final, event_quality, results, features


def main() -> None:
    section("FOR1SIGHT V2.6 - POST-QUALIFYING RACE-WINNER DATASET BUILDER")
    emit(f"Run time: {datetime.now().isoformat(timespec='seconds')}")
    emit(f"Project root: {ROOT}")
    emit(f"HDFS Silver source: {HDFS_SILVER_LAPS}")
    emit(f"HDFS Gold destination: {HDFS_GOLD_OUTPUT}")
    emit(f"Local Parquet destination: {LOCAL_OUTPUT}")
    emit("Mode: build-only. No model training, API calls, ETL rerun, Race telemetry features or dashboard edits.")

    spark = None
    staging_uri = None
    local_tmp = None
    success = False
    try:
        from pyspark.sql import SparkSession, functions as F

        WAREHOUSE.mkdir(parents=True, exist_ok=True)
        RUNTIME.mkdir(parents=True, exist_ok=True)
        if LOCAL_OUTPUT.exists():
            raise FileExistsError(
                f"Local output already exists: {LOCAL_OUTPUT}. It was not overwritten. "
                "Back it up and change LOCAL_OUTPUT to a new versioned filename before rerunning."
            )

        result_rows, missing_driver_codes = load_official_results()
        if not result_rows:
            raise RuntimeError("No official driver-result rows were loaded from the race-result JSON caches.")
        emit(f"Official result rows loaded: {len(result_rows):,}")
        emit(f"Result rows skipped for missing driver code: {missing_driver_codes:,}")

        spark = (
            SparkSession.builder
            .appName("For1Sight_V2_6_Prediction_Training_Dataset")
            .master("local[2]")
            .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
            .config("spark.driver.maxResultSize", "1g")
            .config("spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTITIONS)
            .config("spark.sql.parquet.compression.codec", "snappy")
            .config("spark.local.dir", str(RUNTIME))
            .config("spark.hadoop.fs.defaultFS", HDFS_URI)
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")
        emit(f"Spark version: {spark.version}")

        if hdfs_path_exists(spark, HDFS_GOLD_OUTPUT):
            raise FileExistsError(
                f"HDFS Gold output already exists: {HDFS_GOLD_OUTPUT}. It was not overwritten. "
                "Review/backup it and change HDFS_GOLD_OUTPUT to a new versioned path before rerunning."
            )

        final, event_quality, results, features = build_dataset(spark, result_rows)
        total_rows = final.count()
        eligible_rows = final.filter(F.col("model_training_eligible")).count()
        event_count = event_quality.count()
        ready_event_count = event_quality.filter(F.col("event_prediction_ready")).count()
        duplicate_final_keys = (
            final.groupBy("year", "event_norm", "driver_code_norm").count().filter(F.col("count") > 1).count()
        )
        if duplicate_final_keys:
            raise RuntimeError(f"Final dataset contains {duplicate_final_keys:,} duplicate driver/event keys.")

        winner_integrity = (
            event_quality.groupBy("year", "event_norm")
            .agg(F.max("winner_label_count").alias("winner_label_count"))
            .filter(F.col("winner_label_count") != 1)
            .count()
        )
        no_quali_rows = final.filter(F.col("best_qualifying_lap_s").isNull()).count()
        no_both_features = final.filter(
            F.col("best_practice_lap_s").isNull() & F.col("best_qualifying_lap_s").isNull()
        ).count()

        section("DATASET QUALITY SUMMARY")
        emit(f"Final driver-event rows: {total_rows:,}")
        emit(f"Unique Grand Prix events: {event_count:,}")
        emit(f"Events meeting prediction-readiness rules: {ready_event_count:,}")
        emit(f"Rows eligible for model training: {eligible_rows:,}")
        emit(f"Rows missing qualifying lap features: {no_quali_rows:,}")
        emit(f"Rows missing both practice and qualifying features: {no_both_features:,}")
        emit(f"Duplicate final driver/event keys: {duplicate_final_keys:,}")
        emit(f"Events without exactly one winner label: {winner_integrity:,}")

        section("EVENT COVERAGE AND MATCH REVIEW")
        silver_events = features.select("year", "event_norm").distinct()
        result_events = results.select("year", "event_norm", "event_name").distinct()
        unmatched = result_events.join(silver_events, on=["year", "event_norm"], how="left_anti")
        unmatched_rows = unmatched.orderBy("year", "event_name").collect()
        if unmatched_rows:
            emit("Official result events without matching Silver practice/qualifying event keys:")
            for row in unmatched_rows:
                emit(f"  UNMATCHED: {row['year']} | {row['event_name']}")
        else:
            emit("All official result events have a matching Silver event key.")
        emit("No event aliases were guessed or automatically applied.")

        emit("\nPer-event readiness:")
        for row in event_quality.orderBy("year", "event_name").select(
            "year", "event_name", "event_driver_count", "qualifying_coverage", "grid_coverage",
            "winner_label_count", "event_prediction_ready"
        ).collect():
            emit(
                f"  {row['year']} | {row['event_name']} | drivers={row['event_driver_count']} | "
                f"quali={row['qualifying_coverage']:.1%} | grid={row['grid_coverage']:.1%} | "
                f"winner_labels={row['winner_label_count']} | ready={row['event_prediction_ready']}"
            )

        if total_rows != len(result_rows):
            raise RuntimeError(
                f"Row-count validation failed: expected {len(result_rows):,} official result rows, got {total_rows:,}."
            )
        if winner_integrity:
            emit("WARNING: Some events do not have exactly one winner label. They are marked not prediction-ready.")

        # Prepare the small local Parquet mirror before publishing anything to
        # Gold. This catches local conversion/dependency errors before the HDFS
        # final path is created.
        local_tmp = LOCAL_OUTPUT.with_name(LOCAL_OUTPUT.stem + ".tmp" + LOCAL_OUTPUT.suffix)
        if local_tmp.exists():
            raise FileExistsError(f"Temporary local output already exists; inspect before removing: {local_tmp}")
        local_pdf = final.toPandas()
        local_pdf.to_parquet(local_tmp, index=False, engine="pyarrow", compression="snappy")

        # Write to a unique HDFS staging directory first. Only publish the final
        # Gold directory after reading the staged Parquet back and validating it.
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        staging_uri = HDFS_STAGING_ROOT + "/run_" + stamp
        if hdfs_path_exists(spark, staging_uri):
            raise FileExistsError(f"HDFS staging path already exists: {staging_uri}")
        emit(f"\nWriting HDFS staging dataset: {staging_uri}")
        final.write.mode("errorifexists").option("compression", "snappy").partitionBy("year").parquet(staging_uri)
        readback = spark.read.parquet(staging_uri)
        readback_rows = readback.count()
        if readback_rows != total_rows:
            raise RuntimeError(f"HDFS read-back row count mismatch: wrote {total_rows:,}, read {readback_rows:,}.")

        final_path = spark._jvm.org.apache.hadoop.fs.Path(HDFS_GOLD_OUTPUT)
        stage_path = spark._jvm.org.apache.hadoop.fs.Path(staging_uri)
        fs = stage_path.getFileSystem(spark._jsc.hadoopConfiguration())
        parent = final_path.getParent()
        if not fs.exists(parent) and not fs.mkdirs(parent):
            raise RuntimeError(f"Could not create HDFS Gold parent directory: {parent}")
        if fs.exists(final_path):
            raise FileExistsError(f"HDFS destination appeared during the run; refusing to overwrite: {HDFS_GOLD_OUTPUT}")
        if not fs.rename(stage_path, final_path):
            raise RuntimeError(f"HDFS staging-to-Gold rename failed. Staging data preserved at: {staging_uri}")
        emit(f"Published HDFS Gold dataset: {HDFS_GOLD_OUTPUT}")

        # Publish the already-validated local Parquet mirror for Streamlit
        # integration in the next phase.
        if LOCAL_OUTPUT.exists():
            raise FileExistsError(f"Local output appeared during the run; refusing to overwrite: {LOCAL_OUTPUT}")
        local_tmp.replace(LOCAL_OUTPUT)
        emit(f"Saved local Parquet mirror: {LOCAL_OUTPUT}")
        emit(f"Local Parquet rows: {len(local_pdf):,}")
        success = True

    except Exception as exc:
        if local_tmp is not None and local_tmp.exists():
            try:
                local_tmp.unlink()
            except OSError:
                pass
        section("V2.6 BUILD ERROR")
        emit(f"{type(exc).__name__}: {exc}")
        emit(traceback.format_exc())
        if staging_uri:
            emit(f"Any HDFS staging output has been preserved for inspection: {staging_uri}")
    finally:
        if spark is not None:
            try:
                spark.stop()
                emit("Spark session stopped.")
            except Exception as exc:
                emit(f"Spark stop warning: {type(exc).__name__}: {exc}")

        report_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = WAREHOUSE / f"prediction_training_v2_6_audit_{report_stamp}.txt"
        try:
            WAREHOUSE.mkdir(parents=True, exist_ok=True)
            REPORT.append("\nRESULT: " + ("SUCCESS" if success else "FAILED"))
            report_path.write_text("\n".join(REPORT) + "\n", encoding="utf-8")
            print(f"\nAudit report saved to: {report_path}")
        except Exception as exc:
            print(f"\nCould not save audit report: {type(exc).__name__}: {exc}")

    if not success:
        raise SystemExit(1)
    print("\nFOR1SIGHT V2.6 DATASET BUILD COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
