"""
For1Sight V2.5.0 — Prediction Data Audit
READ-ONLY audit of existing local Parquet, HDFS Parquet, race-result JSON caches,
and existing prediction artifacts. Does not modify HDFS or retrain any model.

Run from the For1Sight project root:
    python .\audit_for1sight_prediction_data_v2_5_0.py

Optional log capture:
    python .\\audit_for1sight_prediction_data_v2_5_0.py *> .\\prediction_data_audit_v2_5_0.log
"""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
WAREHOUSE_LOCAL = PROJECT_ROOT / "for1sight_warehouse"
RUNTIME_ROOT = PROJECT_ROOT / "for1sight_runtime"
TEMP_ROOT = RUNTIME_ROOT / "python_temp"
SPARK_ROOT = RUNTIME_ROOT / "spark_local" / "prediction_data_audit"

# Keep Spark/Python temporary files inside the existing project runtime.
TEMP_ROOT.mkdir(parents=True, exist_ok=True)
SPARK_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(TEMP_ROOT)
os.environ["TMP"] = str(TEMP_ROOT)
os.environ["TMPDIR"] = str(TEMP_ROOT)

import tempfile
tempfile.tempdir = str(TEMP_ROOT)

HDFS_URI = "hdfs://localhost:9000"
HDFS_PATHS = {
    "SILVER LAP TIMES": f"{HDFS_URI}/for1sight_warehouse/silver/hf_laptimes",
    "LEGACY COMBINED LAP TIMES": f"{HDFS_URI}/for1sight_warehouse/laps_2024_2026.parquet",
    "LEGACY USER LAP TIMES": f"{HDFS_URI}/user/for1sight/warehouse/hf_laptimes.parquet",
    "SILVER RACE TELEMETRY (SCHEMA ONLY)": f"{HDFS_URI}/for1sight_warehouse/silver/race_telemetry",
    "GOLD TELEMETRY AUDIT": f"{HDFS_URI}/for1sight_warehouse/gold/race_telemetry/telemetry_audit",
    "GOLD COMPARISON CANDIDATES": f"{HDFS_URI}/for1sight_warehouse/gold/race_telemetry/comparison_candidates",
}

LOCAL_PARQUET_CANDIDATES = {
    "EXISTING QUALIFYING ML TRAINING TABLE": [
        WAREHOUSE_LOCAL / "ml_training_table.parquet",
        PROJECT_ROOT / "ml_training_table.parquet",
    ],
    "LOCAL CLEAN LAP TIMES": [
        WAREHOUSE_LOCAL / "hf_laptimes_clean.parquet",
        PROJECT_ROOT / "hf_laptimes_clean.parquet",
    ],
    "LOCAL COMBINED LAP TIMES": [
        WAREHOUSE_LOCAL / "laps_2024_2026.parquet",
        PROJECT_ROOT / "laps_2024_2026.parquet",
    ],
}

EXPECTED_QUALI_COLUMNS = [
    "Year", "Event", "Driver", "Best_FP_Time", "Best_Q_Time", "Pace_Improvement"
]
EXPECTED_LAP_COLUMNS = [
    "year", "event_name", "session_type", "driver_code", "lap", "time",
    "s1", "s2", "s3", "lSD"
]
POTENTIAL_PRE_RACE_FEATURES = [
    "Best_FP_Time", "Best_Q_Time", "Pace_Improvement",
    "s1", "s2", "s3", "compound", "tyre_compound", "tyre_life",
    "track_temp", "air_temp", "humidity", "pressure", "rainfall",
    "driver_code", "team", "constructor", "event_name", "year",
]

REPORT = []

def emit(message=""):
    line = str(message)
    REPORT.append(line)
    print(line)

def section(title):
    emit("\n" + "=" * 88)
    emit(title)
    emit("=" * 88)

def find_first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None

def norm_map(columns):
    return {str(c).lower(): c for c in columns}

def get_col(columns, candidates):
    mapping = norm_map(columns)
    for candidate in candidates:
        if candidate.lower() in mapping:
            return mapping[candidate.lower()]
    return None

def safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

def extract_races(obj):
    """Accept common Jolpica/Ergast cache shapes without making network calls."""
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
        value = obj.get(key)
        if isinstance(value, list):
            return value
    return []

def audit_race_result_caches():
    section("PART 4 — OFFICIAL RACE-RESULT CACHE AUDIT")
    candidates = []
    if WAREHOUSE_LOCAL.exists():
        for pattern in ("race_results_*.json", "race-results-*.json", "results_*.json"):
            candidates.extend(WAREHOUSE_LOCAL.glob(pattern))
    for pattern in ("race_results_*.json", "race-results-*.json", "results_*.json"):
        candidates.extend(PROJECT_ROOT.glob(pattern))
    candidates = sorted(set(candidates))

    if not candidates:
        emit("No local race_results_*.json / results_*.json cache was found.")
        emit("The existing Driver Comparison code uses Jolpica/Ergast race-result data.")
        emit("This audit does not call the API. We will need to confirm whether the shared")
        emit("Jolpica cache can be reused or create a controlled, versioned results snapshot.")
        return

    for path in candidates:
        emit(f"\nFile: {path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path}")
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            races = extract_races(obj)
            result_rows = []
            for race in races:
                results = race.get("Results") or race.get("results") or []
                for result in results:
                    driver = result.get("Driver") or result.get("driver") or {}
                    constructor = result.get("Constructor") or result.get("constructor") or {}
                    pos_order = result.get("positionOrder", result.get("position_order"))
                    pos = result.get("position")
                    result_rows.append({
                        "round": race.get("round"),
                        "event": race.get("raceName") or race.get("event_name") or race.get("name"),
                        "date": race.get("date"),
                        "driver_code": driver.get("code") or driver.get("driverId") or driver.get("familyName"),
                        "constructor": constructor.get("name") or constructor.get("constructorId"),
                        "position_order": pos_order,
                        "position": pos,
                        "grid": result.get("grid"),
                        "status": result.get("status"),
                        "points": result.get("points"),
                    })
            winner_rows = [
                row for row in result_rows
                if str(row["position_order"] or row["position"]) == "1"
            ]
            numeric_positions = sum(
                1 for row in result_rows
                if safe_float(row["position_order"]) is not None or safe_float(row["position"]) is not None
            )
            emit(f"  Grand Prix events: {len(races)}")
            emit(f"  Driver-result rows: {len(result_rows)}")
            emit(f"  Winner-labelled rows: {len(winner_rows)}")
            emit(f"  Rows with numeric classification position: {numeric_positions}")
            if result_rows:
                emit("  Example result row: " + json.dumps(result_rows[0], ensure_ascii=False))
        except Exception as exc:
            emit(f"  ERROR reading JSON cache: {type(exc).__name__}: {exc}")

def inspect_dataframe(spark, label, path, count_rows=True, detailed=False):
    section(f"DATASET — {label}")
    emit(f"Source: {path}")
    try:
        df = spark.read.parquet(path)
        columns = df.columns
        emit(f"Column count: {len(columns)}")
        emit("Columns: " + ", ".join(columns))
        emit("Schema:")
        for field in df.schema.fields:
            emit(f"  - {field.name}: {field.dataType.simpleString()} (nullable={field.nullable})")

        if count_rows:
            row_count = df.count()
            emit(f"Row count: {row_count:,}")
        else:
            row_count = None
            emit("Row count: SKIPPED (large telemetry dataset; schema-only audit)")

        year_col = get_col(columns, ["year", "Year"])
        event_col = get_col(columns, ["event_name", "Event", "event", "raceName"])
        session_col = get_col(columns, ["session_type", "Session", "session"])
        driver_col = get_col(columns, ["driver_code", "Driver", "driver", "name_acronym"])
        lap_col = get_col(columns, ["lap", "lap_number", "LapNumber"])

        from pyspark.sql import functions as F

        for col_name, label_name in [
            (year_col, "Year distribution"),
            (session_col, "Session distribution"),
        ]:
            if col_name:
                emit(f"{label_name}:")
                pairs = df.groupBy(col_name).count().orderBy(col_name).limit(30).collect()
                for row in pairs:
                    emit(f"  {row[0]}: {row['count']:,}")

        for col_name, label_name in [
            (event_col, "Distinct events"),
            (driver_col, "Distinct drivers"),
        ]:
            if col_name:
                value = df.select(F.countDistinct(F.col(col_name)).alias("n")).first()["n"]
                emit(f"{label_name}: {value}")

        if detailed and row_count is not None and row_count > 0:
            # Compact null/validity profile for expected prediction fields.
            expected = EXPECTED_QUALI_COLUMNS if "TRAINING TABLE" in label else EXPECTED_LAP_COLUMNS
            actual_map = norm_map(columns)
            available = [actual_map[c.lower()] for c in expected if c.lower() in actual_map]
            if available:
                emit("Selected-column non-null counts:")
                exprs = [
                    F.sum(F.when(F.col(c).isNotNull(), 1).otherwise(0)).alias(c)
                    for c in available
                ]
                values = df.agg(*exprs).first().asDict()
                for c in available:
                    n = int(values.get(c) or 0)
                    pct = (n / row_count * 100) if row_count else 0
                    emit(f"  {c}: {n:,}/{row_count:,} ({pct:.1f}% non-null)")

            if "TRAINING TABLE" in label:
                required = ["Year", "Event", "Driver", "Best_FP_Time", "Best_Q_Time"]
                missing = [c for c in required if c.lower() not in actual_map]
                emit("Qualifying prototype required columns missing: " + (", ".join(missing) if missing else "NONE"))
                fp_col = get_col(columns, ["Best_FP_Time"])
                q_col = get_col(columns, ["Best_Q_Time"])
                if fp_col and q_col:
                    emit("Qualifying target/input sanity:")
                    sanity = df.agg(
                        F.sum(F.when(F.col(fp_col).isNull(), 1).otherwise(0)).alias("fp_null"),
                        F.sum(F.when(F.col(q_col).isNull(), 1).otherwise(0)).alias("q_null"),
                        F.sum(F.when(F.col(fp_col) <= 0, 1).otherwise(0)).alias("fp_nonpositive"),
                        F.sum(F.when(F.col(q_col) <= 0, 1).otherwise(0)).alias("q_nonpositive"),
                    ).first().asDict()
                    emit("  " + json.dumps({k: int(v or 0) for k, v in sanity.items()}))

            if "LAP TIMES" in label or "CLEAN LAP" in label:
                key_cols = [c for c in [year_col, event_col, session_col, driver_col, lap_col] if c]
                if len(key_cols) >= 4:
                    duplicate_groups = (
                        df.groupBy(*key_cols).count()
                        .filter(F.col("count") > 1)
                        .count()
                    )
                    emit(f"Duplicate event/session/driver/lap key groups: {duplicate_groups:,}")

            sample_cols = []
            for candidate in (
                "year", "Year", "event_name", "Event", "session_type",
                "driver_code", "Driver", "lap", "time", "Best_FP_Time",
                "Best_Q_Time", "Pace_Improvement", "position", "positionOrder",
            ):
                actual = actual_map.get(candidate.lower())
                if actual and actual not in sample_cols:
                    sample_cols.append(actual)
            if sample_cols:
                emit("Sample rows (up to 5):")
                for row in df.select(*sample_cols).limit(5).toJSON().collect():
                    emit("  " + row)

        return {"ok": True, "columns": columns, "rows": row_count}
    except Exception as exc:
        emit(f"NOT AVAILABLE / READ FAILED: {type(exc).__name__}: {exc}")
        return {"ok": False, "columns": [], "rows": None}

def audit_local_artifacts():
    section("PART 1 — EXISTING LOCAL PREDICTION ARTIFACTS")
    for label, candidates in LOCAL_PARQUET_CANDIDATES.items():
        found = find_first_existing(candidates)
        if found:
            emit(f"{label}: FOUND — {found}")
        else:
            emit(f"{label}: NOT FOUND")
            for candidate in candidates:
                emit(f"  checked: {candidate}")

    model_candidates = [
        PROJECT_ROOT / "models" / "quali_model_xgb.joblib",
        WAREHOUSE_LOCAL / "models" / "quali_model_xgb.joblib",
    ]
    model_path = find_first_existing(model_candidates)
    emit(f"Existing XGBoost qualifying model: {model_path if model_path else 'NOT FOUND'}")

    section("PART 2 — PROJECT TRAINING / PREDICTION SOURCE FILES")
    search_dirs = [PROJECT_ROOT, PROJECT_ROOT / "pages", PROJECT_ROOT / "models"]
    matches = []
    for folder in search_dirs:
        if folder.exists():
            try:
                for path in folder.iterdir():
                    if path.is_file() and re.search(r"(predict|train|quali|model|feature)", path.name, re.I):
                        matches.append(path)
            except OSError:
                pass
    if matches:
        for path in sorted(set(matches)):
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            emit(f"  {path.relative_to(PROJECT_ROOT)} ({size:,} bytes)")
    else:
        emit("No matching top-level/pages/models training or prediction files found.")

def main():
    section("FOR1SIGHT V2.5.0 — PREDICTION DATA AUDIT")
    emit(f"Audit time: {datetime.now().isoformat(timespec='seconds')}")
    emit(f"Project root: {PROJECT_ROOT}")
    emit("Mode: READ-ONLY. No HDFS writes, no ETL rerun, no model training, no API requests.")

    audit_local_artifacts()
    audit_race_result_caches()

    try:
        from pyspark.sql import SparkSession
    except Exception as exc:
        section("SPARK INITIALISATION")
        emit(f"PySpark import failed: {type(exc).__name__}: {exc}")
        emit("Install/use the same Python environment that currently runs your For1Sight PySpark scripts.")
        return

    spark = None
    try:
        spark = (
            SparkSession.builder
            .appName("For1Sight_V2_5_0_Prediction_Data_Audit")
            .master("local[2]")
            .config("spark.driver.memory", "3g")
            .config("spark.driver.maxResultSize", "1g")
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.local.dir", str(SPARK_ROOT))
            .config("spark.hadoop.fs.defaultFS", HDFS_URI)
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")
        section("PART 3 — SPARK / HDFS CONNECTION")
        emit(f"Spark version: {spark.version}")
        emit(f"Default HDFS: {HDFS_URI}")
        emit(f"Spark local temp: {SPARK_ROOT}")

        # Local qualifying prototype dataset.
        training_path = find_first_existing(LOCAL_PARQUET_CANDIDATES["EXISTING QUALIFYING ML TRAINING TABLE"])
        if training_path:
            inspect_dataframe(
                spark, "EXISTING QUALIFYING ML TRAINING TABLE",
                str(training_path), count_rows=True, detailed=True
            )
        else:
            emit("\nExisting qualifying training table was not found locally; skipping its schema inspection.")

        # Local cleaned lap table if present.
        clean_path = find_first_existing(LOCAL_PARQUET_CANDIDATES["LOCAL CLEAN LAP TIMES"])
        if clean_path:
            inspect_dataframe(spark, "LOCAL CLEAN LAP TIMES", str(clean_path), count_rows=True, detailed=True)

        # Primary HDFS Silver lap-time warehouse.
        inspect_dataframe(
            spark, "HDFS SILVER LAP TIMES",
            HDFS_PATHS["SILVER LAP TIMES"], count_rows=True, detailed=True
        )

        # Legacy aggregate sources are inspected only if present.
        for label in ("LEGACY COMBINED LAP TIMES", "LEGACY USER LAP TIMES"):
            inspect_dataframe(spark, label, HDFS_PATHS[label], count_rows=True, detailed=False)

        # Race telemetry is intentionally schema-only: it is a large race-session dataset
        # and must not be used as a pre-race feature source.
        inspect_dataframe(
            spark, "HDFS SILVER RACE TELEMETRY (SCHEMA ONLY)",
            HDFS_PATHS["SILVER RACE TELEMETRY (SCHEMA ONLY)"],
            count_rows=False, detailed=False
        )
        inspect_dataframe(
            spark, "HDFS GOLD TELEMETRY AUDIT",
            HDFS_PATHS["GOLD TELEMETRY AUDIT"],
            count_rows=True, detailed=False
        )
        inspect_dataframe(
            spark, "HDFS GOLD COMPARISON CANDIDATES",
            HDFS_PATHS["GOLD COMPARISON CANDIDATES"],
            count_rows=True, detailed=False
        )

    except Exception as exc:
        section("AUDIT EXECUTION ERROR")
        emit(f"{type(exc).__name__}: {exc}")
        emit(traceback.format_exc())
    finally:
        if spark is not None:
            try:
                spark.stop()
                emit("\nSpark session stopped.")
            except Exception:
                pass

    section("PART 5 — AUDIT INTERPRETATION CHECKLIST")
    emit("1. Qualifying prediction prototype: Best_FP_Time -> Best_Q_Time is a qualifying-time task,")
    emit("   not yet a race-winner or finishing-position model.")
    emit("2. The current Streamlit page calculates an event RMSE after model.predict on the selected")
    emit("   event. The page itself does not demonstrate an independent chronological holdout; do not")
    emit("   present that displayed RMSE as final test performance until the training split is verified.")
    emit("3. Event_Code must use a stable training-time mapping. Re-creating category codes separately")
    emit("   inside each selected race/year can map an event to a different integer.")
    emit("4. Pre-race winner/position prediction needs official race classification labels joined at")
    emit("   driver + event + season level. Race-session telemetry must not be used for a pre-race forecast.")
    emit("5. No model training or dashboard files are changed by this audit.")

    report_path = WAREHOUSE_LOCAL / "prediction_data_audit_v2_5_0.txt"
    try:
        WAREHOUSE_LOCAL.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(REPORT) + "\n", encoding="utf-8")
        print(f"\nAudit report saved to: {report_path}")
    except Exception as exc:
        print(f"\nCould not save report file: {exc}")

if __name__ == "__main__":
    main()
