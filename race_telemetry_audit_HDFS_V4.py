"""
For1Sight — Grand Prix Race Telemetry Audit & HDFS Loader
=========================================================

Purpose
-------
1. Scan ONLY Grand Prix Race telemetry files for seasons 2024, 2025 and 2026.
2. Audit per-lap telemetry quality without modifying source JSON files.
3. Identify laps suitable for track-map and driving-style comparison.
4. Flatten valid telemetry samples with PySpark.
5. Write Silver telemetry samples and Gold audit/comparison-candidate datasets
   to the existing HDFS warehouse.
6. Export small CSV audit reports locally for easy inspection.

Expected local structure
------------------------
For1Sight Project/
    raw_hf_data/
        2024/<Event>/Race/<DRIVER>/<lap>_tel.json
        2025/<Event>/Race/<DRIVER>/<lap>_tel.json
        2026/<Event>/Race/<DRIVER>/<lap>_tel.json
    for1sight_warehouse/                 # existing local warehouse; not modified

Expected telemetry JSON structure
----------------------------------
{
  "tel": {
    "time": [...], "rpm": [...], "speed": [...], "gear": [...],
    "throttle": [...], "brake": [...], "distance": [...],
    "rel_distance": [...], "x": [...], "y": [...], ...
  }
}

Safety
------
- Original raw JSON is read-only.
- Existing hf_laptimes_clean Parquet is never read or modified.
- HDFS output directories are dedicated to this new Race telemetry pipeline.
- Existing final HDFS output paths are never overwritten by this script.
- HDFS path checks use `hdfs dfs -ls -d` output, not unreliable exit codes from `-test -e`.
- Raw Bronze copying is optional and disabled by default to protect HDFS capacity.

Run from the For1Sight project folder:
    python race_telemetry_audit_HDFS_V4.py

Requirements:
    pyspark, Java, Hadoop/HDFS configured and running.
"""

from __future__ import annotations

import csv
import gc
import itertools
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

# Keep Python/PySpark temporary files in a dedicated project runtime folder.
# This avoids relying on Windows AppData\\Local\\Temp during long ETL runs.
RUNTIME_ROOT = Path(__file__).resolve().parent / "for1sight_runtime"
PYSPARK_TEMP_ROOT = RUNTIME_ROOT / "python_temp"
SPARK_LOCAL_ROOT = RUNTIME_ROOT / "spark_local"

PYSPARK_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
SPARK_LOCAL_ROOT.mkdir(parents=True, exist_ok=True)

os.environ["TEMP"] = str(PYSPARK_TEMP_ROOT)
os.environ["TMP"] = str(PYSPARK_TEMP_ROOT)
os.environ["TMPDIR"] = str(PYSPARK_TEMP_ROOT)
tempfile.tempdir = str(PYSPARK_TEMP_ROOT)

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)


# ============================================================================
# 1. USER CONFIGURATION — CHECK THESE PATHS BEFORE RUNNING
# ============================================================================

PROJECT_ROOT = Path(r"C:\Users\ayepd\Desktop\For1Sight Project")
RAW_ROOT = PROJECT_ROOT / "raw_hf_data"
LOCAL_REPORT_DIR = PROJECT_ROOT / "for1sight_warehouse" / "race_telemetry_audit"

SEASONS = [2024, 2025, 2026]
RACE_FOLDER_NAME = "Race"  # exact session folder; Practice/Sprint/Qualifying excluded

HDFS_URI = "hdfs://localhost:9000"
HDFS_BIN = r"C:\hadoop\bin\hdfs.cmd"
HDFS_WAREHOUSE_ROOT = "/for1sight_warehouse"

HDFS_SILVER_PATH = f"{HDFS_WAREHOUSE_ROOT}/silver/race_telemetry"
HDFS_GOLD_AUDIT_PATH = f"{HDFS_WAREHOUSE_ROOT}/gold/race_telemetry/telemetry_audit"
HDFS_GOLD_PAIRS_PATH = f"{HDFS_WAREHOUSE_ROOT}/gold/race_telemetry/comparison_candidates"
HDFS_BRONZE_PATH = f"{HDFS_WAREHOUSE_ROOT}/bronze/race_telemetry"

# First run: keep False. Silver and Gold outputs will still be stored in HDFS.
# Turn True only if you also want to copy the original Race folders to HDFS Bronze.
# The script checks HDFS paths and will never overwrite an existing Bronze Race folder.
COPY_RAW_TO_BRONZE = False

# V4 safety lock: existing final Silver/Gold outputs are never overwritten.

# PySpark settings for a 16 GB Windows development machine.
# V4 uses two local Spark workers and bounded batches to reduce memory pressure.
SPARK_MASTER = "local[2]"
SPARK_DRIVER_MEMORY = "4g"
SPARK_SHUFFLE_PARTITIONS = "4"
SPARK_WRITE_PARTITIONS = 1
BATCH_SAMPLE_LIMIT = 50000

# V4 writes each Silver batch directly to temporary HDFS storage. It does not
# create a second full Silver Parquet copy on the local C: drive.
PREFLIGHT_FILE_COUNT = 500
SPACE_CHECK_INTERVAL_FILES = 5000
HDFS_SPACE_SAFETY_MULTIPLIER = 1.50
MIN_LOCAL_FREE_GB = 2.0
MIN_HDFS_RESERVE_GB = 2.0
MIN_LOCAL_FREE_BYTES = int(MIN_LOCAL_FREE_GB * (1024 ** 3))
MIN_HDFS_RESERVE_BYTES = int(MIN_HDFS_RESERVE_GB * (1024 ** 3))

# Quality thresholds. These are audit rules, not claims that every F1 lap must
# have exactly the same sampling rate or telemetry profile.
MIN_SAMPLES_PER_LAP = 50
MIN_VALID_RATIO = 0.80
MIN_UNIQUE_SPEED_VALUES = 5
MIN_UNIQUE_GPS_POSITIONS = 10
MIN_DISTANCE_SPAN_METRES = 1000.0

# Optional local CSV exports (small audit summaries only; not the warehouse).
EXPORT_LOCAL_CSV = True


# ============================================================================
# 2. LOGGING AND SPARK SCHEMAS
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("For1Sight_Race_Telemetry_Audit")

SAMPLE_SCHEMA = StructType([
    StructField("year", IntegerType(), False),
    StructField("event_name", StringType(), False),
    StructField("event_slug", StringType(), False),
    StructField("session_type", StringType(), False),
    StructField("driver_code", StringType(), False),
    StructField("lap_number", IntegerType(), False),
    StructField("sample_index", IntegerType(), False),
    StructField("time_s", DoubleType(), True),
    StructField("rpm", DoubleType(), True),
    StructField("speed_kmh", DoubleType(), True),
    StructField("gear", IntegerType(), True),
    StructField("throttle_pct", DoubleType(), True),
    StructField("brake", IntegerType(), True),
    StructField("drs", IntegerType(), True),
    StructField("distance_m", DoubleType(), True),
    StructField("rel_distance", DoubleType(), True),
    StructField("rel_distance_source", StringType(), False),
    StructField("x", DoubleType(), True),
    StructField("y", DoubleType(), True),
    StructField("z", DoubleType(), True),
    StructField("acc_x", DoubleType(), True),
    StructField("acc_y", DoubleType(), True),
    StructField("acc_z", DoubleType(), True),
    StructField("driver_ahead", StringType(), True),
    StructField("distance_to_driver_ahead_m", DoubleType(), True),
    StructField("data_key", StringType(), True),
    StructField("source_file", StringType(), False),
    StructField("driving_style_usable", BooleanType(), False),
])

AUDIT_SCHEMA = StructType([
    StructField("year", IntegerType(), False),
    StructField("event_name", StringType(), False),
    StructField("event_slug", StringType(), False),
    StructField("session_type", StringType(), False),
    StructField("driver_code", StringType(), False),
    StructField("lap_number", IntegerType(), False),
    StructField("source_file", StringType(), False),
    StructField("sample_count", IntegerType(), False),
    StructField("quality_status", StringType(), False),
    StructField("track_map_usable", BooleanType(), False),
    StructField("driving_style_usable", BooleanType(), False),
    StructField("required_lengths_match", BooleanType(), False),
    StructField("speed_unique_values", IntegerType(), False),
    StructField("throttle_unique_values", IntegerType(), False),
    StructField("gear_unique_values", IntegerType(), False),
    StructField("rpm_unique_values", IntegerType(), False),
    StructField("gps_unique_positions", IntegerType(), False),
    StructField("distance_span_m", DoubleType(), True),
    StructField("rel_distance_source", StringType(), False),
    StructField("missing_fields", StringType(), True),
    StructField("constant_fields", StringType(), True),
    StructField("issues", StringType(), True),
])

PAIR_SCHEMA = StructType([
    StructField("year", IntegerType(), False),
    StructField("event_name", StringType(), False),
    StructField("event_slug", StringType(), False),
    StructField("session_type", StringType(), False),
    StructField("driver_a", StringType(), False),
    StructField("driver_b", StringType(), False),
    StructField("common_lap_count", IntegerType(), False),
    StructField("first_common_lap", IntegerType(), False),
    StructField("last_common_lap", IntegerType(), False),
    StructField("common_laps", StringType(), False),
])


# ============================================================================
# 3. GENERAL HELPERS
# ============================================================================


def slugify(value: str) -> str:
    """Create a stable, filesystem-friendly event partition name."""
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return value or "unknown_event"


def as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def as_int(value: Any) -> int | None:
    number = as_float(value)
    return int(number) if number is not None else None


def numeric_values(values: Any) -> list[float | None]:
    if not isinstance(values, list):
        return []
    return [as_float(value) for value in values]


def valid_ratio(values: list[float | None]) -> float:
    if not values:
        return 0.0
    return sum(value is not None for value in values) / len(values)


def unique_count(values: list[float | None], decimals: int = 2) -> int:
    return len({round(value, decimals) for value in values if value is not None})


def monotonic_ratio(values: list[float | None]) -> float:
    clean = [value for value in values if value is not None]
    if len(clean) < 2:
        return 0.0
    comparisons = sum(clean[i] >= clean[i - 1] for i in range(1, len(clean)))
    return comparisons / (len(clean) - 1)


def relative_distance_values(
    original: list[float | None], distance: list[float | None]
) -> tuple[list[float | None], str]:
    """Use valid rel_distance; otherwise derive a normalized value from distance."""
    if original and valid_ratio(original) >= MIN_VALID_RATIO:
        clean = [value for value in original if value is not None]
        in_range = sum(-0.01 <= value <= 1.01 for value in clean) / max(1, len(clean))
        if in_range >= MIN_VALID_RATIO and monotonic_ratio(original) >= 0.90:
            return original, "original"

    clean_distance = [value for value in distance if value is not None]
    if len(clean_distance) == len(distance) and len(clean_distance) >= 2:
        low, high = min(clean_distance), max(clean_distance)
        if high > low and monotonic_ratio(distance) >= 0.90:
            derived = [(value - low) / (high - low) for value in clean_distance]
            return derived, "derived_from_distance"

    return [None] * len(distance), "unavailable"


def local_relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(RAW_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def hdfs_executable() -> str:
    """Prefer the explicitly configured Hadoop installation on this machine."""
    if Path(HDFS_BIN).is_file():
        return HDFS_BIN
    found = shutil.which("hdfs")
    if found:
        return found
    raise FileNotFoundError(
        f"HDFS command not found. Checked configured path {HDFS_BIN} and PATH. "
        "Update HDFS_BIN at the top of this script."
    )


def hdfs_run(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run Hadoop's CLI safely on Windows (.cmd) or other platforms."""
    exe = hdfs_executable()
    command = [exe, *args]
    if os.name == "nt" and exe.lower().endswith((".cmd", ".bat")):
        completed = subprocess.run(
            subprocess.list2cmdline(command),
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    else:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    if check and completed.returncode != 0:
        raise RuntimeError(
            "HDFS command failed:\n"
            + " ".join(command)
            + "\n"
            + (completed.stderr or completed.stdout)
        )
    return completed


def hdfs_exists(path: str) -> bool:
    """Check an HDFS path from the actual listing, not the CLI exit code.

    On this Windows setup, `hdfs dfs -test -e` returned exit code 0 even for
    a path that `hdfs dfs -ls -R` reported as missing. To avoid that false
    positive, list the exact path with `-ls -d` and inspect the result text.
    """
    uri = path if path.startswith("hdfs://") else f"{HDFS_URI}{path}"
    result = hdfs_run(["dfs", "-ls", "-d", uri], check=False)
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()

    # Existing files and directories are returned as permission-style rows,
    # for example: drwxr-xr-x ... hdfs://localhost:9000/path
    for line in stdout.splitlines():
        if re.match(r"^[d\-l][rwx\-]{9}\s+", line.strip()):
            return True

    combined = f"{stdout}\n{stderr}"
    missing_patterns = (
        r"no such file or directory",
        r"filenotfoundexception",
        r"path .* does not exist",
        r"cannot find .*file",
    )
    if any(re.search(pattern, combined, flags=re.IGNORECASE) for pattern in missing_patterns):
        return False

    # Do not silently interpret permission, connection, or configuration
    # failures as a missing directory. Stop so the user can investigate.
    if not stdout and not stderr and result.returncode == 0:
        raise RuntimeError(
            f"HDFS existence check returned no listing for {uri}. "
            "The path status is ambiguous; refusing to continue."
        )
    raise RuntimeError(
        f"Could not determine whether this HDFS path exists: {uri}\n"
        f"Exit code: {result.returncode}\n"
        f"STDOUT: {stdout or '<empty>'}\n"
        f"STDERR: {stderr or '<empty>'}"
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("No records found.\n", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ============================================================================
# 4. DISCOVER RACE TELEMETRY FILES ONLY
# ============================================================================


def discover_race_telemetry() -> list[dict[str, Any]]:
    if not RAW_ROOT.exists():
        raise FileNotFoundError(
            f"Raw data folder not found: {RAW_ROOT}\n"
            "Correct RAW_ROOT in the configuration section before running."
        )

    discovered: list[dict[str, Any]] = []
    for year in SEASONS:
        year_dir = RAW_ROOT / str(year)
        if not year_dir.is_dir():
            logger.warning("Season folder not found; skipping: %s", year_dir)
            continue

        event_dirs = sorted(path for path in year_dir.iterdir() if path.is_dir())
        for event_dir in event_dirs:
            race_dir = next(
                (child for child in event_dir.iterdir()
                 if child.is_dir() and child.name.casefold() == RACE_FOLDER_NAME.casefold()),
                None,
            )
            if race_dir is None:
                continue

            for file_path in race_dir.rglob("*_tel.json"):
                match = re.fullmatch(r"(\d+)_tel\.json", file_path.name, flags=re.IGNORECASE)
                if not match:
                    continue
                discovered.append({
                    "year": year,
                    "event_name": event_dir.name,
                    "event_slug": slugify(event_dir.name),
                    "session_type": "Race",
                    "driver_code": file_path.parent.name.upper(),
                    "lap_number": int(match.group(1)),
                    "path": file_path,
                    "race_dir": race_dir,
                })

    discovered.sort(key=lambda item: (
        item["year"], item["event_name"], item["driver_code"], item["lap_number"]
    ))
    return discovered


# ============================================================================
# 5. AUDIT ONE TELEMETRY FILE AND FLATTEN USABLE SAMPLES
# ============================================================================

REQUIRED_FIELDS = [
    "time", "rpm", "speed", "gear", "throttle", "brake", "distance", "x", "y"
]
OPTIONAL_NUMERIC_FIELDS = [
    "drs", "z", "acc_x", "acc_y", "acc_z", "DistanceToDriverAhead"
]


def audit_and_flatten_file(meta: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path: Path = meta["path"]
    source_rel = local_relative_path(path)
    issues: list[str] = []
    missing_fields: list[str] = []
    constant_fields: list[str] = []
    flattened: list[dict[str, Any]] = []

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        audit = _base_audit(meta, source_rel)
        audit.update({
            "quality_status": "INVALID",
            "issues": f"JSON_READ_ERROR: {type(exc).__name__}: {exc}",
        })
        return audit, flattened

    tel = payload.get("tel") if isinstance(payload, dict) else None
    if not isinstance(tel, dict):
        audit = _base_audit(meta, source_rel)
        audit.update({"quality_status": "INVALID", "issues": "MISSING_OR_INVALID_TEL_OBJECT"})
        return audit, flattened

    for field in REQUIRED_FIELDS:
        if not isinstance(tel.get(field), list):
            missing_fields.append(field)

    required_lengths = {
        len(tel[field]) for field in REQUIRED_FIELDS if isinstance(tel.get(field), list)
    }
    required_lengths_match = len(required_lengths) == 1 and not missing_fields
    if not required_lengths_match:
        issues.append("REQUIRED_ARRAY_LENGTH_MISMATCH_OR_MISSING_FIELD")

    sample_count = next(iter(required_lengths)) if required_lengths_match else 0
    if sample_count < MIN_SAMPLES_PER_LAP:
        issues.append(f"TOO_FEW_SAMPLES<{MIN_SAMPLES_PER_LAP}")

    arrays: dict[str, list[float | None]] = {
        field: numeric_values(tel.get(field)) for field in REQUIRED_FIELDS
    }
    optional_arrays: dict[str, list[float | None]] = {
        field: numeric_values(tel.get(field)) for field in OPTIONAL_NUMERIC_FIELDS
    }

    # Check all required arrays for sufficient numeric coverage.
    for field, values in arrays.items():
        if values and valid_ratio(values) < MIN_VALID_RATIO:
            issues.append(f"LOW_NUMERIC_COVERAGE:{field}")

    speed = arrays.get("speed", [])
    throttle = arrays.get("throttle", [])
    gear = arrays.get("gear", [])
    rpm = arrays.get("rpm", [])
    x_values = arrays.get("x", [])
    y_values = arrays.get("y", [])
    distance = arrays.get("distance", [])

    speed_unique = unique_count(speed, decimals=1)
    throttle_unique = unique_count(throttle, decimals=1)
    gear_unique = unique_count(gear, decimals=0)
    rpm_unique = unique_count(rpm, decimals=0)
    gps_unique = len({(round(x, 2), round(y, 2)) for x, y in zip(x_values, y_values)
                      if x is not None and y is not None})

    if speed_unique < MIN_UNIQUE_SPEED_VALUES:
        issues.append("SUSPICIOUS_CONSTANT_OR_LOW_VARIATION_SPEED")
    if gps_unique < MIN_UNIQUE_GPS_POSITIONS:
        issues.append("INSUFFICIENT_GPS_POSITION_VARIATION")

    distance_clean = [value for value in distance if value is not None]
    distance_span = (
        float(max(distance_clean) - min(distance_clean)) if distance_clean else None
    )
    if monotonic_ratio(distance) < 0.90:
        issues.append("DISTANCE_NOT_MONOTONIC_ENOUGH")
    if distance_span is None or distance_span < MIN_DISTANCE_SPAN_METRES:
        issues.append("DISTANCE_SPAN_TOO_SMALL_OR_MISSING")

    # Constant fields are reported. Throttle/gear/RPM variation is required for
    # a full driving-style candidate, but map-usable laps can still be retained.
    for field, values, decimals in (
        ("speed", speed, 1), ("throttle", throttle, 1),
        ("gear", gear, 0), ("rpm", rpm, 0),
    ):
        if unique_count(values, decimals) <= 1 and values:
            constant_fields.append(field)

    track_map_usable = (
        required_lengths_match
        and sample_count >= MIN_SAMPLES_PER_LAP
        and valid_ratio(speed) >= MIN_VALID_RATIO
        and valid_ratio(x_values) >= MIN_VALID_RATIO
        and valid_ratio(y_values) >= MIN_VALID_RATIO
        and speed_unique >= MIN_UNIQUE_SPEED_VALUES
        and gps_unique >= MIN_UNIQUE_GPS_POSITIONS
        and distance_span is not None
        and distance_span >= MIN_DISTANCE_SPAN_METRES
        and monotonic_ratio(distance) >= 0.90
    )

    driving_style_usable = (
        track_map_usable
        and valid_ratio(throttle) >= MIN_VALID_RATIO
        and valid_ratio(gear) >= MIN_VALID_RATIO
        and valid_ratio(rpm) >= MIN_VALID_RATIO
        and throttle_unique >= 2
        and gear_unique >= 2
        and rpm_unique >= 5
    )

    if track_map_usable and driving_style_usable:
        quality_status = "VALID_DRIVING_STYLE"
    elif track_map_usable:
        quality_status = "MAP_ONLY_REVIEW"
        issues.append("TRACK_MAP_USABLE_BUT_DRIVING_INPUTS_NEED_REVIEW")
    else:
        quality_status = "INVALID"

    # rel_distance can be missing in some extracted files. If so, derive it
    # from the distance array when that array is usable and monotonic.
    rel_distance_raw = numeric_values(tel.get("rel_distance"))
    if len(rel_distance_raw) != sample_count:
        rel_distance_raw = []
    rel_distance, rel_distance_source = relative_distance_values(
        rel_distance_raw, distance
    ) if sample_count else ([], "unavailable")

    # Optional arrays with mismatched lengths are not allowed to shift core
    # telemetry. Their values are set to null rather than truncating other data.
    aligned_optional: dict[str, list[float | None]] = {}
    for field in OPTIONAL_NUMERIC_FIELDS:
        values = optional_arrays.get(field, [])
        aligned_optional[field] = values if len(values) == sample_count else [None] * sample_count

    driver_ahead_raw = tel.get("DriverAhead")
    if not isinstance(driver_ahead_raw, list) or len(driver_ahead_raw) != sample_count:
        driver_ahead_raw = [None] * sample_count
    distance_ahead_raw = optional_arrays.get("DistanceToDriverAhead", [])
    if len(distance_ahead_raw) != sample_count:
        distance_ahead_raw = [None] * sample_count

    data_key = tel.get("dataKey")
    if track_map_usable:
        for index in range(sample_count):
            # Required fields are aligned and numeric coverage was validated.
            # Null samples remain null; no interpolation is invented.
            row = {
                "year": meta["year"],
                "event_name": meta["event_name"],
                "event_slug": meta["event_slug"],
                "session_type": "Race",
                "driver_code": meta["driver_code"],
                "lap_number": meta["lap_number"],
                "sample_index": index,
                "time_s": arrays["time"][index],
                "rpm": arrays["rpm"][index],
                "speed_kmh": arrays["speed"][index],
                "gear": as_int(arrays["gear"][index]),
                "throttle_pct": arrays["throttle"][index],
                "brake": as_int(arrays["brake"][index]),
                "drs": as_int(aligned_optional["drs"][index]),
                "distance_m": arrays["distance"][index],
                "rel_distance": rel_distance[index] if index < len(rel_distance) else None,
                "rel_distance_source": rel_distance_source,
                "x": arrays["x"][index],
                "y": arrays["y"][index],
                "z": aligned_optional["z"][index],
                "acc_x": aligned_optional["acc_x"][index],
                "acc_y": aligned_optional["acc_y"][index],
                "acc_z": aligned_optional["acc_z"][index],
                "driver_ahead": (
                    str(driver_ahead_raw[index])
                    if driver_ahead_raw[index] is not None else None
                ),
                "distance_to_driver_ahead_m": distance_ahead_raw[index],
                "data_key": str(data_key) if data_key is not None else None,
                "source_file": source_rel,
                "driving_style_usable": bool(driving_style_usable),
            }
            flattened.append(row)

    audit = {
        "year": meta["year"],
        "event_name": meta["event_name"],
        "event_slug": meta["event_slug"],
        "session_type": "Race",
        "driver_code": meta["driver_code"],
        "lap_number": meta["lap_number"],
        "source_file": source_rel,
        "sample_count": sample_count,
        "quality_status": quality_status,
        "track_map_usable": bool(track_map_usable),
        "driving_style_usable": bool(driving_style_usable),
        "required_lengths_match": bool(required_lengths_match),
        "speed_unique_values": speed_unique,
        "throttle_unique_values": throttle_unique,
        "gear_unique_values": gear_unique,
        "rpm_unique_values": rpm_unique,
        "gps_unique_positions": gps_unique,
        "distance_span_m": distance_span,
        "rel_distance_source": rel_distance_source,
        "missing_fields": ";".join(missing_fields),
        "constant_fields": ";".join(constant_fields),
        "issues": ";".join(dict.fromkeys(issues)),
    }
    return audit, flattened


def _base_audit(meta: dict[str, Any], source_rel: str) -> dict[str, Any]:
    return {
        "year": meta["year"],
        "event_name": meta["event_name"],
        "event_slug": meta["event_slug"],
        "session_type": "Race",
        "driver_code": meta["driver_code"],
        "lap_number": meta["lap_number"],
        "source_file": source_rel,
        "sample_count": 0,
        "quality_status": "INVALID",
        "track_map_usable": False,
        "driving_style_usable": False,
        "required_lengths_match": False,
        "speed_unique_values": 0,
        "throttle_unique_values": 0,
        "gear_unique_values": 0,
        "rpm_unique_values": 0,
        "gps_unique_positions": 0,
        "distance_span_m": None,
        "rel_distance_source": "unavailable",
        "missing_fields": "",
        "constant_fields": "",
        "issues": "",
    }


# ============================================================================
# 6. IDENTIFY SAME-EVENT, SAME-LAP DRIVER COMPARISON CANDIDATES
# ============================================================================


def build_comparison_pairs(audit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid_by_event: dict[tuple[int, str, str], dict[str, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    event_names: dict[tuple[int, str, str], str] = {}

    for row in audit_rows:
        if row["driving_style_usable"]:
            key = (row["year"], row["event_slug"], row["session_type"])
            valid_by_event[key][row["driver_code"]].add(row["lap_number"])
            event_names[key] = row["event_name"]

    pairs: list[dict[str, Any]] = []
    for key, driver_laps in valid_by_event.items():
        year, event_slug, session_type = key
        drivers = sorted(driver_laps)
        for driver_a, driver_b in itertools.combinations(drivers, 2):
            common = sorted(driver_laps[driver_a].intersection(driver_laps[driver_b]))
            if not common:
                continue
            pairs.append({
                "year": year,
                "event_name": event_names[key],
                "event_slug": event_slug,
                "session_type": session_type,
                "driver_a": driver_a,
                "driver_b": driver_b,
                "common_lap_count": len(common),
                "first_common_lap": common[0],
                "last_common_lap": common[-1],
                "common_laps": ",".join(str(value) for value in common),
            })

    return sorted(pairs, key=lambda row: (
        row["year"], row["event_name"], -row["common_lap_count"],
        row["driver_a"], row["driver_b"]
    ))


# ============================================================================
# 7. OPTIONAL RAW BRONZE COPY — OFF BY DEFAULT
# ============================================================================


def copy_raw_race_folders_to_bronze(discovered: list[dict[str, Any]]) -> None:
    """Copy each Race folder once. Existing HDFS Race folders are skipped."""
    unique_race_dirs: dict[tuple[int, str], Path] = {}
    for meta in discovered:
        unique_race_dirs[(meta["year"], meta["event_slug"])] = meta["race_dir"]

    copied = 0
    skipped = 0
    for (year, event_slug), local_race_dir in sorted(unique_race_dirs.items()):
        event_target = f"{HDFS_BRONZE_PATH}/year={year}/event={event_slug}"
        race_target = f"{event_target}/Race"
        hdfs_run(["dfs", "-mkdir", "-p", f"{HDFS_URI}{event_target}"])
        if hdfs_exists(race_target):
            skipped += 1
            logger.info("Bronze Race folder already exists; skipping: %s", race_target)
            continue
        hdfs_run([
            "dfs", "-put", "-p", str(local_race_dir), f"{HDFS_URI}{event_target}"
        ])
        copied += 1
        logger.info("Copied Bronze Race folder: %s", race_target)

    logger.info("Bronze copy complete | copied=%s | existing_skipped=%s", copied, skipped)



def hdfs_remaining_bytes() -> int:
    """Return aggregate HDFS remaining capacity in bytes."""
    result = hdfs_run(["dfsadmin", "-report"])
    combined = f"{result.stdout}\n{result.stderr}"
    match = re.search(r"DFS Remaining:\s*([\d,]+)", combined, flags=re.IGNORECASE)
    if not match:
        raise RuntimeError(
            "Could not parse HDFS remaining capacity from `hdfs dfsadmin -report`.\n"
            + combined[-4000:]
        )
    return int(match.group(1).replace(",", ""))


def hdfs_path_size_bytes(path: str) -> int:
    """Return HDFS space consumed for a path using the second `-du -s` column."""
    uri = path if path.startswith("hdfs://") else f"{HDFS_URI}{path}"
    result = hdfs_run(["dfs", "-du", "-s", uri], check=False)
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()

    if not stdout:
        combined = f"{stdout}\n{stderr}"
        if re.search(
            r"no such file or directory|filenotfoundexception|does not exist",
            combined,
            flags=re.IGNORECASE,
        ):
            return 0
        if result.returncode == 0:
            return 0
        raise RuntimeError(
            f"Could not measure HDFS path size for {uri}.\n"
            f"Exit code: {result.returncode}\n{combined}"
        )

    for line in stdout.splitlines():
        parts = line.split()
        numeric = [part.replace(",", "") for part in parts[:2]]
        if numeric and numeric[0].isdigit():
            # `hdfs dfs -du` returns: logical_size disk_space_consumed path
            if len(numeric) > 1 and numeric[1].isdigit():
                return int(numeric[1])
            return int(numeric[0])

    raise RuntimeError(
        f"Could not parse HDFS size output for {uri}:\n{stdout}"
    )


def hdfs_delete_tree(path: str, ignore_errors: bool = False) -> None:
    """Delete only the exact temporary path created by this script."""
    uri = path if path.startswith("hdfs://") else f"{HDFS_URI}{path}"
    result = hdfs_run(
        ["dfs", "-rm", "-r", "-skipTrash", uri],
        check=not ignore_errors,
    )
    if result.returncode != 0 and not ignore_errors:
        raise RuntimeError(
            f"Could not remove temporary HDFS path {uri}:\n"
            f"{result.stderr or result.stdout}"
        )


def hdfs_move(source_path: str, destination_path: str) -> None:
    """Move a validated staging directory to its final warehouse location."""
    source_uri = source_path if source_path.startswith("hdfs://") else f"{HDFS_URI}{source_path}"
    destination_uri = (
        destination_path
        if destination_path.startswith("hdfs://")
        else f"{HDFS_URI}{destination_path}"
    )
    hdfs_run(["dfs", "-mv", source_uri, destination_uri])



# ============================================================================
# 8. MAIN PIPELINE
# ============================================================================


def main() -> None:
    print("=" * 78)
    print("FOR1SIGHT — GRAND PRIX RACE TELEMETRY AUDIT & HDFS LOADER (V4)")
    print("=" * 78)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Raw root     : {RAW_ROOT}")
    print(f"Seasons      : {SEASONS}")
    print("Sessions     : Race only")
    print(f"HDFS         : {HDFS_URI}")
    print(f"Bronze copy  : {'ENABLED' if COPY_RAW_TO_BRONZE else 'DISABLED'}")
    print(f"Spark master : {SPARK_MASTER}")
    print(f"Driver memory: {SPARK_DRIVER_MEMORY}")
    print(f"Batch limit  : {BATCH_SAMPLE_LIMIT:,} samples")
    print("=" * 78)

    # Confirm HDFS and check storage before starting the expensive scan.
    hdfs_run(["dfs", "-ls", f"{HDFS_URI}{HDFS_WAREHOUSE_ROOT}"])
    logger.info("HDFS connection confirmed: %s", HDFS_WAREHOUSE_ROOT)

    output_paths = [HDFS_SILVER_PATH, HDFS_GOLD_AUDIT_PATH, HDFS_GOLD_PAIRS_PATH]
    existing_outputs = [path for path in output_paths if hdfs_exists(path)]
    if existing_outputs:
        raise RuntimeError(
            "Safety stop: one or more final Race telemetry output paths already exist:\n"
            + "\n".join(existing_outputs)
            + "\nNo existing HDFS data was overwritten. Inspect these paths before rerunning."
        )

    local_free = shutil.disk_usage(PROJECT_ROOT).free
    hdfs_free = hdfs_remaining_bytes()
    print(f"Local disk free : {local_free / (1024 ** 3):.2f} GB")
    print(f"HDFS free       : {hdfs_free / (1024 ** 3):.2f} GB")

    if local_free < MIN_LOCAL_FREE_BYTES:
        raise RuntimeError(
            f"Only {local_free / (1024 ** 3):.2f} GB is free on the project drive. "
            f"V4 requires at least {MIN_LOCAL_FREE_GB:.1f} GB free for Spark runtime files. "
            "Free disk space before running."
        )
    if hdfs_free < MIN_HDFS_RESERVE_BYTES:
        raise RuntimeError(
            f"HDFS has only {hdfs_free / (1024 ** 3):.2f} GB free. "
            f"V4 reserves at least {MIN_HDFS_RESERVE_GB:.1f} GB and will not start."
        )

    discovered = discover_race_telemetry()
    if not discovered:
        raise RuntimeError(
            f"No Race telemetry files matching '*_tel.json' were found under {RAW_ROOT}."
        )

    events = sorted({(item["year"], item["event_name"]) for item in discovered})
    print(f"Race telemetry files discovered : {len(discovered):,}")
    print(f"Grand Prix events discovered    : {len(events):,}")
    print(f"Season folders scanned          : {', '.join(map(str, SEASONS))}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    run_root = f"{HDFS_WAREHOUSE_ROOT}/_staging/race_telemetry_v4/{run_id}"
    stage_silver_path = f"{run_root}/silver"
    stage_audit_path = f"{run_root}/gold/telemetry_audit"
    stage_pairs_path = f"{run_root}/gold/comparison_candidates"

    hdfs_run(["dfs", "-mkdir", "-p", f"{HDFS_URI}{run_root}"])

    spark_local_dir = SPARK_LOCAL_ROOT / "for1sight_race_telemetry_spark_v4"
    spark_local_dir.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("For1Sight_Grand_Prix_Race_Telemetry_Audit_V4")
        .master(SPARK_MASTER)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.driver.maxResultSize", "1g")
        .config("spark.default.parallelism", str(SPARK_WRITE_PARTITIONS))
        .config("spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTITIONS)
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.python.worker.reuse", "true")
        .config("spark.local.dir", str(spark_local_dir))
        .config("spark.hadoop.fs.defaultFS", HDFS_URI)
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    print(f"Spark version                    : {spark.version}")
    print(f"Spark application ID             : {spark.sparkContext.applicationId}")
    logger.info("Spark local directory: %s", spark_local_dir)
    logger.info("HDFS staging root    : %s", run_root)
    logger.info("V4 writes Silver batches directly to HDFS; no full local Silver copy is created.")

    staged_silver_uri = f"{HDFS_URI}{stage_silver_path}"
    staged_audit_uri = f"{HDFS_URI}{stage_audit_path}"
    staged_pairs_uri = f"{HDFS_URI}{stage_pairs_path}"

    audit_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    valid_sample_rows = 0
    valid_map_laps = 0
    valid_style_laps = 0
    invalid_laps = 0
    processed_files = 0
    batch_number = 0
    published_paths: list[str] = []

    def flush_sample_batch() -> None:
        nonlocal batch_rows, batch_number
        if not batch_rows:
            return

        row_count = len(batch_rows)
        batch_df = spark.createDataFrame(batch_rows, schema=SAMPLE_SCHEMA)

        # One output task per batch limits small files while keeping each batch
        # bounded in Python memory. Partition folders remain year/event_slug.
        (
            batch_df.coalesce(SPARK_WRITE_PARTITIONS)
            .write.mode("append")
            .option("compression", "snappy")
            .partitionBy("year", "event_slug")
            .parquet(staged_silver_uri)
        )

        batch_number += 1
        logger.info(
            "HDFS Silver batch %s written | samples=%s | staging=%s",
            batch_number, row_count, stage_silver_path
        )
        batch_rows = []
        del batch_df
        gc.collect()

    def process_one_file(meta: dict[str, Any]) -> None:
        nonlocal valid_sample_rows, valid_map_laps
        nonlocal valid_style_laps, invalid_laps, processed_files

        audit, flattened = audit_and_flatten_file(meta)
        audit_rows.append(audit)
        processed_files += 1

        if audit["track_map_usable"]:
            valid_map_laps += 1
            valid_sample_rows += len(flattened)
            batch_rows.extend(flattened)
        else:
            invalid_laps += 1

        if audit["driving_style_usable"]:
            valid_style_laps += 1

        if len(batch_rows) >= BATCH_SAMPLE_LIMIT:
            flush_sample_batch()

        del flattened
        if processed_files % 1000 == 0:
            gc.collect()

    def check_storage_capacity(checkpoint: str) -> None:
        """Stop safely before the HDFS DataNode volume runs out of space."""
        if valid_sample_rows <= 0 or processed_files <= 0:
            return

        current_silver_bytes = hdfs_path_size_bytes(stage_silver_path)
        free_bytes = hdfs_remaining_bytes()
        remaining_files = max(0, len(discovered) - processed_files)

        projected_remaining = (
            current_silver_bytes
            * remaining_files
            / max(1, processed_files)
            * HDFS_SPACE_SAFETY_MULTIPLIER
        )
        required_bytes = projected_remaining + MIN_HDFS_RESERVE_BYTES

        logger.info(
            "Storage checkpoint=%s | processed=%s/%s | staged_silver_GB=%.2f | "
            "HDFS_free_GB=%.2f | projected_remaining_GB=%.2f",
            checkpoint,
            processed_files,
            len(discovered),
            current_silver_bytes / (1024 ** 3),
            free_bytes / (1024 ** 3),
            projected_remaining / (1024 ** 3),
        )

        local_free_now = shutil.disk_usage(PROJECT_ROOT).free
        if local_free_now < MIN_LOCAL_FREE_BYTES:
            raise RuntimeError(
                f"Local project drive has only {local_free_now / (1024 ** 3):.2f} GB free "
                f"at checkpoint {checkpoint}. Stop and free local disk space. "
                f"Partial HDFS staging is preserved at {run_root}."
            )

        if required_bytes > free_bytes:
            LOCAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)
            write_csv(
                LOCAL_REPORT_DIR / "telemetry_quality_audit_partial.csv",
                audit_rows,
            )
            raise RuntimeError(
                f"Storage checkpoint {checkpoint} predicts insufficient HDFS capacity. "
                f"Projected remaining output: {projected_remaining / (1024 ** 3):.2f} GB; "
                f"HDFS currently free: {free_bytes / (1024 ** 3):.2f} GB; "
                f"required reserve: {MIN_HDFS_RESERVE_GB:.1f} GB. "
                f"Partial staging is preserved at {run_root}. "
                "Free disk space or move the HDFS DataNode storage directory before resuming."
            )

    try:
        # ------------------------------------------------------------------
        # Representative storage preflight: select files across all seasons
        # and events, rather than only the first event in alphabetical order.
        # ------------------------------------------------------------------
        pilot_count = min(PREFLIGHT_FILE_COUNT, len(discovered))
        if pilot_count == len(discovered):
            pilot_indices = list(range(len(discovered)))
        elif pilot_count == 1:
            pilot_indices = [len(discovered) // 2]
        else:
            pilot_indices = sorted({
                round(i * (len(discovered) - 1) / (pilot_count - 1))
                for i in range(pilot_count)
            })

        pilot_paths = {str(discovered[index]["path"]) for index in pilot_indices}
        logger.info(
            "Storage preflight started | representative files=%s",
            len(pilot_indices)
        )

        for index in pilot_indices:
            process_one_file(discovered[index])

        flush_sample_batch()

        pilot_size_bytes = hdfs_path_size_bytes(stage_silver_path)
        hdfs_free_after_pilot = hdfs_remaining_bytes()
        logger.info(
            "Storage preflight | pilot_files=%s | pilot_silver_bytes=%s | "
            "HDFS_free_after_pilot_GB=%.2f",
            len(pilot_indices),
            pilot_size_bytes,
            hdfs_free_after_pilot / (1024 ** 3),
        )

        if pilot_size_bytes > 0:
            estimated_remaining_bytes = (
                pilot_size_bytes
                * max(0, len(discovered) - len(pilot_indices))
                / max(1, len(pilot_indices))
                * HDFS_SPACE_SAFETY_MULTIPLIER
            )
            print(
                "Estimated remaining Silver storage : "
                f"{estimated_remaining_bytes / (1024 ** 3):.2f} GB "
                f"(includes {HDFS_SPACE_SAFETY_MULTIPLIER:.2f}x safety factor)"
            )
            print(
                f"HDFS free after preflight          : "
                f"{hdfs_free_after_pilot / (1024 ** 3):.2f} GB"
            )
            check_storage_capacity("representative pilot")
        else:
            logger.warning(
                "Pilot produced no map-usable samples; storage projection is not "
                "available yet. Continuing with periodic capacity checks."
            )
            if hdfs_free_after_pilot < MIN_HDFS_RESERVE_BYTES:
                hdfs_delete_tree(run_root)
                raise RuntimeError(
                    "HDFS free space fell below the configured reserve during preflight."
                )

        # Process all files not selected for the representative pilot.
        for meta in discovered:
            if str(meta["path"]) in pilot_paths:
                continue

            process_one_file(meta)

            if (
                processed_files % SPACE_CHECK_INTERVAL_FILES == 0
                and processed_files < len(discovered)
            ):
                flush_sample_batch()
                check_storage_capacity(f"after {processed_files:,} files")

            if processed_files % 100 == 0 or processed_files == len(discovered):
                logger.info(
                    "Audit progress %s/%s | map-usable laps=%s | "
                    "style-usable laps=%s | invalid=%s",
                    processed_files, len(discovered),
                    valid_map_laps, valid_style_laps, invalid_laps
                )

        flush_sample_batch()

        if processed_files != len(discovered):
            raise RuntimeError(
                f"File processing count mismatch: processed {processed_files:,} "
                f"of {len(discovered):,} discovered files."
            )
        if not audit_rows:
            raise RuntimeError("Telemetry audit produced no file-level results.")

        pair_rows = build_comparison_pairs(audit_rows)

        # Save local CSV audit reports. These are small diagnostic reports;
        # HDFS Parquet remains the analytical warehouse.
        if EXPORT_LOCAL_CSV:
            LOCAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)
            write_csv(LOCAL_REPORT_DIR / "telemetry_quality_audit.csv", audit_rows)
            write_csv(LOCAL_REPORT_DIR / "valid_comparison_pairs.csv", pair_rows)
            logger.info("Local CSV reports saved to: %s", LOCAL_REPORT_DIR)

        if COPY_RAW_TO_BRONZE:
            copy_raw_race_folders_to_bronze(discovered)

        # ------------------------------------------------------------------
        # Write Gold outputs to the same temporary HDFS run directory.
        # ------------------------------------------------------------------
        print("\n" + "=" * 78)
        print("WRITING GOLD AUDIT RESULTS TO HDFS STAGING")
        print("=" * 78)

        audit_df = spark.createDataFrame(audit_rows, schema=AUDIT_SCHEMA)
        (
            audit_df.coalesce(1).write
            .mode("errorifexists")
            .option("compression", "snappy")
            .partitionBy("year")
            .parquet(staged_audit_uri)
        )
        del audit_df
        gc.collect()

        if pair_rows:
            pairs_df = spark.createDataFrame(pair_rows, schema=PAIR_SCHEMA)
            (
                pairs_df.coalesce(1).write
                .mode("errorifexists")
                .option("compression", "snappy")
                .partitionBy("year")
                .parquet(staged_pairs_uri)
            )
            del pairs_df
            gc.collect()

        # ------------------------------------------------------------------
        # Read-back validation BEFORE publishing final warehouse paths.
        # ------------------------------------------------------------------
        print("\n" + "=" * 78)
        print("HDFS STAGING READ-BACK VALIDATION")
        print("=" * 78)

        if valid_sample_rows > 0:
            staged_silver_df = spark.read.parquet(staged_silver_uri)
            silver_count = staged_silver_df.count()
            silver_columns = len(staged_silver_df.columns)
        else:
            silver_count = 0
            silver_columns = len(SAMPLE_SCHEMA.fields)

        staged_audit_df = spark.read.parquet(staged_audit_uri)
        audit_count = staged_audit_df.count()

        if pair_rows:
            staged_pairs_df = spark.read.parquet(staged_pairs_uri)
            pair_count = staged_pairs_df.count()
        else:
            pair_count = 0

        print(f"Expected Silver sample rows : {valid_sample_rows:,}")
        print(f"Actual Silver sample rows   : {silver_count:,}")
        print(f"Expected audit file rows    : {len(audit_rows):,}")
        print(f"Actual audit file rows      : {audit_count:,}")
        print(f"Comparison candidate pairs  : {pair_count:,}")
        print(f"Silver columns              : {silver_columns}")
        print(f"Audit columns               : {len(staged_audit_df.columns)}")

        if silver_count != valid_sample_rows or audit_count != len(audit_rows):
            raise RuntimeError(
                "HDFS staging read-back validation failed. "
                "Final warehouse paths have not been published."
            )
        if pair_rows and pair_count != len(pair_rows):
            raise RuntimeError(
                "HDFS comparison-pair validation failed. "
                "Final warehouse paths have not been published."
            )

        # ------------------------------------------------------------------
        # Publish by HDFS rename. The temporary and final directories are on
        # the same HDFS filesystem, so each directory move is atomic.
        # ------------------------------------------------------------------
        existing_outputs = [path for path in output_paths if hdfs_exists(path)]
        if existing_outputs:
            raise RuntimeError(
                "Safety stop before publishing; a final output path appeared:\n"
                + "\n".join(existing_outputs)
                + "\nStaging data has been preserved."
            )

        hdfs_run(["dfs", "-mkdir", "-p", f"{HDFS_URI}{HDFS_WAREHOUSE_ROOT}/silver"])
        hdfs_run(["dfs", "-mkdir", "-p", f"{HDFS_URI}{HDFS_WAREHOUSE_ROOT}/gold/race_telemetry"])

        if valid_sample_rows > 0:
            hdfs_move(stage_silver_path, HDFS_SILVER_PATH)
            published_paths.append(HDFS_SILVER_PATH)

        hdfs_move(stage_audit_path, HDFS_GOLD_AUDIT_PATH)
        published_paths.append(HDFS_GOLD_AUDIT_PATH)

        if pair_rows:
            hdfs_move(stage_pairs_path, HDFS_GOLD_PAIRS_PATH)
            published_paths.append(HDFS_GOLD_PAIRS_PATH)

        # Confirm that the published paths now exist.
        missing_published = [path for path in published_paths if not hdfs_exists(path)]
        if missing_published:
            raise RuntimeError(
                "One or more published HDFS paths could not be confirmed:\n"
                + "\n".join(missing_published)
            )

        print("\n" + "=" * 78)
        print("FOR1SIGHT RACE TELEMETRY AUDIT COMPLETED SUCCESSFULLY")
        print("=" * 78)
        print(f"Files scanned               : {processed_files:,}")
        print(f"Grand Prix events           : {len(events):,}")
        print(f"Track-map usable laps       : {valid_map_laps:,}")
        print(f"Driving-style usable laps   : {valid_style_laps:,}")
        print(f"Invalid laps                : {invalid_laps:,}")
        print(f"Silver telemetry samples    : {silver_count:,}")
        print(f"Comparison candidate pairs  : {pair_count:,}")
        print("\nPublished HDFS destinations:")
        for path in published_paths:
            print(f"  {path}")
        if valid_sample_rows == 0:
            print(
                "\nWARNING: No track-map-usable samples were found. "
                "Gold audit is available, but Silver telemetry was not created."
            )
        if EXPORT_LOCAL_CSV:
            print(f"\nLocal reports: {LOCAL_REPORT_DIR}")

        # Remove only this run's now-empty staging root.
        hdfs_delete_tree(run_root, ignore_errors=True)

    except Exception:
        logger.exception(
            "V4 pipeline failed. Temporary HDFS staging has been preserved at: %s",
            run_root,
        )
        print("\nThe existing final warehouse datasets were not intentionally overwritten.")
        print(f"Inspect this run's temporary HDFS path: {run_root}")
        raise
    finally:
        try:
            spark.stop()
        except Exception as stop_error:
            logger.warning("Spark shutdown reported an additional cleanup error: %s", stop_error)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Source JSON and existing warehouse files remain unchanged.")
        raise
    except Exception as error:
        logger.exception("Race telemetry audit failed: %s", error)
        raise
