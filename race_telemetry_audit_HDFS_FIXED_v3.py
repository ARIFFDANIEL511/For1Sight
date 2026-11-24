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
- Existing HDFS output is NOT overwritten unless ALLOW_OVERWRITE_OUTPUTS=True.
- HDFS path checks use `hdfs dfs -ls -d` output, not unreliable exit codes from `-test -e`.
- Raw Bronze copying is optional and disabled by default to protect HDFS capacity.

Run from the For1Sight project folder:
    python race_telemetry_audit.py

Requirements:
    pyspark, Java, Hadoop/HDFS configured and running.
"""

from __future__ import annotations

import csv
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

# Safety lock. If one of the dedicated Silver/Gold output paths already exists,
# the script stops instead of overwriting it. Change to True only when you intend
# to replace ONLY these new Race telemetry outputs.
ALLOW_OVERWRITE_OUTPUTS = False

# PySpark performance settings suitable for a local Windows development machine.
SPARK_MASTER = "local[4]"
SPARK_DRIVER_MEMORY = "4g"
SPARK_SHUFFLE_PARTITIONS = "8"
BATCH_SAMPLE_LIMIT = 50000
FINAL_OUTPUT_PARTITIONS = 12

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


# ============================================================================
# 8. MAIN PIPELINE
# ============================================================================


def main() -> None:
    print("=" * 78)
    print("FOR1SIGHT — GRAND PRIX RACE TELEMETRY AUDIT & HDFS LOADER")
    print("=" * 78)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Raw root     : {RAW_ROOT}")
    print(f"Seasons      : {SEASONS}")
    print("Sessions     : Race only")
    print(f"HDFS         : {HDFS_URI}")
    print(f"Bronze copy  : {'ENABLED' if COPY_RAW_TO_BRONZE else 'DISABLED (safe first run)'}")
    print("=" * 78)

    # Validate HDFS connectivity before spending time reading all JSON files.
    hdfs_run(["dfs", "-ls", f"{HDFS_URI}{HDFS_WAREHOUSE_ROOT}"])
    logger.info("HDFS connection confirmed: %s", HDFS_WAREHOUSE_ROOT)

    output_paths = [HDFS_SILVER_PATH, HDFS_GOLD_AUDIT_PATH, HDFS_GOLD_PAIRS_PATH]
    existing_outputs = [path for path in output_paths if hdfs_exists(path)]
    if existing_outputs and not ALLOW_OVERWRITE_OUTPUTS:
        raise RuntimeError(
            "Safety stop: one or more dedicated Race telemetry output paths already exist:\n"
            + "\n".join(existing_outputs)
            + "\nNo HDFS data was overwritten. If these are previous outputs from this same "
              "pipeline and you intentionally want to replace them, review the paths and "
              "set ALLOW_OVERWRITE_OUTPUTS=True in the configuration."
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

    # Build Spark only after confirming the input data and HDFS connection.
    spark_local_dir = SPARK_LOCAL_ROOT / "for1sight_race_telemetry_spark"
    spark_local_dir.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("For1Sight_Grand_Prix_Race_Telemetry_Audit")
        .master(SPARK_MASTER)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTITIONS)
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.local.dir", str(spark_local_dir))
        .config("spark.hadoop.fs.defaultFS", HDFS_URI)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print(f"Spark version                    : {spark.version}")
    print(f"Spark application ID             : {spark.sparkContext.applicationId}")

    # PySpark uses this private directory when serializing local Python data
    # for createDataFrame(). Keep it available throughout the long audit run.
    spark_python_temp_dir = Path(spark.sparkContext._temp_dir)
    spark_python_temp_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Stable runtime temp root : %s", PYSPARK_TEMP_ROOT)
    logger.info("Spark local directory    : %s", spark_local_dir)
    logger.info("PySpark context temp dir : %s", spark_python_temp_dir)

    staging_dir = Path(tempfile.mkdtemp(prefix="for1sight_race_telemetry_stage_"))
    logger.info("Local Parquet staging dir: %s", staging_dir)
    staged_silver_path = staging_dir / "silver_samples"
    # Spark uses HDFS as fs.defaultFS. Use an explicit local file URI
    # so Windows paths are not misread as HDFS paths.
    staged_silver_uri = staged_silver_path.resolve().as_uri()
    audit_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    valid_sample_rows = 0
    valid_map_laps = 0
    valid_style_laps = 0
    invalid_laps = 0
    processed_files = 0
    batch_number = 0

    def flush_sample_batch() -> None:
        nonlocal batch_rows, batch_number
        if not batch_rows:
            return
        # Recreate the directory if Windows or another cleanup process removed
        # it during this long-running job.
        spark_python_temp_dir.mkdir(parents=True, exist_ok=True)
        batch_df = spark.createDataFrame(batch_rows, schema=SAMPLE_SCHEMA)
        batch_df.write.mode("append").option("compression", "snappy").parquet(
            staged_silver_uri
        )
        batch_number += 1
        logger.info(
            "Local staging batch %s written | samples=%s",
            batch_number, len(batch_rows)
        )
        batch_rows = []

    try:
        for index, meta in enumerate(discovered, start=1):
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

            if index % 100 == 0 or index == len(discovered):
                logger.info(
                    "Audit progress %s/%s | map-usable laps=%s | style-usable laps=%s | invalid=%s",
                    index, len(discovered), valid_map_laps, valid_style_laps, invalid_laps
                )

        flush_sample_batch()

        if not audit_rows:
            raise RuntimeError("Telemetry audit produced no file-level results.")
        if valid_sample_rows == 0 or not staged_silver_path.exists():
            raise RuntimeError(
                "No telemetry samples passed the track-map quality checks. "
                "Review the local audit report before attempting an HDFS export."
            )

        pair_rows = build_comparison_pairs(audit_rows)

        # Save small local CSV diagnostics. These are convenience reports only;
        # HDFS Parquet remains the main analytical storage.
        if EXPORT_LOCAL_CSV:
            LOCAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)
            write_csv(LOCAL_REPORT_DIR / "telemetry_quality_audit.csv", audit_rows)
            write_csv(LOCAL_REPORT_DIR / "valid_comparison_pairs.csv", pair_rows)
            logger.info("Local CSV reports saved to: %s", LOCAL_REPORT_DIR)

        # Optional Bronze copy. This copies raw Race folders without changing them.
        if COPY_RAW_TO_BRONZE:
            copy_raw_race_folders_to_bronze(discovered)

        # Recheck HDFS output paths immediately before writing.
        existing_outputs = [path for path in output_paths if hdfs_exists(path)]
        if existing_outputs and not ALLOW_OVERWRITE_OUTPUTS:
            raise RuntimeError(
                "Safety stop before HDFS write; output path appeared during processing:\n"
                + "\n".join(existing_outputs)
            )

        write_mode = "overwrite" if ALLOW_OVERWRITE_OUTPUTS else "errorifexists"

        print("\n" + "=" * 78)
        print("WRITING SILVER TELEMETRY TO HDFS")
        print("=" * 78)
        silver_df = spark.read.parquet(staged_silver_uri)
        silver_df = silver_df.repartition(
            FINAL_OUTPUT_PARTITIONS, "year", "event_slug"
        )
        (
            silver_df.write
            .mode(write_mode)
            .option("compression", "snappy")
            .partitionBy("year", "event_slug")
            .parquet(f"{HDFS_URI}{HDFS_SILVER_PATH}")
        )
        logger.info("Silver telemetry written: %s", HDFS_SILVER_PATH)

        print("\n" + "=" * 78)
        print("WRITING GOLD AUDIT RESULTS TO HDFS")
        print("=" * 78)
        spark_python_temp_dir.mkdir(parents=True, exist_ok=True)
        audit_df = spark.createDataFrame(audit_rows, schema=AUDIT_SCHEMA)
        (
            audit_df.coalesce(4).write
            .mode(write_mode)
            .option("compression", "snappy")
            .partitionBy("year")
            .parquet(f"{HDFS_URI}{HDFS_GOLD_AUDIT_PATH}")
        )

        if pair_rows:
            spark_python_temp_dir.mkdir(parents=True, exist_ok=True)
            pairs_df = spark.createDataFrame(pair_rows, schema=PAIR_SCHEMA)
            (
                pairs_df.coalesce(4).write
                .mode(write_mode)
                .option("compression", "snappy")
                .partitionBy("year")
                .parquet(f"{HDFS_URI}{HDFS_GOLD_PAIRS_PATH}")
            )
            logger.info("Gold comparison candidates written: %s", HDFS_GOLD_PAIRS_PATH)
        else:
            logger.warning(
                "No same-event/same-lap driving-style comparison pairs were found. "
                "The empty candidate dataset will not be written."
            )
        logger.info("Gold audit written: %s", HDFS_GOLD_AUDIT_PATH)

        # Read-back validation from HDFS.
        print("\n" + "=" * 78)
        print("HDFS READ-BACK VALIDATION")
        print("=" * 78)
        hdfs_silver_df = spark.read.parquet(f"{HDFS_URI}{HDFS_SILVER_PATH}")
        hdfs_audit_df = spark.read.parquet(f"{HDFS_URI}{HDFS_GOLD_AUDIT_PATH}")
        silver_count = hdfs_silver_df.count()
        audit_count = hdfs_audit_df.count()
        if pair_rows:
            hdfs_pairs_df = spark.read.parquet(f"{HDFS_URI}{HDFS_GOLD_PAIRS_PATH}")
            pair_count = hdfs_pairs_df.count()
        else:
            pair_count = 0
        print(f"Expected Silver sample rows : {valid_sample_rows:,}")
        print(f"Actual Silver sample rows   : {silver_count:,}")
        print(f"Silver row count matches    : {silver_count == valid_sample_rows}")
        print(f"Expected audit file rows    : {len(audit_rows):,}")
        print(f"Actual audit file rows      : {audit_count:,}")
        print(f"Audit row count matches     : {audit_count == len(audit_rows)}")
        print(f"Comparison candidate pairs  : {pair_count:,}")
        print(f"Silver columns              : {len(hdfs_silver_df.columns)}")
        print(f"Audit columns               : {len(hdfs_audit_df.columns)}")

        if silver_count != valid_sample_rows or audit_count != len(audit_rows):
            raise RuntimeError("HDFS read-back validation failed: row counts do not match.")

        print("\n" + "=" * 78)
        print("FOR1SIGHT RACE TELEMETRY AUDIT COMPLETED SUCCESSFULLY")
        print("=" * 78)
        print(f"Files scanned               : {processed_files:,}")
        print(f"Track-map usable laps       : {valid_map_laps:,}")
        print(f"Driving-style usable laps   : {valid_style_laps:,}")
        print(f"Invalid laps                : {invalid_laps:,}")
        print(f"Silver telemetry samples    : {silver_count:,}")
        print(f"Comparison candidate pairs  : {pair_count:,}")
        print("\nHDFS destinations:")
        print(f"  Silver : {HDFS_SILVER_PATH}")
        print(f"  Gold audit : {HDFS_GOLD_AUDIT_PATH}")
        if pair_rows:
            print(f"  Gold pairs : {HDFS_GOLD_PAIRS_PATH}")
        else:
            print("  Gold pairs : not created (no valid common-lap pairs found)")
        if COPY_RAW_TO_BRONZE:
            print(f"  Bronze : {HDFS_BRONZE_PATH}")
        if EXPORT_LOCAL_CSV:
            print(f"\nLocal reports: {LOCAL_REPORT_DIR}")

        # Remove only this script's temporary local staging directory after success.
        shutil.rmtree(staging_dir, ignore_errors=True)

    finally:
        spark.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Source JSON and existing warehouse files remain unchanged.")
        raise
    except Exception as error:
        logger.exception("Race telemetry audit failed: %s", error)
        raise
