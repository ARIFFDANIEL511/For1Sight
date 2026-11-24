"""For1Sight — Race Telemetry Reader (HDFS + cloud deployment).

Reads the existing Silver and Gold Race telemetry Parquet datasets.
This module does not download raw data, run ETL, or write/modify warehouse files.

Two interchangeable storage back-ends return identical pandas DataFrames:

* "hdfs"  - local Big Data environment: PySpark reads Silver/Gold from HDFS.
* "cloud" - online deployment (no Java/Spark/HDFS available):
            Gold tables  -> small Parquet snapshot inside the repository
                            (for1sight_warehouse/telemetry_gold/...)
            Silver table -> full 4 GB Parquet dataset hosted on a Hugging Face
                            dataset repository; only the one year/event
                            partition that the user selects is downloaded.

Back-end selection (FOR1SIGHT_TELEMETRY_BACKEND):
    auto (default) - use HDFS when PySpark is installed, and fall back to the
                     cloud snapshot if HDFS cannot be reached; use the cloud
                     snapshot directly when PySpark is not installed.
    hdfs / cloud   - force one back-end.

Settings (environment variable or Streamlit secret, all optional):
    FOR1SIGHT_TELEMETRY_BACKEND: auto | hdfs | cloud
    FOR1SIGHT_HDFS_URI: default hdfs://localhost:9000
    FOR1SIGHT_TELEMETRY_SILVER_PATH: default /for1sight_warehouse/silver/race_telemetry
    FOR1SIGHT_TELEMETRY_AUDIT_PATH: default /for1sight_warehouse/gold/race_telemetry/telemetry_audit
    FOR1SIGHT_TELEMETRY_PAIRS_PATH: default /for1sight_warehouse/gold/race_telemetry/comparison_candidates
    FOR1SIGHT_HF_DATASET: Hugging Face dataset repo id, e.g. "username/for1sight-race-telemetry"
    FOR1SIGHT_HF_SILVER_DIR: folder inside that repo, default "race_telemetry"
    HF_TOKEN: only needed if the Hugging Face dataset is private
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from pyspark.sql import SparkSession, functions as F
except ImportError as exc:  # pragma: no cover - depends on the user's runtime
    SparkSession = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _PYSPARK_IMPORT_ERROR = exc
else:
    _PYSPARK_IMPORT_ERROR = None


def _setting(name: str, default: str = "") -> str:
    """Read a setting from the environment first, then from Streamlit secrets."""
    value = os.getenv(name)
    if value:
        return value
    try:
        return str(st.secrets.get(name, default))
    except Exception:  # no secrets.toml present
        return default


HDFS_URI = _setting("FOR1SIGHT_HDFS_URI", "hdfs://localhost:9000").rstrip("/")
SILVER_PATH = _setting(
    "FOR1SIGHT_TELEMETRY_SILVER_PATH",
    "/for1sight_warehouse/silver/race_telemetry",
)
AUDIT_PATH = _setting(
    "FOR1SIGHT_TELEMETRY_AUDIT_PATH",
    "/for1sight_warehouse/gold/race_telemetry/telemetry_audit",
)
PAIRS_PATH = _setting(
    "FOR1SIGHT_TELEMETRY_PAIRS_PATH",
    "/for1sight_warehouse/gold/race_telemetry/comparison_candidates",
)

BACKEND = _setting("FOR1SIGHT_TELEMETRY_BACKEND", "auto").strip().lower()
PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_GOLD_DIR = PROJECT_ROOT / "for1sight_warehouse" / "telemetry_gold"
LOCAL_AUDIT_DIR = LOCAL_GOLD_DIR / "telemetry_audit"
LOCAL_PAIRS_DIR = LOCAL_GOLD_DIR / "comparison_candidates"
HF_DATASET = _setting("FOR1SIGHT_HF_DATASET", "").strip()
HF_SILVER_DIR = _setting("FOR1SIGHT_HF_SILVER_DIR", "race_telemetry").strip().strip("/")

SAMPLE_COLUMNS = [
    "year", "event_name", "event_slug", "session_type", "driver_code",
    "lap_number", "sample_index", "time_s", "rpm", "speed_kmh", "gear",
    "throttle_pct", "brake", "drs", "distance_m", "rel_distance",
    "rel_distance_source", "x", "y", "acc_x", "acc_y", "acc_z",
    "driver_ahead", "distance_to_driver_ahead_m", "driving_style_usable",
]


class TelemetryDataError(RuntimeError):
    """Raised when the telemetry datasets cannot be read or queried."""


def _hdfs_location(path: str) -> str:
    return f"{HDFS_URI}{path if path.startswith('/') else '/' + path}"


@st.cache_resource(show_spinner=False)
def get_telemetry_spark() -> "SparkSession":
    """Create one reusable local Spark session for the Streamlit process."""
    if _PYSPARK_IMPORT_ERROR is not None or SparkSession is None:
        raise TelemetryDataError(
            "PySpark is not installed in the Python environment running Streamlit. "
            "Install the same PySpark version used by the For1Sight ETL environment."
        ) from _PYSPARK_IMPORT_ERROR

    try:
        spark = (
            SparkSession.builder
            .appName("For1Sight_Streamlit_Race_Telemetry")
            .master("local[2]")
            .config("spark.hadoop.fs.defaultFS", HDFS_URI)
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.sql.parquet.compression.codec", "snappy")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")
        return spark
    except Exception as exc:
        raise TelemetryDataError(
            f"Could not start the Spark session for HDFS URI {HDFS_URI}. "
            "Check Java, PySpark and Hadoop configuration."
        ) from exc


def _read_parquet(path: str):
    spark = get_telemetry_spark()
    uri = _hdfs_location(path)
    try:
        return spark.read.parquet(uri)
    except Exception as exc:
        raise TelemetryDataError(
            f"Unable to read telemetry Parquet dataset: {uri}. "
            "Confirm that HDFS is running and the ETL output has been published."
        ) from exc


def _ordered_pair(driver_a: str, driver_b: str) -> tuple[str, str]:
    a = str(driver_a).strip().upper()
    b = str(driver_b).strip().upper()
    if not a or not b or a == b:
        raise TelemetryDataError("Select two different valid driver codes.")
    return tuple(sorted((a, b)))  # pair table stores combinations alphabetically


def _hdfs_telemetry_pair_candidates(
    year: int, driver_a: str, driver_b: str
) -> pd.DataFrame:
    """Return same-event Race pairs with common driving-style-valid laps."""
    a, b = _ordered_pair(driver_a, driver_b)
    try:
        frame = (
            _read_parquet(PAIRS_PATH)
            .filter(
                (F.col("year") == int(year))
                & (F.col("session_type") == "Race")
                & (F.col("driver_a") == a)
                & (F.col("driver_b") == b)
            )
            .select(
                "year", "event_name", "event_slug", "session_type",
                "driver_a", "driver_b", "common_lap_count", "common_laps",
            )
            .orderBy(F.col("common_lap_count").desc(), F.col("event_name").asc())
            .toPandas()
        )
        return frame
    except TelemetryDataError:
        raise
    except Exception as exc:
        raise TelemetryDataError("Failed while filtering Gold telemetry comparison candidates.") from exc


def _hdfs_telemetry_audit(
    year: int,
    event_slug: str,
    driver_a: str,
    driver_b: str,
    lap_number: int,
) -> pd.DataFrame:
    """Return the Gold audit quality records for the selected driver/lap pair."""
    codes = sorted({str(driver_a).strip().upper(), str(driver_b).strip().upper()})
    try:
        return (
            _read_parquet(AUDIT_PATH)
            .filter(
                (F.col("year") == int(year))
                & (F.col("event_slug") == str(event_slug))
                & (F.col("session_type") == "Race")
                & (F.col("lap_number") == int(lap_number))
                & F.col("driver_code").isin(codes)
            )
            .select(
                "driver_code", "lap_number", "sample_count", "quality_status",
                "track_map_usable", "driving_style_usable", "rel_distance_source",
                "missing_fields", "issues",
            )
            .orderBy("driver_code")
            .toPandas()
        )
    except TelemetryDataError:
        raise
    except Exception as exc:
        raise TelemetryDataError("Failed while reading the Gold telemetry quality audit.") from exc


def _hdfs_telemetry_lap_samples(
    year: int,
    event_slug: str,
    driver_a: str,
    driver_b: str,
    lap_number: int,
) -> pd.DataFrame:
    """Read only the two selected drivers' validated samples for one common lap."""
    codes = sorted({str(driver_a).strip().upper(), str(driver_b).strip().upper()})
    try:
        source = _read_parquet(SILVER_PATH)
        available = set(source.columns)
        missing = sorted(set(SAMPLE_COLUMNS) - available)
        if missing:
            raise TelemetryDataError(
                "Silver telemetry schema is missing expected columns: " + ", ".join(missing)
            )
        selected = [column for column in SAMPLE_COLUMNS if column in available]
        return (
            source
            .filter(
                (F.col("year") == int(year))
                & (F.col("event_slug") == str(event_slug))
                & (F.col("session_type") == "Race")
                & (F.col("lap_number") == int(lap_number))
                & F.col("driver_code").isin(codes)
                & (F.col("driving_style_usable") == True)  # noqa: E712
            )
            .select(*selected)
            .orderBy("driver_code", "sample_index")
            .toPandas()
        )
    except TelemetryDataError:
        raise
    except Exception as exc:
        raise TelemetryDataError("Failed while loading the selected Silver telemetry lap samples.") from exc


# ---------------------------------------------------------------------------
# Cloud back-end (pandas + pyarrow, no Spark/Java/HDFS)
# ---------------------------------------------------------------------------
def _cloud_available() -> bool:
    return LOCAL_PAIRS_DIR.exists() and LOCAL_AUDIT_DIR.exists()


def _read_local_gold(path: Path, year: int) -> pd.DataFrame:
    if not path.exists():
        raise TelemetryDataError(
            f"Gold telemetry snapshot not found: {path}. Export it from HDFS into "
            "for1sight_warehouse/telemetry_gold/ before deploying."
        )
    try:
        frame = pd.read_parquet(path, engine="pyarrow", filters=[("year", "=", int(year))])
    except Exception as exc:
        raise TelemetryDataError(f"Unable to read Gold telemetry snapshot: {path}.") from exc
    if "year" in frame.columns:  # hive partition column arrives as a category
        frame["year"] = pd.to_numeric(frame["year"].astype(str), errors="coerce").astype("Int64")
    return frame


def _cloud_pair_candidates(year: int, driver_a: str, driver_b: str) -> pd.DataFrame:
    a, b = _ordered_pair(driver_a, driver_b)
    frame = _read_local_gold(LOCAL_PAIRS_DIR, year)
    frame = frame[
        (frame["session_type"] == "Race")
        & (frame["driver_a"] == a)
        & (frame["driver_b"] == b)
    ]
    columns = ["year", "event_name", "event_slug", "session_type",
               "driver_a", "driver_b", "common_lap_count", "common_laps"]
    return (
        frame[columns]
        .sort_values(["common_lap_count", "event_name"], ascending=[False, True])
        .reset_index(drop=True)
    )


def _cloud_audit(year: int, event_slug: str, driver_a: str, driver_b: str, lap_number: int) -> pd.DataFrame:
    codes = sorted({str(driver_a).strip().upper(), str(driver_b).strip().upper()})
    frame = _read_local_gold(LOCAL_AUDIT_DIR, year)
    frame = frame[
        (frame["event_slug"] == str(event_slug))
        & (frame["session_type"] == "Race")
        & (frame["lap_number"] == int(lap_number))
        & (frame["driver_code"].isin(codes))
    ]
    columns = ["driver_code", "lap_number", "sample_count", "quality_status",
               "track_map_usable", "driving_style_usable", "rel_distance_source",
               "missing_fields", "issues"]
    return frame[columns].sort_values("driver_code").reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def _download_silver_partition(year: int, event_slug: str) -> str:
    """Download (once per server) the Silver partition for one race from Hugging Face."""
    if not HF_DATASET:
        raise TelemetryDataError(
            "FOR1SIGHT_HF_DATASET is not set, so the cloud Silver telemetry cannot be located."
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise TelemetryDataError("huggingface_hub is not installed (add it to requirements.txt).") from exc

    partition = f"{HF_SILVER_DIR}/year={int(year)}/event_slug={event_slug}"
    token = _setting("HF_TOKEN", "") or None
    try:
        local_root = snapshot_download(
            repo_id=HF_DATASET,
            repo_type="dataset",
            allow_patterns=[f"{partition}/*.parquet"],
            token=token,
        )
    except Exception as exc:
        raise TelemetryDataError(
            f"Could not download Silver telemetry partition {partition} from {HF_DATASET}."
        ) from exc

    folder = Path(local_root) / partition
    if not folder.exists() or not any(folder.glob("*.parquet")):
        raise TelemetryDataError(f"No Silver telemetry files found for {partition} in {HF_DATASET}.")
    return str(folder)


def _cloud_lap_samples(year: int, event_slug: str, driver_a: str, driver_b: str, lap_number: int) -> pd.DataFrame:
    import pyarrow.dataset as ds

    codes = sorted({str(driver_a).strip().upper(), str(driver_b).strip().upper()})
    folder = _download_silver_partition(int(year), str(event_slug))
    try:
        dataset = ds.dataset(folder, format="parquet")
        file_columns = set(dataset.schema.names)
        # year and event_slug are folder (partition) names, not stored inside the files
        missing = sorted(set(SAMPLE_COLUMNS) - file_columns - {"year", "event_slug"})
        if missing:
            raise TelemetryDataError(
                "Silver telemetry schema is missing expected columns: " + ", ".join(missing)
            )
        expression = (
            (ds.field("session_type") == "Race")
            & (ds.field("lap_number") == int(lap_number))
            & ds.field("driver_code").isin(codes)
            & (ds.field("driving_style_usable") == True)  # noqa: E712
        )
        read_columns = [c for c in SAMPLE_COLUMNS if c in file_columns]
        frame = dataset.to_table(columns=read_columns, filter=expression).to_pandas()
    except TelemetryDataError:
        raise
    except Exception as exc:
        raise TelemetryDataError("Failed while loading the selected Silver telemetry lap samples.") from exc

    if "year" not in frame.columns:
        frame["year"] = int(year)
    if "event_slug" not in frame.columns:
        frame["event_slug"] = str(event_slug)
    return (
        frame[[c for c in SAMPLE_COLUMNS if c in frame.columns]]
        .sort_values(["driver_code", "sample_index"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Public API used by pages/3_Driver_Comparison.py (names and outputs unchanged)
# ---------------------------------------------------------------------------
def _run(hdfs_call, cloud_call):
    """Run the query on the selected back-end, falling back to cloud in auto mode."""
    use_hdfs = BACKEND == "hdfs" or (BACKEND == "auto" and _PYSPARK_IMPORT_ERROR is None)
    if not use_hdfs:
        return cloud_call()
    try:
        return hdfs_call()
    except TelemetryDataError:
        if BACKEND == "auto" and _cloud_available():
            return cloud_call()
        raise


def active_backend() -> str:
    """Short label of the storage back-end that queries will try first."""
    if BACKEND == "hdfs" or (BACKEND == "auto" and _PYSPARK_IMPORT_ERROR is None):
        return "HDFS"
    return "Cloud Parquet snapshot"


@st.cache_data(ttl=600, show_spinner=False)
def load_telemetry_pair_candidates(year: int, driver_a: str, driver_b: str) -> pd.DataFrame:
    """Return same-event Race pairs with common driving-style-valid laps."""
    return _run(
        lambda: _hdfs_telemetry_pair_candidates(year, driver_a, driver_b),
        lambda: _cloud_pair_candidates(year, driver_a, driver_b),
    )


@st.cache_data(ttl=600, show_spinner=False)
def load_telemetry_audit(year: int, event_slug: str, driver_a: str, driver_b: str, lap_number: int) -> pd.DataFrame:
    """Return the Gold audit quality records for the selected driver/lap pair."""
    return _run(
        lambda: _hdfs_telemetry_audit(year, event_slug, driver_a, driver_b, lap_number),
        lambda: _cloud_audit(year, event_slug, driver_a, driver_b, lap_number),
    )


@st.cache_data(ttl=600, show_spinner=False)
def load_telemetry_lap_samples(year: int, event_slug: str, driver_a: str, driver_b: str, lap_number: int) -> pd.DataFrame:
    """Read only the two selected drivers' validated samples for one common lap."""
    return _run(
        lambda: _hdfs_telemetry_lap_samples(year, event_slug, driver_a, driver_b, lap_number),
        lambda: _cloud_lap_samples(year, event_slug, driver_a, driver_b, lap_number),
    )


def clear_telemetry_cache() -> None:
    """Clear only this module's cached telemetry query results."""
    load_telemetry_pair_candidates.clear()
    load_telemetry_audit.clear()
    load_telemetry_lap_samples.clear()
