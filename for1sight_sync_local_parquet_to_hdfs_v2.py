"""
For1Sight: safe local Parquet -> HDFS Silver synchronisation (Windows v2).

Fixes Windows paths containing spaces by staging a read-only copy of the local
Parquet dataset into a temporary path without spaces before Spark reads it.

Run from the For1Sight project root:
    python for1sight_sync_local_parquet_to_hdfs_v2.py

Safety:
- Original local Parquet dataset is never modified.
- Existing legacy HDFS path is only read, never overwritten.
- New HDFS destination is created only if it does not already exist.
- HDFS read-back row count must match the local source row count.
"""
from pathlib import Path
import os
import shutil
import sys
import tempfile
from pyspark.sql import SparkSession

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_PARQUET = PROJECT_ROOT / "for1sight_warehouse" / "hf_laptimes.parquet"

HDFS_DEFAULT_FS = "hdfs://localhost:9000"
LEGACY_HDFS_PATH = (
    "hdfs://localhost:9000/user/for1sight/warehouse/hf_laptimes.parquet"
)
NEW_HDFS_PATH = (
    "hdfs://localhost:9000/for1sight_warehouse/silver/hf_laptimes"
)

REQUIRED_COLUMNS = {
    "year", "session_type", "event_name", "driver_code", "lap", "time"
}


def heading(message):
    print("\n" + "=" * 72)
    print(message)
    print("=" * 72)


def hdfs_exists(spark, uri):
    conf = spark.sparkContext._jsc.hadoopConfiguration()
    jvm = spark._jvm
    path = jvm.org.apache.hadoop.fs.Path(uri)
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(
        jvm.java.net.URI(uri), conf
    )
    return bool(fs.exists(path))


def get_size_bytes(path):
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def stage_without_spaces(source):
    """
    Copy source to a temporary Windows path without spaces.
    The original local dataset remains untouched.
    """
    temp_root = Path(tempfile.gettempdir()).resolve()
    if " " in str(temp_root):
        temp_root = Path(r"C:\for1sight_spark_stage")
    temp_root.mkdir(parents=True, exist_ok=True)

    stage_dir = Path(tempfile.mkdtemp(prefix="for1sight_", dir=str(temp_root)))
    stage_source = stage_dir / source.name

    if source.is_dir():
        shutil.copytree(source, stage_source)
    else:
        shutil.copy2(source, stage_source)

    if " " in str(stage_source):
        raise RuntimeError(
            "Temporary staging path still contains spaces: " + str(stage_source)
        )

    return stage_dir, stage_source


def main():
    heading("FOR1SIGHT LOCAL PARQUET -> HDFS SILVER SYNC (V2)")
    print("Project root:", PROJECT_ROOT)
    print("Python executable:", sys.executable)
    print("Python version:", sys.version.split()[0])
    print("Local source:", LOCAL_PARQUET)
    print("Local source exists:", LOCAL_PARQUET.exists())
    print("Local source is directory:", LOCAL_PARQUET.is_dir() if LOCAL_PARQUET.exists() else "N/A")
    print("Legacy HDFS path (read-only):", LEGACY_HDFS_PATH)
    print("New HDFS destination:", NEW_HDFS_PATH)

    if not LOCAL_PARQUET.exists():
        expected_dir = LOCAL_PARQUET.parent
        raise FileNotFoundError(
            "Local Parquet source was not found.\n"
            f"Expected: {LOCAL_PARQUET}\n"
            f"Parent exists: {expected_dir.exists()}\n"
            f"Files in parent: {[p.name for p in expected_dir.iterdir()] if expected_dir.exists() else 'parent missing'}"
        )

    source_size = get_size_bytes(LOCAL_PARQUET)
    print("Source size (MB):", round(source_size / (1024 * 1024), 3))

    stage_dir = None
    spark = None
    try:
        heading("1. CREATE TEMPORARY PATH WITHOUT SPACES")
        stage_dir, stage_source = stage_without_spaces(LOCAL_PARQUET)
        stage_uri = stage_source.resolve().as_uri()
        print("Temporary staging directory:", stage_dir)
        print("Temporary Spark source:", stage_uri)
        print("Staged size (MB):", round(get_size_bytes(stage_source) / (1024 * 1024), 3))

        spark = (
            SparkSession.builder
            .appName("For1Sight_Safe_HDFS_Silver_Sync_v2")
            .config("spark.driver.memory", "4g")
            .config("spark.hadoop.fs.defaultFS", HDFS_DEFAULT_FS)
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")

        heading("2. READ AND VALIDATE LOCAL PARQUET COPY")
        local_df = spark.read.parquet(stage_uri)
        local_count = local_df.count()
        print("Local source row count:", f"{local_count:,}")
        print("Columns:", local_df.columns)

        missing = sorted(REQUIRED_COLUMNS - set(local_df.columns))
        if missing:
            raise ValueError(
                "Required columns missing: " + ", ".join(missing)
            )
        if local_count <= 0:
            raise ValueError("Local Parquet dataset is empty. Sync cancelled.")

        print("\nRows by year:")
        local_df.groupBy("year").count().orderBy("year").show(50, truncate=False)
        print("\nRows by session type:")
        local_df.groupBy("session_type").count().orderBy("session_type").show(50, truncate=False)

        heading("3. CHECK LEGACY HDFS DATASET (READ ONLY)")
        try:
            legacy_df = spark.read.parquet(LEGACY_HDFS_PATH)
            legacy_count = legacy_df.count()
            print("Legacy HDFS row count:", f"{legacy_count:,}")
            print("Legacy HDFS columns:", legacy_df.columns)
        except Exception as exc:
            print("Could not read legacy HDFS dataset:", repr(exc))
            print("Legacy HDFS path remains untouched.")

        heading("4. CHECK NEW HDFS DESTINATION")
        if hdfs_exists(spark, NEW_HDFS_PATH):
            raise FileExistsError(
                "New destination already exists. Refusing to overwrite:\n"
                + NEW_HDFS_PATH
                + "\nStop and share this output so the existing destination can be inspected."
            )
        print("New destination does not exist. Safe to write.")

        heading("5. WRITE DATA TO HDFS SILVER")
        (
            local_df.write
            .mode("errorifexists")
            .partitionBy("year", "session_type")
            .parquet(NEW_HDFS_PATH)
        )
        print("HDFS write completed.")

        heading("6. READ-BACK VALIDATION")
        hdfs_df = spark.read.parquet(NEW_HDFS_PATH)
        hdfs_count = hdfs_df.count()
        print("Local source rows:", f"{local_count:,}")
        print("New HDFS rows:", f"{hdfs_count:,}")

        if hdfs_count != local_count:
            raise RuntimeError(
                f"Row count mismatch: local={local_count:,}, HDFS={hdfs_count:,}. "
                "Do not delete either copy; investigate before retrying."
            )

        print("\nSUCCESS: local and HDFS row counts match.")
        print("New HDFS Silver dataset:", NEW_HDFS_PATH)
        print("Existing legacy HDFS dataset was not modified.")
        print("Original local Parquet dataset was not modified.")

    finally:
        if spark is not None:
            spark.stop()
        if stage_dir is not None and stage_dir.exists():
            try:
                shutil.rmtree(stage_dir)
                print("\nTemporary staging copy removed.")
            except Exception as exc:
                print("\nWARNING: Could not remove temporary staging copy:", repr(exc))
                print("You may remove this temporary folder manually:", stage_dir)


if __name__ == "__main__":
    main()
