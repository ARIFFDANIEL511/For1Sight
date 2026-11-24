"""
For1Sight read-only warehouse and HDFS audit
Run from the For1Sight project root with:
    python for1sight_warehouse_hdfs_audit.py

This script DOES NOT modify, delete, overwrite, or upload any files.
"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

try:
    import pandas as pd
except ImportError:
    pd = None

PROJECT_ROOT = Path(__file__).resolve().parent
WAREHOUSE = PROJECT_ROOT / "for1sight_warehouse"
PARQUET_PATH = WAREHOUSE / "hf_laptimes.parquet"
RAW_ROOT = PROJECT_ROOT / "raw_hf_data"
HDFS_TARGET = "/user/for1sight/warehouse/hf_laptimes.parquet"


def heading(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def run_command(args):
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=30,
            cwd=str(PROJECT_ROOT), shell=False
        )
        print("$", " ".join(args))
        print("Return code:", result.returncode)
        if result.stdout.strip():
            print("STDOUT:\n", result.stdout.strip())
        if result.stderr.strip():
            print("STDERR:\n", result.stderr.strip())
        return result.returncode == 0
    except FileNotFoundError:
        print("Command not found:", args[0])
    except subprocess.TimeoutExpired:
        print("Command timed out:", " ".join(args))
    except Exception as exc:
        print("Command error:", repr(exc))
    return False


def inspect_parquet():
    heading("1. LOCAL PARQUET WAREHOUSE (READ ONLY)")
    print("Project root:", PROJECT_ROOT)
    print("Warehouse exists:", WAREHOUSE.exists())
    print("Expected Parquet path:", PARQUET_PATH)
    print("Parquet path exists:", PARQUET_PATH.exists())

    if not PARQUET_PATH.exists():
        print("RESULT: hf_laptimes.parquet was not found at the expected path.")
        return

    if PARQUET_PATH.is_file():
        print("File size (MB):", round(PARQUET_PATH.stat().st_size / (1024 * 1024), 3))
    else:
        files = list(PARQUET_PATH.rglob("*.parquet"))
        print("Parquet dataset directory; part files:", len(files))
        print("Total part-file size (MB):",
              round(sum(p.stat().st_size for p in files) / (1024 * 1024), 3))

    if pd is None:
        print("Pandas is not installed in this environment; skipped data read.")
        return

    try:
        df = pd.read_parquet(PARQUET_PATH)
        print("Rows:", len(df))
        print("Columns:", list(df.columns))
        print("\nDtypes:")
        print(df.dtypes.to_string())
        for col in ["year", "session_type", "event_name", "driver_code",
                    "drv", "lap", "lap_number", "time", "pos",
                    "compound", "stint", "s1", "s2", "s3"]:
            if col in df.columns:
                values = df[col].dropna()
                unique = values.astype(str).unique()[:20].tolist()
                print(f"\n{col}: non-null={len(values):,}; sample unique={unique}")
        print("\nFirst 3 rows:")
        print(df.head(3).to_string(index=False, max_cols=30))
    except Exception as exc:
        print("PARQUET READ ERROR:", repr(exc))


def inspect_raw_files():
    heading("2. RAW HUGGINGFACE / TRACINGINSIGHTS FILES (READ ONLY)")
    print("Raw folder:", RAW_ROOT)
    print("Raw folder exists:", RAW_ROOT.exists())
    if not RAW_ROOT.exists():
        return

    files = list(RAW_ROOT.rglob("laptimes.json"))
    print("laptimes.json files found:", len(files))
    for p in files[:15]:
        try:
            print("-", p.relative_to(PROJECT_ROOT), "|", round(p.stat().st_size / 1024, 1), "KB")
        except OSError:
            print("-", p)

    if not files:
        return

    sample = files[0]
    print("\nInspecting first file structure:", sample.relative_to(PROJECT_ROOT))
    try:
        with sample.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            print("Top-level keys:", list(payload.keys()))
            for key, value in payload.items():
                if isinstance(value, list):
                    print(f"  {key}: list length={len(value)}; first item={value[:2]}")
                else:
                    print(f"  {key}: type={type(value).__name__}; sample={str(value)[:180]}")
        elif isinstance(payload, list):
            print("Top-level list length:", len(payload))
            if payload and isinstance(payload[0], dict):
                print("First record keys:", list(payload[0].keys()))
                print("First record sample:", str(payload[0])[:1200])
        else:
            print("Top-level JSON type:", type(payload).__name__)
    except Exception as exc:
        print("RAW JSON SAMPLE ERROR:", repr(exc))


def inspect_hdfs():
    heading("3. HDFS CONNECTIVITY AND EXISTING DATA (READ ONLY)")
    hdfs_exe = shutil.which("hdfs")
    print("HDFS executable:", hdfs_exe or "Not found on PATH")

    if not hdfs_exe:
        print("HDFS CLI is unavailable in this Python environment.")
        print("If Hadoop works in another terminal, run this script from that same")
        print("Anaconda Prompt / environment, or add Hadoop bin to PATH.")
        return

    run_command([hdfs_exe, "getconf", "-confKey", "fs.defaultFS"])
    run_command([hdfs_exe, "dfs", "-ls", "/"])
    run_command([hdfs_exe, "dfs", "-ls", "/user/for1sight"])
    run_command([hdfs_exe, "dfs", "-ls", "/user/for1sight/warehouse"])
    run_command([hdfs_exe, "dfs", "-count", "-q", HDFS_TARGET])
    run_command([hdfs_exe, "dfs", "-du", "-h", HDFS_TARGET])


def main():
    heading("FOR1SIGHT WAREHOUSE + HDFS AUDIT")
    print("Python:", sys.version.replace("\n", " "))
    print("Working directory:", Path.cwd())
    print("Script directory (project root assumed):", PROJECT_ROOT)
    print("This is a READ-ONLY audit. No data will be changed.")
    inspect_parquet()
    inspect_raw_files()
    inspect_hdfs()
    heading("AUDIT FINISHED")
    print("Please share the complete terminal output or save it as a .txt file.")


if __name__ == "__main__":
    main()
