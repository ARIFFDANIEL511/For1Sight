"""For1Sight - diagnose the cloud telemetry back-end outside Streamlit.

Run from the project folder:  python diagnose_telemetry_cloud.py
Optionally choose the case:   python diagnose_telemetry_cloud.py 2024 abu_dhabi_grand_prix ALO HAM 10
"""
import os
import sys
import traceback

os.environ.setdefault("FOR1SIGHT_TELEMETRY_BACKEND", "cloud")
os.environ.setdefault("FOR1SIGHT_HF_DATASET", "AyepIsHere/for1sight-race-telemetry")

import pandas as pd
import pyarrow
import huggingface_hub
import telemetry_reader as T

print("pandas", pd.__version__, "| pyarrow", pyarrow.__version__, "| huggingface_hub", huggingface_hub.__version__)
print("backend:", T.BACKEND, "| dataset:", T.HF_DATASET, "| gold dir exists:", T.LOCAL_GOLD_DIR.exists())


def step(name, fn):
    print(f"\n=== {name}")
    try:
        result = fn()
        if isinstance(result, pd.DataFrame):
            print(f"OK - {len(result)} rows, columns: {list(result.columns)}")
            print(result.head(3).to_string())
        else:
            print("OK -", result)
        return result
    except Exception:
        traceback.print_exc()  # prints the full chain, including the original cause
        return None


if len(sys.argv) == 6:
    year, slug, a, b, lap = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
else:
    pairs = step("Gold pairs (2024, all)", lambda: T._read_local_gold(T.LOCAL_PAIRS_DIR, 2024))
    if pairs is None or pairs.empty:
        sys.exit("Cannot continue without Gold pairs.")
    row = pairs.iloc[0]
    year, slug, a, b = 2024, str(row["event_slug"]), str(row["driver_a"]), str(row["driver_b"])
    print("common_laps value type:", type(row["common_laps"]).__name__, "->", str(row["common_laps"])[:60])
    lap = 10

print(f"\nTest case: {year} {slug} {a} vs {b}, lap {lap}")
step("Gold pair candidates", lambda: T._cloud_pair_candidates(year, a, b))
step("Gold audit", lambda: T._cloud_audit(year, slug, a, b, lap))
step("Hugging Face partition download", lambda: T._download_silver_partition(year, slug))
step("Silver lap samples", lambda: T._cloud_lap_samples(year, slug, a, b, lap))
