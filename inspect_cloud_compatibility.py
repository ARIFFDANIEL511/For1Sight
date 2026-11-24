
from pathlib import Path
import re

# =====================================================
# FOR1SIGHT V2.8 - CLOUD COMPATIBILITY AUDIT
# Read-only source code inspection
# =====================================================

PROJECT_ROOT = Path(
    r"C:\Users\ayepd\Desktop\For1Sight Project"
)

OUTPUT_FILE = PROJECT_ROOT / "Cloud_Compatibility_Report.txt"

EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "__pycache__",
    "node_modules", "raw_hf_data"
}

PATTERNS = {
    "PySpark imports": r"pyspark|SparkSession",
    "HDFS references": r"hdfs://|HDFS_URI",
    "Windows paths": r"[A-Za-z]:\\\\|C:\\\\",
    "Localhost references": r"localhost|127\.0\.0\.1",
    "Parquet access": r"read_parquet|\.parquet|parquet",
    "Model loading": r"joblib\.load|pickle\.load",
    "FastF1 cache": r"Cache\.enable_cache|enable_cache",
    "Relative paths": r"Path\(__file__\)|os\.path\.join",
}

print("=" * 65)
print("FOR1SIGHT V2.8 - CLOUD COMPATIBILITY AUDIT")
print("=" * 65)

if not PROJECT_ROOT.exists():
    raise FileNotFoundError(PROJECT_ROOT)

results = []

python_files = []

for path in PROJECT_ROOT.rglob("*.py"):
    if any(part in EXCLUDED_DIRS for part in path.parts):
        continue
    python_files.append(path)

print("\nPython files detected:", len(python_files))

for path in python_files:

    try:
        content = path.read_text(
            encoding="utf-8",
            errors="ignore"
        )
    except Exception:
        continue

    for category, pattern in PATTERNS.items():

        for line_number, line in enumerate(
            content.splitlines(), start=1
        ):

            if re.search(pattern, line, re.IGNORECASE):

                results.append({
                    "file": str(path.relative_to(PROJECT_ROOT)),
                    "category": category,
                    "line": line_number,
                    "code": line.strip()[:250]
                })

with OUTPUT_FILE.open("w", encoding="utf-8") as f:

    f.write("FOR1SIGHT V2.8 - CLOUD COMPATIBILITY REPORT\n")
    f.write("=" * 65 + "\n")

    for category in PATTERNS:

        matches = [
            item for item in results
            if item["category"] == category
        ]

        f.write(f"\n{category}: {len(matches)} matches\n")
        f.write("-" * 65 + "\n")

        for item in matches:
            f.write(
                f"{item['file']} | Line {item['line']}\n"
                f"  {item['code']}\n\n"
            )

print("\nAUDIT SUMMARY")

for category in PATTERNS:
    count = sum(
        item["category"] == category
        for item in results
    )
    print(f"{category}: {count}")

print("\nReport generated:")
print(OUTPUT_FILE)
print("\nAudit completed.")
