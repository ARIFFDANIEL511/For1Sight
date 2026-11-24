
import zipfile
from pathlib import Path
from collections import Counter
import re
import tkinter as tk
from tkinter import filedialog

# ============================================================
# FOR1SIGHT V2.8 - DEPLOYMENT INSPECTION
# Read-only inspection. Does not extract or modify ZIP contents.
# ============================================================

print("=" * 65)
print("FOR1SIGHT V2.8 - DEPLOYMENT INSPECTION")
print("=" * 65)

# Select ZIP file using Windows file picker
root = tk.Tk()
root.withdraw()

zip_path = filedialog.askopenfilename(
    title="Select For1Sight Deployment ZIP",
    filetypes=[("ZIP files", "*.zip")]
)

if not zip_path:
    print("No ZIP file selected.")
    raise SystemExit

zip_path = Path(zip_path)

print("\nSelected file:", zip_path.name)
print("File size (MB):", round(zip_path.stat().st_size / (1024 * 1024), 2))

report = []

def log(text=""):
    print(text)
    report.append(str(text))

try:
    with zipfile.ZipFile(zip_path, "r") as archive:

        # 1. ZIP integrity
        log("\n1. ZIP INTEGRITY")
        bad_file = archive.testzip()

        if bad_file is None:
            log("ZIP integrity: PASSED")
        else:
            log(f"ZIP integrity: FAILED - {bad_file}")

        files = [
            item for item in archive.infolist()
            if not item.is_dir()
        ]

        names = [item.filename for item in files]

        log(f"Total files: {len(files)}")
        log(f"Total uncompressed size (MB): "
            f"{sum(x.file_size for x in files) / (1024*1024):.2f}")

        # 2. File extension summary
        log("\n2. FILE TYPE SUMMARY")

        extensions = Counter(
            Path(name).suffix.lower() or "[no extension]"
            for name in names
        )

        for ext, count in extensions.most_common():
            log(f"{ext}: {count}")

        # 3. Important project files
        log("\n3. IMPORTANT PROJECT FILES")

        keywords = [
            "app.py",
            "dashboard.py",
            "requirements.txt",
            "packages.txt",
            "future_race_predictor.py",
            "config.toml",
            "readme.md"
        ]

        for keyword in keywords:
            matches = [
                name for name in names
                if Path(name).name.lower() == keyword
            ]

            log(f"\n{keyword}:")
            if matches:
                for match in matches:
                    log(f"  {match}")
            else:
                log("  Not found")

        # 4. Streamlit pages
        log("\n4. STREAMLIT PAGES")

        pages = [
            name for name in names
            if (
                "/pages/" in "/" + name.lower()
                and name.lower().endswith(".py")
            )
        ]

        if pages:
            for page in pages:
                log(f"  {page}")
        else:
            log("  No pages folder detected.")

        log(f"Total Python pages: {len(pages)}")

        # 5. ML models
        log("\n5. MACHINE LEARNING MODELS")

        models = [
            name for name in names
            if name.lower().endswith(
                (".joblib", ".pkl", ".pickle")
            )
        ]

        if models:
            for model in models:
                log(f"  {model}")
        else:
            log("  No model files found.")

        # 6. Parquet datasets
        log("\n6. PARQUET DATASETS")

        parquet = [
            name for name in names
            if name.lower().endswith(".parquet")
        ]

        for dataset in parquet:
            log(f"  {dataset}")

        log(f"Total Parquet files: {len(parquet)}")

        # 7. HDFS and PySpark indicators
        log("\n7. BIG DATA COMPONENTS")

        big_data_terms = [
            "hdfs", "spark", "pyspark",
            "warehouse", "parquet"
        ]

        for term in big_data_terms:
            matches = [
                name for name in names
                if term in name.lower()
            ]

            log(f"{term}: {len(matches)} matching paths")

        # 8. Python imports and local path references
        log("\n8. PYTHON DEPENDENCY INSPECTION")

        py_files = [
            item for item in files
            if item.filename.lower().endswith(".py")
        ]

        imports = set()
        local_paths = set()
        hdfs_references = set()

        for item in py_files:
            if item.file_size > 2_000_000:
                continue

            try:
                content = archive.read(item).decode(
                    "utf-8", errors="ignore"
                )
            except Exception:
                continue

            for match in re.findall(
                r"^\s*(?:import|from)\s+([A-Za-z_][\w.]*)",
                content,
                re.MULTILINE
            ):
                imports.add(match.split(".")[0])

            for match in re.findall(
                r"""(?:[A-Za-z]:\\[^"'\\n]+|/home/[^"'\\n]+|/mnt/[^"'\\n]+)""",
                content
            ):
                local_paths.add(match[:180])

            for line in content.splitlines():
                if "hdfs://" in line.lower():
                    hdfs_references.add(line.strip()[:180])

        log("\nDetected Python imports:")
        for name in sorted(imports):
            log(f"  {name}")

        log("\nPossible hardcoded local paths:")
        for path in sorted(local_paths):
            log(f"  {path}")

        log("\nPossible HDFS references:")
        for ref in sorted(hdfs_references):
            log(f"  {ref}")

        # 9. Deployment exclusions
        log("\n9. POTENTIAL LARGE / UNNECESSARY FOLDERS")

        excluded_terms = [
            "raw_hf_data",
            "venv",
            ".venv",
            "__pycache__",
            "checkpoint",
            "spark-warehouse",
            "logs"
        ]

        for term in excluded_terms:
            matches = [
                name for name in names
                if term in name.lower()
            ]

            log(f"{term}: {len(matches)} files")

        log("\n" + "=" * 65)
        log("INSPECTION COMPLETED")
        log("=" * 65)

except zipfile.BadZipFile:
    log("ERROR: The selected file is not a valid ZIP archive.")

except Exception as error:
    log(f"INSPECTION ERROR: {error}")

# Save inspection report beside ZIP
report_path = zip_path.parent / "For1Sight_Inspection_Report.txt"

report_path.write_text(
    "\n".join(report),
    encoding="utf-8"
)

print("\nReport saved to:", report_path)
