
import zipfile
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

zip_path = filedialog.askopenfilename(
    title="Select For1Sight Deployment ZIP",
    filetypes=[("ZIP files", "*.zip")]
)

if not zip_path:
    raise SystemExit("No ZIP selected.")

report = []

def add(text):
    report.append(str(text))
    print(text)

with zipfile.ZipFile(zip_path, "r") as z:
    names = z.namelist()

    # Read requirements
    add("=" * 60)
    add("FOR1SIGHT DEPENDENCY INSPECTION")
    add("=" * 60)

    requirements = [
        n for n in names
        if Path(n).name.lower() == "requirements.txt"
    ]

    for name in requirements:
        add(f"\nREQUIREMENTS: {name}")
        add(z.read(name).decode("utf-8", errors="replace"))

    # Inspect main Python files
    targets = [
        n for n in names
        if n.endswith(".py")
        and (
            Path(n).name == "dashboard.py"
            or "/pages/" in n.lower()
            or Path(n).name in [
                "future_race_predictor.py",
                "for1sight_shared.py",
                "telemetry_reader.py",
                "telemetry_replay.py"
            ]
        )
    ]

    add("\n" + "=" * 60)
    add("PYTHON FILE IMPORTS AND PATH REFERENCES")
    add("=" * 60)

    for name in targets:
        content = z.read(name).decode(
            "utf-8", errors="replace"
        )

        add(f"\n--- {name} ---")

        for line in content.splitlines():
            line = line.strip()

            if (
                line.startswith("import ")
                or line.startswith("from ")
                or "hdfs://" in line.lower()
                or "C:\\" in line
                or "spark" in line.lower()
                and ("start" in line.lower() or "master" in line.lower())
            ):
                add(line[:250])

# Save report beside the ZIP
output = Path(zip_path).parent / "For1Sight_Dependency_Report.txt"

output.write_text(
    "\n".join(report),
    encoding="utf-8"
)

print("\nReport saved:", output)
